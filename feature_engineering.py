from __future__ import annotations

import inspect
import logging
import pickle
from functools import reduce
from textwrap import dedent
import time
from typing import Any, Callable, Optional, TYPE_CHECKING, Union
import re
import sys

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pandas as pd
import duckdb
from pyarrow.flight import FlightClient, FlightCallOptions, FlightDescriptor, Ticket
from ibis.backends.duckdb import Backend as DuckdbBackend
from datafusion import udaf, Accumulator, SessionContext
from tqdm import tqdm
from turboml.common.internal import TbPyArrow
import cloudpickle
import base64

if TYPE_CHECKING:
    import ibis
    from ibis.expr.types.relations import Table as IbisTable

from .util import risingwave_type_to_pyarrow, get_imports_used_in_function
from .models import (
    AggregateFeatureSpec,
    BackEnd,
    Dataset,
    IbisFeatureMaterializationRequest,
    FeatureMaterializationRequest,
    FetchFeatureRequest,
    SqlFeatureSpec,
    UdfFeatureSpec,
    UdfFunctionSpec,
    TimestampRealType,
    DuckDbVarcharType,
    RisingWaveVarcharType,
    TimestampQuery,
    UdafFunctionSpec,
    UdafFeatureSpec,
    IbisFeatureSpec,
    FeatureGroup,
    RwEmbeddedUdafFunctionSpec,
)
from .sources import (
    DataSource,
    FileSource,
    PostgresSource,
    FeatureGroupSource,
    S3Config,
)
from .api import api
from .env import CONFIG

logger = logging.getLogger("turboml.feature_engineering")


def get_timestamp_formats() -> list[str]:
    """get the possible timestamp format strings

    Returns:
        list[str]: list of format strings
    """
    return [enum.value for enum in RisingWaveVarcharType] + [
        enum.name for enum in TimestampRealType
    ]


def convert_timestamp(
    timestamp_column_name: str, timestamp_type: str
) -> tuple[str, str]:
    """converts a timestamp string to a timestamp query for usage in db

    Args:
        timestamp_column_name (str): column name for the timestamp_query
        timestamp_type (str): It must be one of real or varchar types

    Raises:
        Exception: If a valid timestamp type is not selected throws an exception

    Returns:
        str: timestamp_query
    """
    for enum in RisingWaveVarcharType:
        if timestamp_type == enum.value:
            return (
                f"to_timestamp({timestamp_column_name}, '{RisingWaveVarcharType[enum.name].value}')",
                f"try_strptime({timestamp_column_name}, '{DuckDbVarcharType[enum.name].value}')",
            )
    if timestamp_type == "epoch_seconds":
        return (
            f"to_timestamp({timestamp_column_name}::double)",
            f"to_timestamp({timestamp_column_name}::double)",
        )
    if timestamp_type == "epoch_milliseconds":
        return (
            f"to_timestamp({timestamp_column_name}::double/1000)",
            f"to_timestamp({timestamp_column_name}::double/1000)",
        )
    raise Exception("Please select a valid option")


def retrieve_features(dataset_id: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieve all materialized features for the given dataset and dataframe containing raw data as a dataframe.

    Args:
        dataset_id (str): The dataset user wants to explore
        df (pd.DataFrame): The dataframe of raw data

    Returns:
        pandas.DataFrame: The dataframe of the dataset features
    """
    try:
        arrow_table = pa.Table.from_pandas(df)

        dataset_name = dataset_id.encode("utf8")
        descriptor = FlightDescriptor.for_path(dataset_name)
        options = FlightCallOptions(headers=api.arrow_headers, timeout=120)
        flight_client = FlightClient(CONFIG.FEATURE_SERVER_ADDRESS)

        features_table = TbPyArrow._exchange_and_retry(
            flight_client, descriptor, options, arrow_table, max_chunksize=10000
        )

        return TbPyArrow.arrow_table_to_pandas(features_table)

    except Exception as e:
        raise Exception("An error occurred while fetching features") from e


def get_features(
    dataset_id: str,
    limit: int = -1,
    to_pandas_opts: dict | None = None,
) -> pd.DataFrame:
    """
    Retrieve all materialized features from the given dataset as a dataframe.

    Args:
        dataset_id (str): The dataset_id user wants to explore
        limit (int): Limit the number of rows returned. Default is -1 (no limit).
        to_pandas_opts (dict | None): Options to pass to the `to_pandas` method.
            Refer https://arrow.apache.org/docs/python/generated/pyarrow.Table.html#pyarrow.Table.to_pandas
            for additional information.

    Returns:
        pandas.DataFrame: The dataframe of the dataset features
    """
    try:
        if dataset_id == "":
            raise ValueError("'' is not a valid dataset_id")

        # Small delay before fetching, in case this was called right after a push or feature materialization
        time.sleep(0.2)

        payload = FetchFeatureRequest(dataset_id=dataset_id, limit=limit)
        options = FlightCallOptions(headers=api.arrow_headers, timeout=120)
        ticket = Ticket(payload.model_dump_json())

        flight_client = FlightClient(CONFIG.FEATURE_SERVER_ADDRESS)
        reader = flight_client.do_get(ticket, options)

        features_table: pa.Table = reader.read_all()

        return TbPyArrow.arrow_table_to_pandas(features_table, to_pandas_opts)
    except Exception as e:
        logger.error(f"Feature server: {e!r}")
        if "Dataset does not exist" in str(e):
            raise Exception(f"Dataset `{dataset_id}` does not exist.") from None
        raise Exception(f"An error occurred while fetching features: {e!r}") from e


def _register_udf(
    input_types: list[str],
    result_type: str,
    name: str,
    function_file_contents: str,
    libraries: list[str],
    is_rich_function: bool,
    initializer_arguments: list[str],
    class_name: Optional[str],
    io_threads: Optional[int],
):
    """Add a User-Defined Function (UDF) to the system.

    This function serializes the provided callable function and sends a series of
    requests to register the UDF in the system.

    Args:
        input_types (list[str]): List of input types expected by the UDF.
        result_type (str): The type of result produced by the UDF.
        name (str): Name of the UDF.
        function_file_contents (str): The contents of the python file that contains the UDF to be registered along with the imports used by it.
        libraries (list[str]): List of libraries required by the UDF.
        is_rich_function (bool): Specifies whether the UDF is a rich function, i.e., a class-based function that uses state
        initializer_arguments (list[str]): Arguments passed to the constructor of the rich function.
        class_name (Optional[str]): Name of the class implementing the rich function, required if `is_rich_function` is True.
        io_threads (Optional[int]): Number of I/O threads allocated for the UDF, applicable for rich functions
                                    that involve I/O operations like database or external service lookups.

    Raises:
        Exception: Raises an exception if the initial POST request to create the UDF fails.
        Exception: Raises an exception if registering the UDF with the system fails.
    """
    payload = UdfFunctionSpec(
        name=name,
        input_types=input_types,
        output_type=result_type,
        libraries=libraries,
        function_file_contents=function_file_contents,
        is_rich_function=is_rich_function,
        initializer_arguments=initializer_arguments,
        class_name=class_name,
        io_threads=io_threads,
    )
    api.post(endpoint="register_udf", json=payload.model_dump())


def _register_udaf(
    input_types: list[str],
    result_type: str,
    name: str,
    function_file_contents: str,
):
    """Add a User-Defined Aggregation Function (UDAF) to the system.
    This function serializes the provided callable function and sends a series of
    requests to register the UDAF in the system.
    Args:
        input_types (list[str]): List of input types expected by the UDAF.
        result_type (str): The type of result produced by the UDAF.
        name (str): Name of the UDAF.
        function_file_contents (str): The contents of the python file that contains the UDAF to be registered along with the imports used by it.
    Raises:
        Exception: Raises an exception if the initial POST request to create the UDAF fails.
        Exception: Raises an exception if registering the UDAF with the system fails.
    """

    rw_embedded_udaf_spec = RwEmbeddedUdafFunctionSpec(
        input_types=input_types,
        output_type=result_type,
        function_file_contents=function_file_contents,
    )
    payload = UdafFunctionSpec(name=name, spec=rw_embedded_udaf_spec, libraries=[])
    api.post(endpoint="register_udaf", json=payload.model_dump())


@dataclass
class _UdafFeature:
    spec: UdafFeatureSpec
    function_file_contents: str
    output_dtype: str


@dataclass
class _UdfFeature:
    spec: UdfFeatureSpec
    function_file_contents: str
    output_dtype: np.dtype


def _fetch_datasource(source_name: str):
    datasource_json = api.get(endpoint="datasource", json=source_name).json()
    datasource = DataSource(**datasource_json)
    return datasource


def _get_udfs_from_ibis_table(table, backend_type):
    """
    Extracts UDFs from an Ibis table and returns their details including name, source code,
    output type, and input types.
    """
    from ibis.backends.risingwave import Backend as RisingwaveBackend
    import ibis.expr.operations as ops

    backend = RisingwaveBackend()
    type_mapper = backend.compiler.type_mapper

    udfs = []

    for udf_node in table.op().find(ops.ScalarUDF):
        source_lines = dedent(inspect.getsource(udf_node.__func__)).splitlines()
        source_code = "\n".join(
            line for line in source_lines if not line.startswith("@")
        )

        result_type = type_mapper.to_string(udf_node.dtype)
        if backend_type == BackEnd.Flink:
            source_code = (
                "from pyflink.table.udf import udf\n\n"
                + f"@udf(result_type='{result_type}', func_type='general')\n"
                + source_code
            )

        fn_imports = get_imports_used_in_function(udf_node.__func__)
        source_code = f"{fn_imports}\n{source_code}"

        udf_function_spec = UdfFunctionSpec(
            name=udf_node.__func_name__,
            input_types=[type_mapper.to_string(arg.dtype) for arg in udf_node.args],
            output_type=result_type,
            libraries=[],
            function_file_contents=source_code,
            is_rich_function=False,
            initializer_arguments=[],
            class_name=None,
            io_threads=None,
        )

        udfs.append(udf_function_spec)
    return udfs


class IbisFeatureEngineering:
    """
    A class for performing feature engineering using Ibis with various data sources.

    Provides methods to set up configurations and retrieve Ibis tables
    for different types of data sources, such as S3, PostgreSQL.
    """

    def __init__(self) -> None:
        from ibis.backends.risingwave import Backend as RisingwaveBackend
        from ibis.backends.flink import Backend as FlinkBackend

        self._risingwave_backend = RisingwaveBackend()
        duckdb_backend = DuckdbBackend()
        duckdb_backend.do_connect()
        self._duckdb_backend = duckdb_backend
        self._flink_backend = FlinkBackend()

    @staticmethod
    def _format_s3_endpoint(endpoint: str) -> str:
        return re.sub(r"^https?://", "", endpoint)

    def _setup_s3_config(self, s3_config: S3Config) -> None:
        """
        Configure S3 settings for DuckDB, ensuring compatibility with MinIO and AWS S3.
        """
        duckdb_con = self._duckdb_backend.con

        duckdb_con.sql(f"SET s3_region='{s3_config.region}';")

        if s3_config.access_key_id:
            duckdb_con.sql(f"SET s3_access_key_id='{s3_config.access_key_id}';")
        if s3_config.secret_access_key:
            duckdb_con.sql(f"SET s3_secret_access_key='{s3_config.secret_access_key}';")

        if s3_config.endpoint and not s3_config.endpoint.endswith("amazonaws.com"):
            duckdb_con.sql(
                f"SET s3_use_ssl={'true' if s3_config.endpoint.startswith('https') else 'false'};"
            )
            endpoint = self._format_s3_endpoint(s3_config.endpoint)

            duckdb_con.sql(f"SET s3_endpoint='{endpoint}';")
            duckdb_con.sql("SET s3_url_style='path';")

    def _read_file_source(self, file_source: FileSource, name: str):
        if file_source.s3_config is None:
            raise ValueError("S3 configuration is required for reading from S3.")
        self._setup_s3_config(file_source.s3_config)
        path = f"s3://{file_source.s3_config.bucket}/{file_source.path}/*"

        if file_source.format == FileSource.Format.CSV:
            return self._duckdb_backend.read_csv(file_source.path, name)
        elif file_source.format == FileSource.Format.PARQUET:
            return self._duckdb_backend.read_parquet(path, name)
        else:
            raise ValueError(f"Unimplemented file format: {file_source.format}")

    def _read_feature_group(self, feature_group_source: FeatureGroupSource):
        df = get_features(feature_group_source.name, limit=100)
        return self._duckdb_backend.read_in_memory(df, feature_group_source.name)

    def _read_postgres_source(self, postgres_source: PostgresSource):
        uri = (
            f"postgres://{postgres_source.username}:{postgres_source.password}"
            f"@{postgres_source.host}:{postgres_source.port}/{postgres_source.database}"
        )
        return self._duckdb_backend.read_postgres(uri, table_name=postgres_source.table)

    def get_ibis_table(self, data_source: Union[str, DataSource]):
        """
        Retrieve an Ibis table from a data source.

        Args:
            data_source (Union[str, DataSource]): The name of the data source as a string,
                or a `DataSource` object.

        Returns:
            Table: An Ibis table object corresponding to the provided data source.

        Raises:
            ValueError: If the input type is invalid or the data source type is unsupported.
        """
        if isinstance(data_source, str):
            data_source = _fetch_datasource(data_source)
        if not isinstance(data_source, DataSource):
            raise TypeError(
                f"Expected 'data_source' to be a DataSource instance, "
                f"but got {type(data_source).__name__}."
            )
        if data_source.file_source:
            return self._read_file_source(data_source.file_source, data_source.name)
        elif data_source.feature_group_source:
            return self._read_feature_group(data_source.feature_group_source)
        elif data_source.postgres_source:
            return self._read_postgres_source(data_source.postgres_source)
        else:
            raise ValueError(
                f"Unsupported data source type for {data_source.name}"
            ) from None

    def materialize_features(
        self,
        table: IbisTable,
        feature_group_name: str,
        key_field: str,
        backend: BackEnd,
        primary_source_name: str,
    ):
        """
        Materialize features into a specified feature group using the selected backend.

        This method registers the UDFs
        with the backend, and triggers the feature materialization process for a specified
        feature group. The backend can either be Risingwave or Flink.

        Args:
            table (IbisTable): The Ibis table representing the features to be materialized.
            feature_group_name (str): The name of the feature group where the features will be stored.
            key_field (str): The primary key field in the table used to uniquely identify records.
            backend (BackEnd): The backend to use for materialization, either `Risingwave` or `Flink`.
            primary_source_name (str): The name of the primary data source for the feature group.

        Raises:
            Exception: If an error occurs during the UDF registration or feature materialization process.
        """
        try:
            udfs_spec = _get_udfs_from_ibis_table(table, backend)

            [
                api.post(
                    endpoint="register_udf",
                    json=udf.model_copy(
                        update={
                            "function_file_contents": re.sub(
                                r"from pyflink\.table\.udf import udf\n|@udf\(result_type=.*\)\n",
                                "",
                                udf.function_file_contents,
                            )
                        }
                    ).model_dump(),
                )
                for udf in udfs_spec
            ]

            serialized_expr = cloudpickle.dumps(table)
            encoded_table = base64.b64encode(serialized_expr).decode("utf-8")

            payload = IbisFeatureMaterializationRequest(
                feature_group_name=feature_group_name,
                udfs_spec=udfs_spec,
                key_field=key_field,
                backend=backend,
                encoded_table=encoded_table,
                primary_source_name=primary_source_name,
            )
            api.post(
                endpoint="ibis_materialize_features",
                json=payload.model_dump(exclude_none=True),
            )
        except Exception as e:
            raise Exception(f"Error while materializing features: {e!r}") from None

    def get_model_inputs(
        self,
        feature_group_name: str,
        numerical_fields: list | None = None,
        categorical_fields: list | None = None,
        textual_fields: list | None = None,
        imaginal_fields: list | None = None,
        time_field: str | None = None,
    ):
        from .datasets import OnlineInputs

        feature_group = FeatureGroup(
            **api.get(endpoint="featuregroup", json=feature_group_name).json()
        )
        # Normalize
        if numerical_fields is None:
            numerical_fields = []
        if categorical_fields is None:
            categorical_fields = []
        if textual_fields is None:
            textual_fields = []
        if imaginal_fields is None:
            imaginal_fields = []

        dataframe = get_features(feature_group_name)

        for field in (
            numerical_fields + categorical_fields + textual_fields + imaginal_fields
        ):
            if field not in dataframe.columns:
                raise ValueError(f"Field '{field}' is not present in the dataset.")
        if time_field is not None:
            if time_field not in dataframe.columns:
                raise ValueError(f"Field '{time_field}' is not present in the dataset.")

        return OnlineInputs(
            dataframe=dataframe,
            key_field=feature_group.key_field,
            numerical_fields=numerical_fields,
            categorical_fields=categorical_fields,
            textual_fields=textual_fields,
            imaginal_fields=imaginal_fields,
            time_field=time_field,
            dataset_id=feature_group_name,
        )


class TurboMLScalarFunction:
    def __init__(self, name=None, io_threads=None):
        self.name = name
        self.io_threads = io_threads

    def func(self, *args):
        raise NotImplementedError("subclasses must implement func")


LocalFeatureTransformer = Callable[[pd.DataFrame], pd.DataFrame]


class LocalFeatureEngineering:
    def __init__(self, features_df: pd.DataFrame):
        self.local_features_df = features_df
        self.pending_sql_features: dict[str, SqlFeatureSpec] = {}
        self.pending_ibis_feature: ibis.Table | None = None
        self.pending_aggregate_features: dict[str, AggregateFeatureSpec] = {}
        self.pending_udf_features: dict[str, _UdfFeature] = {}
        self.pending_udaf_features: dict[str, _UdafFeature] = {}
        self.timestamp_column_format: dict[str, str] = {}
        self.pending_feature_transformations: dict[str, LocalFeatureTransformer] = {}

    def clone_with_df(self, feature_df: pd.DataFrame):
        clone = LocalFeatureEngineering(feature_df)
        clone.pending_sql_features = self.pending_sql_features.copy()
        clone.pending_ibis_feature = self.pending_ibis_feature
        clone.pending_aggregate_features = self.pending_aggregate_features.copy()
        clone.pending_udf_features = self.pending_udf_features.copy()
        clone.pending_udaf_features = self.pending_udaf_features.copy()
        clone.timestamp_column_format = self.timestamp_column_format.copy()
        clone.pending_feature_transformations = (
            self.pending_feature_transformations.copy()
        )
        clone._update_input_df(feature_df)
        return clone

    def _update_input_df(self, df: pd.DataFrame) -> None:
        self.all_materialized_features_df = df
        self.local_features_df = df.copy()
        for feature_name, transformer in self.pending_feature_transformations.items():
            if feature_name in df.columns:
                # TODO: A method to drop features could be nice to have
                raise ValueError(
                    f"Feature '{feature_name}' now exists in upstream materialized features"
                )
            self.local_features_df = transformer(self.local_features_df)

    def register_timestamp(self, column_name: str, format_type: str) -> None:
        if format_type not in get_timestamp_formats():
            raise ValueError(
                f"Choose only the timestamp formats in {get_timestamp_formats()}"
            )
        if column_name in self.timestamp_column_format and (
            (registered_format := self.timestamp_column_format[column_name])
            != format_type
        ):
            raise Exception(
                f"The timestamp is already registered with a different format={registered_format}"
            )
        self.timestamp_column_format[column_name] = format_type

    def _get_timestamp_query(self, timestamp_column: str) -> tuple[str, str]:
        try:
            timestamp_format = self.timestamp_column_format[timestamp_column]
            timestamp_query_rw, timestamp_query_ddb = convert_timestamp(
                timestamp_column_name=timestamp_column,
                timestamp_type=timestamp_format,
            )
            return timestamp_query_rw, timestamp_query_ddb
        except Exception as e:
            raise Exception(
                f"Please register the timestamp column using `register_timestamp()` error caused by {e!r}"
            ) from None

    def create_sql_features(self, sql_definition: str, new_feature_name: str) -> None:
        """
        sql_definition: str
            The SQL query you want to apply on the columns of the dataframe
            Eg. "transactionAmount + localHour"

        new_feature_name: str
            The name of the new feature column
        """
        if new_feature_name in self.local_features_df:
            raise ValueError(f"Feature '{new_feature_name}' already exists")

        pattern = r'"([^"]+)"'
        matches = re.findall(pattern, sql_definition)
        cleaned_operands = [operand.strip('"') for operand in matches]
        result_string = re.sub(
            pattern, lambda m: cleaned_operands.pop(0), sql_definition
        )

        def feature_transformer(dataframe):
            query = f"SELECT {result_string} AS {new_feature_name} FROM dataframe"
            result_df = duckdb.sql(query).df()
            return dataframe.assign(**{new_feature_name: result_df[new_feature_name]})

        self.local_features_df = feature_transformer(self.local_features_df)
        self.pending_feature_transformations[new_feature_name] = feature_transformer
        self.pending_sql_features[new_feature_name] = SqlFeatureSpec(
            feature_name=new_feature_name,
            sql_spec=sql_definition,
        )

    def create_ibis_features(self, table: ibis.Table) -> None:
        """
        Processes an Ibis table and creates features by executing the table query.

        This method verifies whether the provided Ibis table is derived from an in-memory
        table that corresponds to the current dataset. It then connects to a DuckDB backend
        and executes the table query.

        Parameters:
            table (ibis.Table):
                The Ibis table that contains the feature transformations to be executed.

        Raises:
            AssertionError:
                If the provided Ibis table is not derived from an in-memory table associated
                with the current dataset.
        """
        if (local_feats := len(self.pending_feature_transformations)) > 0:
            raise Exception(
                f"Can't create ibis features with other features: {local_feats} local features exist"
            )
        try:
            con = DuckdbBackend()
            con.do_connect()
            self.local_features_df = con.execute(table)
            self.pending_ibis_feature = table
        except Exception as e:
            raise Exception("An error occurred while creating ibis features") from e

    def create_aggregate_features(
        self,
        column_to_operate: str,
        column_to_group: str,
        operation: str,
        new_feature_name: str,
        timestamp_column: str,
        window_duration: float,
        window_unit: str,
    ) -> None:
        """
        column_to_operate: str
            The column to count

        column_to_group: str
            The column to group by

        operation: str
            The operation to perform on the column, one of ["SUM", "COUNT", "AVG", "MAX", "MIN"]

        new_feature_name: str
            The name of the new feature

        time_column: str
            The column representing time or timestamp for windowing

        window_duration: float
            The numeric duration of the window (e.g. 5, 1.1, 24 etc)

        window_unit: str
            The unit of the window, one of ["seconds", "minutes", "hours", "days", "weeks", "months", "years"]
        """
        if new_feature_name in self.local_features_df:
            raise ValueError(f"Feature '{new_feature_name}' already exists")

        if window_unit not in [
            "seconds",
            "minutes",
            "hours",
            "days",
            "weeks",
            "months",
            "years",
        ]:
            raise Exception(
                """Window unit should be one of ["seconds", "minutes", "hours", "days", "weeks", "months", "years"]"""
            )
        if window_unit == "years":
            window_unit = "days"
            window_duration = window_duration * 365
        if window_unit == "months":
            window_unit = "days"
            window_duration = window_duration * 30
        if window_unit == "weeks":
            window_unit = "days"
            window_duration = window_duration * 7
        if window_unit == "days":
            window_unit = "hours"
            window_duration = window_duration * 24

        window_duration_with_unit = str(window_duration) + " " + window_unit
        _, timestamp_query_ddb = self._get_timestamp_query(
            timestamp_column=timestamp_column
        )

        def feature_transformer(dataframe):
            return duckdb.sql(
                f"""
            SELECT *, {operation}({column_to_operate}) OVER win AS {new_feature_name}
            FROM dataframe
            WINDOW win AS (
                PARTITION BY {column_to_group}
                ORDER BY {timestamp_query_ddb}
                RANGE BETWEEN INTERVAL {window_duration_with_unit} PRECEDING
                        AND CURRENT ROW)"""
            ).df()

        self.local_features_df = feature_transformer(self.local_features_df)
        self.pending_feature_transformations[new_feature_name] = feature_transformer
        self.pending_aggregate_features[new_feature_name] = AggregateFeatureSpec(
            feature_name=new_feature_name,
            column=column_to_operate,
            aggregation_function=operation,
            group_by_columns=[column_to_group],
            interval=window_duration_with_unit,
            timestamp_column=timestamp_column,
        )

    def create_rich_udf_features(
        self,
        new_feature_name: str,
        argument_names: list[str],
        class_name: str,
        function_name: str,
        class_file_contents: str,
        libraries: list[str],
        dev_initializer_arguments: list[str],
        prod_initializer_arguments: list[str],
        io_threads=None,
    ) -> None:
        import pandas as pd

        if new_feature_name in self.local_features_df:
            raise ValueError(f"Feature '{new_feature_name}' already exists")

        def feature_transformer(dataframe):
            main_globals = sys.modules["__main__"].__dict__
            exec(class_file_contents, main_globals)
            obj = main_globals[class_name](*dev_initializer_arguments)

            tqdm.pandas(desc="Progress")

            new_col = dataframe.progress_apply(
                lambda row: obj.func(*[row[col] for col in argument_names]),
                axis=1,
            )
            return dataframe.assign(**{new_feature_name: new_col})

        transformed_df = feature_transformer(self.local_features_df)
        out_col = transformed_df[new_feature_name]
        if not isinstance(out_col, pd.Series):
            raise ValueError(
                f"UDF {function_name} must return a scalar value"
            ) from None
        out_type = out_col.dtype
        if not isinstance(out_type, np.dtype):
            raise ValueError(
                f"UDF {function_name} must return a scalar value, instead got {out_type}"
            ) from None

        self.local_features_df = transformed_df
        self.pending_feature_transformations[new_feature_name] = feature_transformer
        self.pending_udf_features[new_feature_name] = _UdfFeature(
            spec=UdfFeatureSpec(
                function_name=function_name,
                arguments=argument_names,
                libraries=libraries,
                feature_name=new_feature_name,
                is_rich_function=True,
                io_threads=io_threads,
                class_name=class_name,
                initializer_arguments=prod_initializer_arguments,
            ),
            function_file_contents=class_file_contents,
            output_dtype=out_type,
        )

    def create_udf_features(
        self,
        new_feature_name: str,
        argument_names: list[str],
        function_name: str,
        function_file_contents: str,
        libraries: list[str],
    ) -> None:
        """
        new_feature_name: str
            The name of the new feature column

        argument_names: list[str]
            The list of column names to pass as argument to the function

        function_name: str
            The function name under which this UDF is registered

        function_file_contents: str
            The contents of the python file that contains the function along with the imports used by it

        libraries: list[str]
            The list of libraries that need to be installed to run the function

        """
        import pandas as pd

        if new_feature_name in self.local_features_df:
            raise ValueError(f"Feature '{new_feature_name}' already exists")

        def feature_transformer(dataframe):
            if len(dataframe) == 0:
                dataframe[new_feature_name] = pd.Series(dtype=object)
                return dataframe
            local_ctxt = {}
            exec(function_file_contents, local_ctxt)
            new_col = dataframe.apply(
                lambda row: local_ctxt[function_name](
                    # TODO: should we pass as kwargs instead of relying on order?
                    *[row[col] for col in argument_names]
                ),
                axis=1,
            )
            # DC_Change :
            # new_col = dataframe.apply(
            #     lambda row: local_ctxt[function_name](
            #         **{col: row[col] for col in argument_names}  # Changed to kwargs
            #     ),
            #     axis=1,
            # )
            if not isinstance(new_col, pd.Series):
                print(new_col.head())
                raise ValueError(f"UDF {function_name} must return a scalar value")
            return dataframe.assign(**{new_feature_name: new_col})

        transformed_df = feature_transformer(self.local_features_df)
        out_col = transformed_df[new_feature_name]
        if not isinstance(out_col, pd.Series):
            raise ValueError(
                f"UDF {function_name} must return a scalar value"
            ) from None
        out_type = out_col.dtype
        if not isinstance(out_type, np.dtype):
            raise ValueError(
                f"UDF {function_name} must return a scalar value, instead got {out_type}"
            ) from None

        self.local_features_df = transformed_df
        self.pending_feature_transformations[new_feature_name] = feature_transformer
        self.pending_udf_features[new_feature_name] = _UdfFeature(
            spec=UdfFeatureSpec(
                function_name=function_name,
                arguments=argument_names,
                libraries=libraries,
                feature_name=new_feature_name,
                is_rich_function=False,
                io_threads=None,
                class_name=None,
                initializer_arguments=[],
            ),
            function_file_contents=function_file_contents,
            output_dtype=out_type,
        )

    def _create_dynamic_udaf_class(self, local_ctxt, return_type):
        class DynamicUDAFClass(Accumulator):
            def __init__(self):
                self._state = pickle.dumps(local_ctxt["create_state"]())

            def update(self, *values):  # Should values be pa.Array?
                for row in zip(*values, strict=True):
                    row_values = [col.as_py() for col in row]
                    state = pickle.loads(self._state)
                    self._state = pickle.dumps(
                        local_ctxt["accumulate"](state, *row_values)
                    )

            def merge(self, states: pa.Array):
                deserialized_values = []
                for list_array in states:
                    for pickled_value in list_array:
                        deserialized_values.append(pickle.loads(pickled_value.as_py()))
                merged_value = reduce(local_ctxt["merge_states"], deserialized_values)
                self._state = pickle.dumps(merged_value)

            def state(self) -> pa.Array:
                return pa.array([[self._state]], type=pa.list_(pa.binary()))

            def evaluate(self) -> pa.Scalar:
                return pa.scalar(
                    local_ctxt["finish"](pickle.loads(self._state)), type=return_type
                )

        return DynamicUDAFClass

    def create_udaf_features(
        self,
        new_feature_name: str,
        column_to_operate: list[str],
        function_name: str,
        return_type: str,
        function_file_contents: str,
        column_to_group: list[str],
        timestamp_column: str,
        window_duration: float,
        window_unit: str,
    ) -> None:
        """
        new_feature_name: str
            The name of the new feature column

        column_to_operate: list[str]
            The list of column names to pass as argument to the function

        function_name: str
            The function name under which this UDF is registered

        function_file_contents: str
            The contents of the python file that contains the function along with the imports used by it

        return_type: list[str]
            The return type the function

        libraries: list[str]
            The list of libraries that need to be installed to run the function

        """
        if new_feature_name in self.local_features_df:
            raise ValueError(f"Feature '{new_feature_name}' already exists")

        local_ctxt = {}
        exec(function_file_contents, local_ctxt)

        required_functions = [
            "create_state",
            "accumulate",
            "retract",
            "merge_states",
            "finish",
        ]
        missing_functions = [f for f in required_functions if f not in local_ctxt]

        if missing_functions:
            raise AssertionError(
                f"Missing functions in UDAF: {', '.join(missing_functions)}. Functions create_state, "
                f"accumulate, retract, merge_states, and finish must be defined."
            )

        if window_unit not in [
            "seconds",
            "minutes",
            "hours",
            "days",
            "weeks",
            "months",
            "years",
        ]:
            raise Exception(
                """Window unit should be one of ["seconds", "minutes", "hours", "days", "weeks", "months", "years"]"""
            )
        if window_unit == "years":
            window_unit = "days"
            window_duration = window_duration * 365
        if window_unit == "months":
            window_unit = "days"
            window_duration = window_duration * 30
        if window_unit == "weeks":
            window_unit = "days"
            window_duration = window_duration * 7
        if window_unit == "days":
            window_unit = "hours"
            window_duration = window_duration * 24

        window_duration_with_unit = str(window_duration) + " " + window_unit
        _, timestamp_query_ddb = self._get_timestamp_query(
            timestamp_column=timestamp_column
        )

        def feature_transformer(dataframe):
            ctx = SessionContext()
            pa_return_type = risingwave_type_to_pyarrow(return_type)
            arrow_table = pa.Table.from_pandas(dataframe)

            ctx.create_dataframe([arrow_table.to_batches()], name="my_table")

            my_udaf = udaf(
                self._create_dynamic_udaf_class(local_ctxt, pa_return_type),
                [arrow_table[col].type for col in column_to_operate],
                pa_return_type,
                [pa.list_(pa.binary())],
                "stable",
                name=function_name,
            )
            ctx.register_udaf(my_udaf)

            column_args = ", ".join(f'"{item}"' for item in column_to_operate)
            group_by_cols = ", ".join(f'"{item}"' for item in column_to_group)

            sql = f"""
            SELECT *, {function_name}({column_args}) OVER win AS {new_feature_name}
            FROM my_table
            WINDOW win AS (
                PARTITION BY {group_by_cols}
                ORDER BY {timestamp_query_ddb}
                RANGE BETWEEN UNBOUNDED PRECEDING
                        AND CURRENT ROW)"""

            dataframe = ctx.sql(sql)
            return dataframe.to_pandas()

        self.local_features_df = feature_transformer(self.local_features_df)
        self.pending_feature_transformations[new_feature_name] = feature_transformer
        self.pending_udaf_features[new_feature_name] = _UdafFeature(
            spec=UdafFeatureSpec(
                function_name=function_name,
                feature_name=new_feature_name,
                arguments=column_to_operate,
                group_by_columns=column_to_group,
                interval=window_duration_with_unit,
                timestamp_column=timestamp_column,
            ),
            function_file_contents=function_file_contents,
            output_dtype=return_type,
        )


class FeatureEngineering:
    def __init__(
        self,
        dataset_id: str,
        features_df: pd.DataFrame,
        local_fe: LocalFeatureEngineering | None = None,
    ):
        self.dataset_id = dataset_id
        self._sync_feature_information()
        self.all_materialized_features_df = features_df.copy()

        self.local_fe = (
            LocalFeatureEngineering(features_df) if local_fe is None else local_fe
        )
        self.local_fe.timestamp_column_format = {
            **self.timestamp_column_format,
            **self.local_fe.timestamp_column_format,
        }

    def _sync_feature_information(self):
        dataset_json = api.get(endpoint=f"dataset?dataset_id={self.dataset_id}").json()
        dataset = Dataset(**dataset_json)
        self.sql_feats = dataset.sql_feats
        self.agg_feats = dataset.agg_feats
        self.udf_feats = dataset.udf_feats
        self.udaf_feats = dataset.udaf_feats
        self.ibis_feats = dataset.ibis_feats
        self.timestamp_column_format = dataset.timestamp_fields

    @staticmethod
    def inherit_from_local(fe: LocalFeatureEngineering, dataset_id: str):
        return FeatureEngineering(dataset_id, fe.local_features_df, fe)

    @property
    def local_features_df(self) -> pd.DataFrame:
        return self.local_fe.local_features_df

    def _update_input_df(self, df: pd.DataFrame) -> None:
        self._sync_feature_information()
        self.local_fe._update_input_df(df.copy())
        self.all_materialized_features_df = df.copy()

    def get_local_features(self) -> pd.DataFrame:
        return self.local_fe.local_features_df

    def get_materialized_features(self) -> pd.DataFrame:
        return self.all_materialized_features_df

    def register_timestamp(self, column_name: str, format_type: str) -> None:
        self.local_fe.register_timestamp(column_name, format_type)

    def _get_timestamp_query(self, timestamp_column: str) -> tuple[str, str]:
        return self.local_fe._get_timestamp_query(timestamp_column)

    def create_sql_features(self, sql_definition: str, new_feature_name: str) -> None:
        """
        sql_definition: str
            The SQL query you want to apply on the columns of the dataframe
            Eg. "transactionAmount + localHour"

        new_feature_name: str
            The name of the new feature column
        """
        self.local_fe.create_sql_features(sql_definition, new_feature_name)

    def create_ibis_features(self, table: ibis.Table) -> None:
        """
        Processes an Ibis table and creates features by executing the table query.

        This method verifies whether the provided Ibis table is derived from an in-memory
        table that corresponds to the current dataset. It then connects to a DuckDB backend
        and executes the table query.

        Parameters:
            table (ibis.Table):
                The Ibis table that contains the feature transformations to be executed.

        Raises:
            AssertionError:
                If the provided Ibis table is not derived from an in-memory table associated
                with the current dataset.
        """
        other_feats = (
            self.sql_feats.keys() | self.agg_feats.keys() | self.udf_feats.keys()
        )
        if len(other_feats) > 0:
            raise Exception(
                f"Can't create ibis features with other features: {other_feats} non-ibis features exist"
            )
        self.local_fe.create_ibis_features(table)

    def create_aggregate_features(
        self,
        column_to_operate: str,
        column_to_group: str,
        operation: str,
        new_feature_name: str,
        timestamp_column: str,
        window_duration: float,
        window_unit: str,
    ) -> None:
        """
        column_to_operate: str
            The column to count

        column_to_group: str
            The column to group by

        operation: str
            The operation to perform on the column, one of ["SUM", "COUNT", "AVG", "MAX", "MIN"]

        new_feature_name: str
            The name of the new feature

        time_column: str
            The column representing time or timestamp for windowing

        window_duration: float
            The numeric duration of the window (e.g. 5, 1.1, 24 etc)

        window_unit: str
            The unit of the window, one of ["seconds", "minutes", "hours", "days", "weeks", "months", "years"]
        """
        self.local_fe.create_aggregate_features(
            column_to_operate,
            column_to_group,
            operation,
            new_feature_name,
            timestamp_column,
            window_duration,
            window_unit,
        )

    def create_rich_udf_features(
        self,
        new_feature_name: str,
        argument_names: list[str],
        class_name: str,
        function_name: str,
        class_file_contents: str,
        libraries: list[str],
        dev_initializer_arguments: list[str],
        prod_initializer_arguments: list[str],
        io_threads=None,
    ) -> None:
        self.local_fe.create_rich_udf_features(
            new_feature_name,
            argument_names,
            class_name,
            function_name,
            class_file_contents,
            libraries,
            dev_initializer_arguments,
            prod_initializer_arguments,
            io_threads,
        )

    def create_udf_features(
        self,
        new_feature_name: str,
        argument_names: list[str],
        function_name: str,
        function_file_contents: str,
        libraries: list[str],
    ) -> None:
        """
        new_feature_name: str
            The name of the new feature column

        argument_names: list[str]
            The list of column names to pass as argument to the function

        function_name: str
            The function name under which this UDF is registered

        function_file_contents: str
            The contents of the python file that contains the function along with the imports used by it

        libraries: list[str]
            The list of libraries that need to be installed to run the function

        """
        self.local_fe.create_udf_features(
            new_feature_name,
            argument_names,
            function_name,
            function_file_contents,
            libraries,
        )

    def create_udaf_features(
        self,
        new_feature_name: str,
        column_to_operate: list[str],
        function_name: str,
        return_type: str,
        function_file_contents: str,
        column_to_group: list[str],
        timestamp_column: str,
        window_duration: float,
        window_unit: str,
    ) -> None:
        """
        new_feature_name: str
            The name of the new feature column

        column_to_operate: list[str]
            The list of column names to pass as argument to the function

        function_name: str
            The function name under which this UDF is registered

        function_file_contents: str
            The contents of the python file that contains the function along with the imports used by it

        return_type: list[str]
            The return type the function

        libraries: list[str]
            The list of libraries that need to be installed to run the function

        """
        self.local_fe.create_udaf_features(
            new_feature_name,
            column_to_operate,
            function_name,
            return_type,
            function_file_contents,
            column_to_group,
            timestamp_column,
            window_duration,
            window_unit,
        )

    def _register_timestamps_for_aggregates(self, feature_names: list[str]):
        """Register timestamps for the provided aggregate features. These timestamps can then be used
        for windows in aggregate features.
        """
        try:
            specs = [
                self.local_fe.pending_aggregate_features[feature_name]
                for feature_name in feature_names
            ]
        except KeyError as V:
            raise ValueError(f"Aggregated feature {V} not found") from V

        for spec in specs:
            timestamp_format = self.local_fe.timestamp_column_format[
                spec.timestamp_column
            ]
            payload = TimestampQuery(
                column_name=spec.timestamp_column,
                timestamp_format=timestamp_format,
            )
            api.post(
                endpoint=f"dataset/{self.dataset_id}/register_timestamp",
                json=payload.model_dump(),
            )

    def _register_udfs(self, feature_names: list[str]):
        """
        Register UDFs corresponding to the provided feature names. These UDFs are then used to create features.
        """
        for feature_name in feature_names:
            try:
                udf = self.local_fe.pending_udf_features[feature_name]
            except KeyError as V:
                raise ValueError(f"UDF feature {feature_name} not found") from V

            def pandas_type_to_risingwave_type(pd_type):
                match pd_type:
                    case np.int32:
                        return "INT"
                    case np.int64:
                        return "BIGINT"
                    case np.float32 | np.float64:
                        return "REAL"
                    case np.object_:
                        return "VARCHAR"
                    case _:
                        return "VARCHAR"

            db_dtype = pandas_type_to_risingwave_type(udf.output_dtype)
            dataset_json = api.get(
                endpoint=f"dataset?dataset_id={self.dataset_id}"
            ).json()
            dataset = Dataset(**dataset_json)
            table_columns = {col.name: col.dtype for col in dataset.table_columns}

            # Converts character varying to VARCHAR as that is only supported
            # by RisingWave as of (18.12.2023)
            input_types = [
                (
                    "VARCHAR"
                    if table_columns[col] == "character varying"
                    else table_columns[col]
                )
                for col in udf.spec.arguments
            ]
            _register_udf(
                name=udf.spec.function_name,
                input_types=input_types,
                libraries=udf.spec.libraries,
                result_type=db_dtype,
                function_file_contents=udf.function_file_contents,
                is_rich_function=udf.spec.is_rich_function,
                initializer_arguments=udf.spec.initializer_arguments,
                class_name=udf.spec.class_name,
                io_threads=udf.spec.io_threads,
            )

    def _register_udafs(self, feature_names: list[str]):
        """
        Register UDAFs corresponding to the provided feature names. These UDAFs are then used to create features.
        """
        for feature_name in feature_names:
            try:
                udaf = self.local_fe.pending_udaf_features[feature_name]
            except KeyError as V:
                raise ValueError(f"UDAF feature {feature_name} not found") from V

            db_dtype = udaf.output_dtype
            dataset_json = api.get(
                endpoint=f"dataset?dataset_id={self.dataset_id}"
            ).json()
            dataset = Dataset(**dataset_json)
            table_columns = {col.name: col.dtype for col in dataset.table_columns}

            # Converts character varying to VARCHAR as that is only supported
            # by RisingWave as of (18.12.2023)
            input_types = [
                (
                    "VARCHAR"
                    if table_columns[col] == "character varying"
                    else table_columns[col]
                )
                for col in udaf.spec.arguments
            ]
            timestamp_format = self.local_fe.timestamp_column_format[
                udaf.spec.timestamp_column
            ]
            payload = TimestampQuery(
                column_name=udaf.spec.timestamp_column,
                timestamp_format=timestamp_format,
            )
            api.post(
                endpoint=f"dataset/{self.dataset_id}/register_timestamp",
                json=payload.model_dump(),
            )
            _register_udaf(
                name=udaf.spec.function_name,
                input_types=input_types,
                result_type=db_dtype,
                function_file_contents=udaf.function_file_contents,
            )

    def materialize_ibis_features(self):
        """Send a POST request to the server to perform feature engineering based on the created ibis features.

        Raises:
            Exception: Raised if the server's response has a non-200 status code.
                The exception message will contain details provided by the server.
        """
        if self.local_fe.pending_ibis_feature is None:
            raise ValueError(
                "No pending Ibis features found. Please create features using `create_ibis_features` first."
            )

        table = self.local_fe.pending_ibis_feature
        udfs_spec = _get_udfs_from_ibis_table(table, BackEnd.Risingwave)

        for udf in udfs_spec:
            api.post(endpoint="register_udf", json=udf.model_dump())

        serialized_expr = cloudpickle.dumps(table)
        encoded_table = base64.b64encode(serialized_expr).decode("utf-8")
        ibis_feat_spec = IbisFeatureSpec(
            dataset_id=self.dataset_id,
            encoded_table=encoded_table,
            udfs_spec=udfs_spec,
        )
        payload = FeatureMaterializationRequest(
            dataset_id=self.dataset_id, ibis_feats=ibis_feat_spec
        )
        api.post(endpoint="materialize_features", json=payload.model_dump())

        self.all_materialized_features_df = get_features(self.dataset_id)

    @property
    def all_pending_features_names(self):
        all_pending_features: list[dict[str, Any]] = [
            self.local_fe.pending_sql_features,
            self.local_fe.pending_aggregate_features,
            self.local_fe.pending_udf_features,
            self.local_fe.pending_udaf_features,
        ]
        return [n for features in all_pending_features for n in features.keys()]

    def materialize_features(self, feature_names: list[str]):
        """Send a POST request to the server to perform feature engineering based on the provided timestamp query.

        Raises:
            Exception: Raised if the server's response has a non-200 status code.
                The exception message will contain details provided by the server.
        """
        missing_features = (
            set(feature_names)
            - set(self.local_fe.pending_feature_transformations.keys())
            - set(self.all_materialized_features_df.columns)
        )
        if missing_features:
            raise ValueError(
                f"Feature names {missing_features} are not pending and also do not exist in the dataset"
            )

        aggregates_to_register = [
            k
            for k, v in self.local_fe.pending_aggregate_features.items()
            if k in feature_names
        ]
        udfs_to_register = [
            k
            for k, v in self.local_fe.pending_udf_features.items()
            if k in feature_names
        ]
        udafs_to_register = [
            k
            for k, v in self.local_fe.pending_udaf_features.items()
            if k in feature_names
        ]
        sql_to_register = [
            k
            for k, v in self.local_fe.pending_sql_features.items()
            if k in feature_names
        ]

        self._register_timestamps_for_aggregates(aggregates_to_register)
        self._register_udfs(udfs_to_register)
        self._register_udafs(udafs_to_register)

        payload = FeatureMaterializationRequest(
            dataset_id=self.dataset_id,
            sql_feats=[self.local_fe.pending_sql_features[k] for k in sql_to_register],
            agg_feats=[
                self.local_fe.pending_aggregate_features[k]
                for k in aggregates_to_register
            ],
            udf_feats=[
                self.local_fe.pending_udf_features[k].spec for k in udfs_to_register
            ],
            udaf_feats=[
                self.local_fe.pending_udaf_features[k].spec for k in udafs_to_register
            ],
        )
        try:
            response = api.post(
                endpoint="materialize_features", json=payload.model_dump()
            )
            if response.status_code != 200:
                raise Exception(f"Error from server: {response.text}")
        except Exception as e:
            raise Exception(f"Failed to materialize features: {e!r}") from e
        finally:
            self.all_materialized_features_df = get_features(self.dataset_id)

        # clean up pending features
        for feature_name in feature_names:
            self.local_fe.pending_feature_transformations.pop(feature_name)
            self.local_fe.pending_sql_features.pop(feature_name, None)
            self.local_fe.pending_aggregate_features.pop(feature_name, None)
            self.local_fe.pending_udf_features.pop(feature_name, None)
            self.local_fe.pending_udaf_features.pop(feature_name, None)
