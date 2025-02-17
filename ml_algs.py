from __future__ import annotations
import logging
from abc import ABC
from typing import Optional, TYPE_CHECKING, List
import random
import string
import urllib.parse as urlparse
import re
import json
import os
import time
import base64
import datetime

from google.protobuf import json_format
import pandas as pd
import pyarrow as pa
import pyarrow.flight
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    validator,
    ConfigDict,
    model_validator,
    field_serializer,
    field_validator,
)


from .types import GGUFModelId  # noqa: TCH001

if TYPE_CHECKING:
    from sklearn import metrics

from .default_model_configs import DefaultModelConfigs
from .internal import TbPyArrow
from .api import api
from .models import (
    InputSpec,
    ModelConfigStorageRequest,
    ModelInfo,
    MetricRegistrationRequest,
    EvaluationMetrics,
    MLModellingRequest,
    LabelAssociation,
    LearnerConfig,
    ModelParams,
    ModelPatchRequest,
    ModelDeleteRequest,
    Evaluations,
    ProcessOutput,
)
from .feature_engineering import retrieve_features
from .env import CONFIG
from .protos import output_pb2, metrics_pb2
from .dataloader import StreamType, get_proto_msgs

from turboml.common.pytypes import InputData, OutputData
from turboml.common.pymodel import create_model_from_config, Model as CppModel
from turboml.common.datasets import (
    LocalInputs,
    LocalLabels,
    OnlineInputs,
    OnlineLabels,
    PandasHelpers,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_configs = DefaultModelConfigs
MAX_RETRY_FOR_TEST = 5


def _istest():
    return os.environ.get("TURBOML_AUTO_RETRY", "false") == "true"


def retry_operation(operation, attempts=MAX_RETRY_FOR_TEST, base_delay=4):
    """Retry operation with exponential backoff"""
    original_level = logger.level
    logger.setLevel(logging.DEBUG)
    last_exception = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as e:
            logger.debug(f"Attempt {attempt+1} failed with error: {str(e)}")
            last_exception = e
            delay = base_delay * (2**attempt)
            logger.debug(f"Retrying in {delay} second.")
            time.sleep(delay)
        finally:
            logger.setLevel(original_level)

    logger.setLevel(original_level)
    raise Exception(f"Failed after {attempts} attempts: {str(last_exception)}")


def validate_non_empty(output):
    if not output or len(output) == 0:
        raise Exception("output cannot be empty")
    return output


# converts camelcase string to underscore seperated
def _identity(name):
    return name


def _camel_to_underscore(name):
    if name[0].isupper():
        name = name[0].lower() + name[1:]

    name = re.sub("([A-Z])", lambda match: "_" + match.group(1).lower(), name)
    return name


def _to_camel(string):
    components = string.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def evaluation_metrics() -> list[str]:
    return [enum.value for enum in EvaluationMetrics]


def get_default_parameters(algorithm):
    parameters = json_format.MessageToDict(default_configs.default_configs[algorithm])
    return parameters


def ml_modelling(
    model_name: str,
    model_configs: list[dict],
    label_dataset: str,
    label_field: str,
    dataset_id: str,
    numerical_fields: list[str] | None = None,
    categorical_fields: list[str] | None = None,
    textual_fields: list[str] | None = None,
    imaginal_fields: list[str] | None = None,
    time_field: str | None = None,
    predict_workers: int = 1,
    update_batch_size: int = 64,
    synchronization_method: str = "",
    predict_only=False,
    initial_model_id="",
):
    """Perform machine learning modeling based on specified configurations.

    This function sends a POST request to the server to initiate machine
    learning modeling using the provided parameters.

    Args:
        model_configs (list[dict]): List of model configs (model parameters)
        model_name (str): Name of the machine learning model.
        label_dataset (str): dataset_id related to the label data.
        label_field (str): Name of the column containing label data.
        dataset_id (str): Dataset related to the input data.
        numerical_fields (list[str], optional): List of numeric fields used in the model.
        categorical_fields (list[str], optional): List of categorical fields used in the model.
        textual_fields (list[str], optional): List of textual fields used in the model.
        imaginal_fields (list[str], optional): List of imaginal fields used in the model.
        time_field (str, optional): The time field used in the model configuration.
        predict_workers (int, optional): The number of threads for prediction.
        update_batch_size (int, optional): The update frequency for models.
        synchronization_method (str, optional): Synchronization method to use. One of "" or "_lr".
        predict_only (bool, optional): Should this model only be used for prediction.
        initial_model_id (str, optional): Model id for deploying a batch trained model

    Raises:
        Exception: Raises an exception if the POST request fails, providing details
            from the response JSON.
    """
    if categorical_fields is None:
        categorical_fields = []
    if numerical_fields is None:
        numerical_fields = []
    if textual_fields is None:
        textual_fields = []
    if imaginal_fields is None:
        imaginal_fields = []

    if label_dataset != "" and label_field != "":
        label = LabelAssociation(
            dataset_id=label_dataset,
            field=label_field,
        )
    else:
        raise Exception("Both label_dataset and label_field must be provided")

    payload = MLModellingRequest(
        id=model_name,
        dataset_id=dataset_id,
        model_configs=model_configs,
        numerical_fields=numerical_fields,
        categorical_fields=categorical_fields,
        textual_fields=textual_fields,
        imaginal_fields=imaginal_fields,
        time_field=time_field,
        label=label,
        learner_config=LearnerConfig(
            predict_workers=predict_workers,
            update_batch_size=update_batch_size,
            synchronization_method=synchronization_method,
        ),
        predict_only=predict_only,
        initial_model_id=initial_model_id,
    )

    api.get(endpoint="model_validation", json=payload.model_dump())
    api.post(endpoint="ml_modelling", json=payload.model_dump())


def _resolve_duplicate_columns(
    input_df: pd.DataFrame, label_df: pd.DataFrame, key_field: str
):
    # Drop any common columns between the two from inputs
    # In the absence of this pandas will rename the conflicting columns to <col>_x <col>_y instead
    # Note that we drop from inputs instead of labels since the label dataframe is only supposed to have
    # the label and key fields, so a conflict would indicate that the label field made its way into the input.
    for col in label_df.columns:
        if col == key_field:
            continue
        if col in input_df.columns:
            logger.warn(
                f"Duplicate column '{col}' in input and label df. Dropping column from inputs"
            )
            input_df = input_df.drop(columns=[col])
    return input_df, label_df


def _prepare_merged_df(input: LocalInputs, labels: LocalLabels):
    """
    It resolves duplicate columns, and merges the input and label dataframes on the key field.
    """
    input_df, label_df = _resolve_duplicate_columns(
        input.dataframe, labels.dataframe, input.key_field
    )
    merged_df = pd.merge(input_df, label_df, on=input.key_field)
    return merged_df


def model_learn(
    model_name: str,
    merged_df: pd.DataFrame,
    key_field: str,
    label_field: str,
    numerical_fields: Optional[list[str]] = None,
    categorical_fields: Optional[list[str]] = None,
    textual_fields: Optional[list[str]] = None,
    imaginal_fields: Optional[list[str]] = None,
    time_field: Optional[str] = None,
    initial_model_key: str | None = None,
    model_configs: Optional[list[dict[str, str]]] = None,
    epochs: int = 1,
):
    if initial_model_key == "" and model_configs is None:
        raise Exception("initial_model_key and model_configs both can't be empty.")

    # Normalize
    if categorical_fields is None:
        categorical_fields = []
    if numerical_fields is None:
        numerical_fields = []
    if textual_fields is None:
        textual_fields = []
    if imaginal_fields is None:
        imaginal_fields = []
    if time_field is None:
        time_field = ""
    if model_configs is None:
        model_configs = []
    if initial_model_key == "":
        initial_model_key = None

    input_spec = InputSpec(
        key_field=key_field,
        time_field=time_field,
        numerical_fields=numerical_fields,
        categorical_fields=categorical_fields,
        textual_fields=textual_fields,
        imaginal_fields=imaginal_fields,
        label_field=label_field,
    )

    if initial_model_key is None:
        model_params = ModelParams(
            model_configs=model_configs,
            numerical_fields=numerical_fields,
            categorical_fields=categorical_fields,
            textual_fields=textual_fields,
            imaginal_fields=imaginal_fields,
            time_field=time_field,
            label=LabelAssociation(field=label_field, dataset_id=key_field),
        )
    else:
        model_params = None

    version_name = _save_model_configs_with_random_version(
        model_name, initial_model_key, model_params
    )

    # Send our training data to the server
    client = pyarrow.flight.connect(f"{CONFIG.ARROW_SERVER_ADDRESS}")

    input_table = TbPyArrow.df_to_table(merged_df, input_spec)

    upload_descriptor = pyarrow.flight.FlightDescriptor.for_command(
        f"learn:{model_name}:{version_name}"
    )
    options = pyarrow.flight.FlightCallOptions(headers=api.arrow_headers)

    TbPyArrow._put_and_retry(
        client, upload_descriptor, options, input_table, epochs=epochs
    )
    return version_name


def _save_model_configs_with_random_version(
    model_name: str,
    initial_model_key: str | None,
    model_params: ModelParams | None,
):
    version_name = "".join(random.choices(string.ascii_lowercase, k=10))
    payload = ModelConfigStorageRequest(
        id=model_name,
        version=version_name,
        initial_model_key=initial_model_key,
        params=model_params,
    )
    res = api.post(endpoint="train_config", json=payload.model_dump())
    if res.status_code != 201:
        raise Exception(f"Failed to save train config: {res.json()['detail']}")
    return version_name


def model_predict(
    model_name: str,
    initial_model_key: str,
    input_df: pd.DataFrame,
    key_field: str,
    numerical_fields: Optional[list[str]] = None,
    categorical_fields: Optional[list[str]] = None,
    textual_fields: Optional[list[str]] = None,
    imaginal_fields: Optional[list[str]] = None,
    time_field: Optional[str] = None,
):
    if model_name == "" or initial_model_key == "":
        raise ValueError("model_name and initial_model_key cannot be empty")

    # Normalize
    if categorical_fields is None:
        categorical_fields = []
    if numerical_fields is None:
        numerical_fields = []
    if time_field is None:
        time_field = ""
    if textual_fields is None:
        textual_fields = []
    if imaginal_fields is None:
        imaginal_fields = []

    # Send our inputs to the server, get back the predictions
    client = pyarrow.flight.connect(f"{CONFIG.ARROW_SERVER_ADDRESS}")

    input_spec = InputSpec(
        key_field=key_field,
        time_field=time_field,
        numerical_fields=numerical_fields,
        categorical_fields=categorical_fields,
        textual_fields=textual_fields,
        imaginal_fields=imaginal_fields,
        label_field="",  # will be ignored by df_to_table below
    )
    input_table = TbPyArrow.df_to_table(input_df, input_spec)

    request_id = "".join(random.choices(string.ascii_lowercase, k=10))
    upload_descriptor = pyarrow.flight.FlightDescriptor.for_command(
        f"predict:{request_id}:{model_name}:{initial_model_key}"
    )
    options = pyarrow.flight.FlightCallOptions(headers=api.arrow_headers)
    read_table = TbPyArrow._exchange_and_retry(
        client, upload_descriptor, options, input_table
    )
    return TbPyArrow.arrow_table_to_pandas(read_table)


def get_score_for_model(
    tmp_model: Model,
    input_table: pa.Table,
    input_spec: InputSpec,
    labels: LocalLabels,
    perf_metric: metrics._scorer._Scorer,
    prediction_column: str,
):
    if not tmp_model.model_id:
        tmp_model.model_id = "".join(random.choices(string.ascii_lowercase, k=10))
    initial_model_key = tmp_model.version
    model_configs = tmp_model.get_model_config()

    if initial_model_key == "" and model_configs is None:
        raise Exception("initial_model_key and model_configs both can't be empty.")

    if model_configs is None:
        model_configs = []
    if initial_model_key == "":
        initial_model_key = None

    label = LabelAssociation(field=labels.label_field, dataset_id=input_spec.key_field)
    if initial_model_key is None:
        model_params = ModelParams(
            model_configs=model_configs,
            numerical_fields=input_spec.numerical_fields,
            categorical_fields=input_spec.categorical_fields,
            textual_fields=input_spec.textual_fields,
            imaginal_fields=input_spec.imaginal_fields,
            time_field=input_spec.time_field,
            label=label,
        )
    else:
        model_params = None
    tmp_model.version = _save_model_configs_with_random_version(
        tmp_model.model_id, initial_model_key, model_params
    )
    # Send our training data to the server
    client = pyarrow.flight.connect(f"{CONFIG.ARROW_SERVER_ADDRESS}")

    upload_descriptor = pyarrow.flight.FlightDescriptor.for_command(
        f"learn:{tmp_model.model_id}:{tmp_model.version}"
    )
    options = pyarrow.flight.FlightCallOptions(headers=api.arrow_headers)
    TbPyArrow._put_and_retry(client, upload_descriptor, options, input_table)
    request_id = "".join(random.choices(string.ascii_lowercase, k=10))
    upload_descriptor = pyarrow.flight.FlightDescriptor.for_command(
        f"predict:{request_id}:{tmp_model.model_id}:{tmp_model.version}"
    )
    read_table = TbPyArrow._exchange_and_retry(
        client, upload_descriptor, options, input_table
    )
    temp_outputs = TbPyArrow.arrow_table_to_pandas(read_table)
    score = perf_metric._score_func(
        labels.dataframe[labels.label_field], temp_outputs[prediction_column]
    )
    return tmp_model, score


def validate_model_configs(model_configs: list[dict], input_spec: InputSpec):
    payload = ModelParams(
        model_configs=model_configs,
        label=LabelAssociation(
            field=input_spec.label_field,
            dataset_id=input_spec.key_field,
        ),
        numerical_fields=input_spec.numerical_fields,
        categorical_fields=input_spec.categorical_fields,
        textual_fields=input_spec.textual_fields,
        imaginal_fields=input_spec.imaginal_fields,
        time_field=input_spec.time_field,
    )

    resp = api.get(endpoint="model_validation", json=payload.model_dump())
    return resp.json()["message"]


class DeployedModel(BaseModel):
    model_name: str
    model_instance: Model
    algorithm: str
    model_configs: list[dict]

    class Config:
        arbitrary_types_allowed = True
        protected_namespaces = ()

    def __init__(self, **data):
        super().__init__(**data)
        api.get(f"model/{self.model_name}/info")

    def pause(self) -> None:
        """Pauses a running model."""
        api.patch(
            endpoint=f"model/{self.model_name}",
            json=ModelPatchRequest(action="pause").model_dump(mode="json"),
        )

    def resume(self) -> None:
        """Resumes a paused model or does nothing if model is already running."""
        api.patch(
            endpoint=f"model/{self.model_name}",
            json=ModelPatchRequest(action="resume").model_dump(mode="json"),
        )

    def delete(self, delete_output_topic: bool = True) -> None:
        """Delete the model.

        Args:
            delete_output_topic (bool, optional): Delete output dataset. Defaults to True.
        """
        api.delete(
            endpoint=f"model/{self.model_name}",
            json=ModelDeleteRequest(delete_output_topic=delete_output_topic).model_dump(
                mode="json"
            ),
        )

    def add_metric(self, metric_name) -> None:
        payload = MetricRegistrationRequest(
            metric=metric_name,
        )
        api.post(
            endpoint=f"model/{self.model_name}/metric",
            json=payload.model_dump(),
        )

    def add_drift(self) -> None:
        api.put(endpoint=f"model/{self.model_name}/target_drift")

    def get_drifts(self, limit: int = -1) -> list:
        return get_proto_msgs(
            StreamType.TARGET_DRIFT,
            self.model_name,
            output_pb2.Output,
            # limit
        )

    def get_outputs(self, limit: int = -1) -> list:
        if _istest():
            return retry_operation(
                lambda: validate_non_empty(
                    get_proto_msgs(
                        StreamType.OUTPUT,
                        self.model_name,
                        output_pb2.Output,
                        # limit
                    )
                ),
            )
        return get_proto_msgs(
            StreamType.OUTPUT,
            self.model_name,
            output_pb2.Output,
            # limit
        )

    def get_evaluation(
        self,
        metric_name: str,
        filter_expression: str = "",
        window_size: int = 1000,
        limit: int = 100000,
        output_type: Evaluations.ModelOutputType = Evaluations.ModelOutputType.SCORE,
    ) -> list:
        """Fetch model evaluation data for the given metric.

        This function sends a POST request to the server to get model evaluation data
        using the provided parameters.

        Args:
            metric_name (str): Evaluation metric to use.
            filter_expression (str): Filter expression for metric calculation, should be a valid SQL expression.
                Fields can be `processing_time` or any of the model `input_data` or `output_data` columns used as
                    `input_data.input_column1`,
                    `output_data.score`,
                    `output_data.predicted_class`,
                    `output_data.class_probabilities[1]`,
                    `output_data.feature_score[2]` etc...
                eg: `input_data.input1 > 100 AND (output_data.score > 0.5 OR output_data.feature_score[1] > 0.3)`,
                    `processing_time between '2024-12-31 15:42:38.425000' AND '2024-12-31 15:42:44.603000'`
            window_size (int): Window size to use for metric calculation.
            limit (int): Limit value for evaluation data response.
            output_type (`Evaluations.ModelOutputType`): Output type to use for response.

        Raises:
            Exception: Raises an exception if the POST request fails, providing details
            from the response JSON.
        """
        payload = Evaluations(
            model_names=[self.model_name],
            metric=metric_name,
            filter_expression=filter_expression,
            window_size=window_size,
            limit=limit,
            output_type=output_type,
        )

        if _istest():
            response = retry_operation(
                lambda: validate_non_empty(
                    api.post(
                        endpoint="model/evaluations",
                        json=payload.model_dump(),
                    ).json()
                ),
            )
        else:
            response = api.post(
                endpoint="model/evaluations",
                json=payload.model_dump(),
            ).json()

        if len(response) == 0:
            return []

        first_element = response[0]

        index_value_pairs = list(
            zip(first_element["index"], first_element["values"], strict=True)
        )
        return [
            metrics_pb2.Metrics(index=index, metric=metric)
            for index, metric in index_value_pairs
        ]

    def get_endpoints(self):
        resp = api.get(f"model/{self.model_name}/info").json()
        info = ModelInfo(**resp)

        base_url = CONFIG.TURBOML_BACKEND_SERVER_ADDRESS
        return [
            urlparse.urljoin(base_url, endpoint) for endpoint in info.endpoint_paths
        ]

    def get_logs(self):
        return ProcessOutput(**api.get(f"model/{self.model_name}/logs").json())

    def get_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        model_info = api.get(f"model/{self.model_name}/info").json()
        model_info = ModelInfo(**model_info)
        df = PandasHelpers.normalize_df(df)
        input_spec = model_info.metadata.get_input_spec()
        df_with_engineered_features = retrieve_features(
            model_info.metadata.input_db_source, df
        )
        table = TbPyArrow.df_to_table(df_with_engineered_features, input_spec)

        client = pyarrow.flight.connect(f"{CONFIG.ARROW_SERVER_ADDRESS}")

        request_id = "".join(random.choices(string.ascii_lowercase, k=10))
        model_port = model_info.metadata.process_config["arrowPort"]
        command_str = f"relay:{self.model_name}:{request_id}:{model_port}"
        upload_descriptor = pyarrow.flight.FlightDescriptor.for_command(command_str)
        options = pyarrow.flight.FlightCallOptions(headers=api.arrow_headers)
        read_table = TbPyArrow._exchange_and_retry(
            client, upload_descriptor, options, table
        )

        return TbPyArrow.arrow_table_to_pandas(read_table)

    def __getattr__(self, name):
        return getattr(self.model_instance, name)


class Model(ABC, BaseModel):
    model_id: str = Field(default=None, exclude=True)
    version: str = Field(default="", exclude=True)

    class Config:
        extra = "forbid"
        protected_namespaces = ()
        validate_assignment = True

    def __init__(self, **data):
        try:
            super().__init__(**data)
        except ValidationError as e:
            for error in e.errors():
                if error["type"] == "extra_forbidden":
                    extra_field = error["loc"][0]
                    raise Exception(
                        f"{extra_field} is not a field in {self.__class__.__name__}"
                    ) from e
            raise e

    def get_model_config(self):
        params = self.model_dump(by_alias=True)
        params["algorithm"] = self.__class__.__name__
        return [params]

    def learn(self, input: LocalInputs, labels: LocalLabels, epochs: int = 1):
        """
        Trains the model on provided input data and labels for the specified number of epochs.

        Parameters:
            input (Inputs): Contains input data.
            labels (Labels): Contains target labels.
            epochs (int, optional): No. of times to iterate over the dataset during training. Defaults to 1.
                - Note: Currently, data is processed in sequential order for each epoch.
                Users who need shuffling or sampling should modify the input data before calling learn method.
                These features may be added in the future.

        Returns:
            Model: A new model instance trained on the provided data.
        """
        if not self.model_id:
            self.model_id = "".join(random.choices(string.ascii_lowercase, k=10))

        merged_df = _prepare_merged_df(input, labels)

        version_name = model_learn(
            model_name=self.model_id,
            merged_df=merged_df,
            key_field=input.key_field,
            label_field=labels.label_field,
            numerical_fields=input.numerical_fields,
            categorical_fields=input.categorical_fields,
            textual_fields=input.textual_fields,
            imaginal_fields=input.imaginal_fields,
            time_field=input.time_field,
            initial_model_key=self.version,
            model_configs=self.get_model_config(),
            epochs=epochs,
        )

        trained_model = self.model_copy()
        trained_model.version = version_name

        return trained_model

    def predict(self, input: LocalInputs):
        if self.model_id is None:
            raise Exception("The model is untrained.")
        return model_predict(
            model_name=self.model_id,
            initial_model_key=self.version,
            input_df=input.dataframe,
            key_field=input.key_field,
            numerical_fields=input.numerical_fields,
            categorical_fields=input.categorical_fields,
            textual_fields=input.textual_fields,
            imaginal_fields=input.imaginal_fields,
            time_field=input.time_field,
        )

    def deploy(
        self, name: str, input: OnlineInputs, labels: OnlineLabels, predict_only=False
    ) -> DeployedModel:
        if self.model_id:
            initial_model_id = f"{api.namespace}.{self.model_id}:{self.version}"
        else:
            initial_model_id = ""

        if not isinstance(input, OnlineInputs) or not isinstance(labels, OnlineLabels):
            explaination = ""
            if isinstance(input, LocalInputs) or isinstance(labels, LocalLabels):
                explaination = (
                    " It looks like you are trying to deploy a model based on a local dataset."
                    " Please use OnlineDataset.from_local() to register your dataset with the"
                    " platform before deploying the model."
                )
            raise ValueError(
                "Inputs/labels must be an OnlineInputs/OnlineLabels object obtained from Online datasets."
                f"{explaination}"
            )

        model_configs = self.get_model_config()

        ml_modelling(
            model_name=name,
            model_configs=model_configs,
            label_dataset=labels.dataset_id if labels else "",
            label_field=labels.label_field if labels else "",
            # QUESTION: here input.dataset_id can be None. Are we
            # allowing deployment without input dataset_ids or should
            # we complain?
            dataset_id=input.dataset_id,
            numerical_fields=input.numerical_fields,
            categorical_fields=input.categorical_fields,
            textual_fields=input.textual_fields,
            imaginal_fields=input.imaginal_fields,
            time_field=input.time_field,
            predict_only=predict_only,
            initial_model_id=initial_model_id,
        )

        return DeployedModel(
            model_name=name,
            model_instance=self,
            algorithm=self.__class__.__name__,
            model_configs=model_configs,
        )

    def set_params(self, model_configs: list[dict]) -> None:
        model_config = model_configs[0]
        del model_config["algorithm"]
        for key, value in model_config.items():
            setattr(self, _camel_to_underscore(key), value)

    @staticmethod
    def _construct_model(
        configs: list, index: int = 0, is_flat: bool = False
    ) -> tuple[Model | None, int]:
        """
        Return (model_instance, next_config_index)
        """
        if index >= len(configs):
            return None, index
        config = configs[index]
        algorithm = config["algorithm"]
        model_class = globals()[algorithm]
        model_instance = model_class.construct()
        specific_config_dict = {k: v for k, v in config.items() if k != "algorithm"}
        convert_func = _identity
        num_children_key = "num_children"
        if not is_flat:
            convert_func = _camel_to_underscore
            num_children_key = "numChildren"
            specific_config_dict = list(specific_config_dict.values())[0]

        for key, value in specific_config_dict.items():
            setattr(model_instance, convert_func(key), value)

        num_children = specific_config_dict.get(num_children_key, 0)
        if num_children > 0:
            next_index = index + 1

            if "base_model" in model_class.__fields__:
                # For models with a single base_model
                base_model, next_index = Model._construct_model(
                    configs, next_index, is_flat
                )
                if base_model:
                    model_instance.base_model = base_model
                    model_instance.num_children = 1
            elif "base_models" in model_class.__fields__:
                # For models with multiple base_models
                base_models = []
                for _ in range(num_children):
                    child_model, next_index = Model._construct_model(
                        configs, next_index, is_flat
                    )
                    if child_model:
                        base_models.append(child_model)
                model_instance.base_models = base_models
                model_instance.num_children = len(base_models)
        else:
            next_index = index + 1

        return model_instance, next_index

    @staticmethod
    def _flatten_model_config(model):
        """
        Recreate flattened model configs
        """
        config = model.model_dump(by_alias=True)
        config["algorithm"] = model.__class__.__name__
        flattened = [config]

        if hasattr(model, "base_models"):
            for base_model in model.base_models:
                flattened.extend(Model._flatten_model_config(base_model))
        elif hasattr(model, "base_model") and model.base_model:
            flattened.extend(Model._flatten_model_config(model.base_model))

        return flattened

    @staticmethod
    def retrieve_model(model_name: str) -> DeployedModel:
        try:
            resp = api.get(f"model/{model_name}/info")
        except Exception as e:
            logger.error(f"Error fetching model: {e!r}")
            raise

        model_meta = ModelInfo(**resp.json()).metadata
        process_config = model_meta.process_config
        model_configs = process_config.get("modelConfigs", [])
        if not model_configs:
            raise ValueError("No model configurations found in the API response")

        root_model, _ = Model._construct_model(model_configs)

        flattened_configs = Model._flatten_model_config(root_model)
        deployed_model = DeployedModel(
            model_name=model_name,
            model_instance=root_model,
            algorithm=root_model.__class__.__name__,
            model_configs=flattened_configs,
        )

        return deployed_model

    def to_local_model(self, input_spec: InputSpec) -> LocalModel:
        """
        Converts the current Model instance into a LocalModel instance.
        """
        # TODO: Shouldn't we be retrieving the latest model snapshot from the server?
        params = self.model_dump(by_alias=True)
        config_key = default_configs.algo_config_mapping.get(self.__class__.__name__)

        if config_key:
            params = {
                _to_camel(config_key): params,
                "algorithm": self.__class__.__name__,
            }

        return LocalModel(model_configs=[params], input_spec=input_spec)


class LocalModel(BaseModel):
    """
    LocalModel allows for local training and prediction using Python bindings
    to the underlying C++ code.
    """

    model_configs: List[dict]
    input_spec: InputSpec
    cpp_model: CppModel = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("cpp_model")
    def serialize_cpp_model(self, cpp_model: CppModel) -> str:
        if cpp_model is None:
            return None
        model_bytes = cpp_model.serialize()
        model_base64 = base64.b64encode(model_bytes).decode("utf-8")
        return model_base64

    @field_validator("cpp_model", mode="before")
    @classmethod
    def deserialize_cpp_model(cls, value):
        if value is None:
            return None
        if isinstance(value, CppModel):
            return value
        if isinstance(value, str):
            model_bytes = base64.b64decode(value.encode("utf-8"))
            cpp_model = CppModel.deserialize(model_bytes)
            return cpp_model
        raise ValueError("Invalid type for cpp_model")

    def __init__(self, **data):
        super().__init__(**data)

        if self.cpp_model is None:
            # Serialize the model_configs to JSON
            config_json = json.dumps({"model_configs": self.model_configs})

            # Prepare the input configuration
            input_config = {
                "keyField": self.input_spec.key_field,
                "time_tick": self.input_spec.time_field or "",
                "numerical": self.input_spec.numerical_fields or [],
                "categorical": self.input_spec.categorical_fields or [],
                "textual": self.input_spec.textual_fields or [],
                "imaginal": self.input_spec.imaginal_fields or [],
            }
            input_config_json = json.dumps(input_config)

            # Create the cpp_model using create_model_from_config
            self.cpp_model = create_model_from_config(config_json, input_config_json)

    def learn_one(self, input: dict, label: int):
        """
        Learn from a single data point.
        """
        input_data = self._dict_to_input_data(input)
        input_data.label = label
        self.cpp_model.learn_one(input_data)

    def predict_one(self, input: dict) -> OutputData:
        """
        Predict for a single data point.
        """
        input_data = self._dict_to_input_data(input)
        output = self.cpp_model.predict_one(input_data)
        return output

    def _dict_to_input_data(self, input: dict) -> InputData:
        """
        Converts a dictionary of input data to an InputData object.
        """
        input_data = InputData()
        if self.input_spec.key_field in input:
            input_data.key = str(input[self.input_spec.key_field])
        else:
            input_data.key = ""
        if self.input_spec.time_field and self.input_spec.time_field in input:
            time_value = input[self.input_spec.time_field]
            if isinstance(time_value, pd.Timestamp):
                input_data.time_tick = int(time_value.timestamp())
            elif isinstance(time_value, datetime.datetime):
                input_data.time_tick = int(time_value.timestamp())
            else:
                input_data.time_tick = int(time_value)
        else:
            input_data.time_tick = 0
        input_data.numeric = [
            float(input[col])
            for col in self.input_spec.numerical_fields
            if col in input
        ]
        input_data.categ = [
            int(input[col])
            for col in self.input_spec.categorical_fields
            if col in input
        ]
        input_data.text = [
            str(input[col]) for col in self.input_spec.textual_fields if col in input
        ]
        input_data.images = [
            str(input[col]) for col in self.input_spec.imaginal_fields if col in input
        ]
        return input_data

    def learn(self, inputs: LocalInputs, labels: LocalLabels):
        """
        Trains the model on provided input data and labels.
        """
        merged_df = _prepare_merged_df(inputs, labels)
        for _, row in merged_df.iterrows():
            input_dict = row.to_dict()
            label = int(row[labels.label_field])
            self.learn_one(input_dict, label)

    def predict(self, inputs: LocalInputs) -> pd.DataFrame:
        """
        Makes predictions on provided input data.
        """
        outputs = []
        for _, row in inputs.dataframe.iterrows():
            input_dict = row.to_dict()
            output = self.predict_one(input_dict)
            outputs.append(output)
        # Convert outputs to DataFrame
        output_dicts = []
        for output in outputs:
            output_dict = {
                "score": output.score(),
                "predicted_class": output.predicted_class(),
                "feature_scores": output.feature_scores,
                "class_probabilities": output.class_probabilities,
                "text_output": output.text_output(),
                "embeddings": output.embeddings,
            }
            output_dicts.append(output_dict)
        output_df = pd.DataFrame(output_dicts)
        return output_df

    def serialize(self) -> bytes:
        return self.cpp_model.serialize()

    def __eq__(self, other):
        return self.cpp_model == other.cpp_model


def is_regressor(model: Model):
    REGRESSOR_CLASSES = [
        HoeffdingTreeRegressor,
        AMFRegressor,
        FFMRegressor,
        SGTRegressor,
        SNARIMAX,
    ]
    PREPROCESSOR_CLASSES = [
        MinMaxPreProcessor,
        NormalPreProcessor,
        RobustPreProcessor,
        LlamaCppPreProcessor,
        ClipEmbeddingPreprocessor,
        LLAMAEmbedding,
        LabelPreProcessor,
        OneHotPreProcessor,
        TargetPreProcessor,
        FrequencyPreProcessor,
        BinaryPreProcessor,
        ImageToNumericPreProcessor,
        RandomSampler,
    ]
    if any(isinstance(model, cls) for cls in REGRESSOR_CLASSES):
        return True

    if isinstance(model, NeuralNetwork):
        return True

    if isinstance(model, ONN):
        return model.n_classes == 1

    if any(isinstance(model, cls) for cls in PREPROCESSOR_CLASSES):
        # TODO: Add this assertion for type narrowing. Currently it fails because preprocessors don't inherit from PreProcessor.
        # Also PreProcessor appears mis-named.
        # assert isinstance(model, PreProcessor)
        return is_regressor(model.base_model)

    return False


def is_classifier(model: Model):
    CLASSIFIER_CLASSES = [
        HoeffdingTreeClassifier,
        AMFClassifier,
        FFMClassifier,
        SGTClassifier,
        LeveragingBaggingClassifier,
        HeteroLeveragingBaggingClassifier,
        AdaBoostClassifier,
        HeteroAdaBoostClassifier,
        AdaptiveXGBoost,
        AdaptiveLGBM,
    ]
    PREPROCESSOR_CLASSES = [
        MinMaxPreProcessor,
        NormalPreProcessor,
        RobustPreProcessor,
        ClipEmbeddingPreprocessor,
        LlamaCppPreProcessor,
        LLAMAEmbedding,
        PreProcessor,
        LabelPreProcessor,
        OneHotPreProcessor,
        TargetPreProcessor,
        FrequencyPreProcessor,
        BinaryPreProcessor,
        ImageToNumericPreProcessor,
        RandomSampler,
    ]
    if any(isinstance(model, cls) for cls in CLASSIFIER_CLASSES):
        return True

    if isinstance(model, NeuralNetwork):
        return True

    if isinstance(model, ONN):
        return model.n_classes > 1

    if any(isinstance(model, cls) for cls in PREPROCESSOR_CLASSES):
        return is_classifier(model.base_model)

    return False


class RCF(Model):
    time_decay: float = Field(default=0.000390625)
    number_of_trees: int = Field(default=50)
    output_after: int = Field(default=64)
    sample_size: int = Field(default=256)


class HST(Model):
    n_trees: int = Field(default=20)
    height: int = Field(default=12)
    window_size: int = Field(default=50)


class MStream(Model):
    num_rows: int = Field(default=2)
    num_buckets: int = Field(default=1024)
    factor: float = Field(default=0.8)


class ONNX(Model):
    model_save_name: str = Field(default="")
    model_config = ConfigDict(protected_namespaces=())


class HoeffdingTreeClassifier(Model):
    delta: float = Field(default=1e-7)
    tau: float = Field(default=0.05)
    grace_period: int = Field(default=200)
    n_classes: int
    leaf_pred_method: str = Field(default="mc")
    split_method: str = Field(default="gini")


class HoeffdingTreeRegressor(Model):
    delta: float = Field(default=1e-7)
    tau: float = Field(default=0.05)
    grace_period: int = Field(default=200)
    leaf_pred_method: str = Field(default="mean")


class AMFClassifier(Model):
    n_classes: int
    n_estimators: int = Field(default=10)
    step: float = Field(default=1)
    use_aggregation: bool = Field(default=True)
    dirichlet: float = Field(default=0.5)
    split_pure: bool = Field(default=False)


class AMFRegressor(Model):
    n_estimators: int = Field(default=10)
    step: float = Field(default=1)
    use_aggregation: bool = Field(default=True)
    dirichlet: float = Field(default=0.5)


class FFMClassifier(Model):
    n_factors: int = Field(default=10)
    l1_weight: float = Field(default=0)
    l2_weight: float = Field(default=0)
    l1_latent: float = Field(default=0)
    l2_latent: float = Field(default=0)
    intercept: float = Field(default=0)
    intercept_lr: float = Field(default=0.01)
    clip_gradient: float = Field(default=1e12)


class FFMRegressor(Model):
    n_factors: int = Field(default=10)
    l1_weight: float = Field(default=0)
    l2_weight: float = Field(default=0)
    l1_latent: float = Field(default=0)
    l2_latent: float = Field(default=0)
    intercept: float = Field(default=0)
    intercept_lr: float = Field(default=0.01)
    clip_gradient: float = Field(default=1e12)


class SGTClassifier(Model):
    delta: float = Field(default=1e-7)
    gamma: float = Field(default=0.1)
    grace_period: int = Field(default=200)
    lambda_: float = Field(default=0.1, alias="lambda")


class SGTRegressor(Model):
    delta: float = Field(default=1e-7)
    gamma: float = Field(default=0.1)
    grace_period: int = Field(default=200)
    lambda_: float = Field(default=0.1, alias="lambda")


class RandomSampler(Model):
    n_classes: int
    desired_dist: list = Field(default=[0.5, 0.5])
    sampling_method: str = Field(default="mixed")
    sampling_rate: float = Field(default=1.0)
    seed: int = Field(default=0)
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)

    def get_model_config(self):
        params = super().get_model_config()
        params += self.base_model.get_model_config()
        return params

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model = globals()[model_configs[0]["algorithm"]].construct()
        self.base_model.set_params(model_configs)
        self.num_children = 1


class NNLayer(BaseModel):
    output_size: int = 64
    activation: str = "relu"
    dropout: float = 0.3
    residual_connections: list = []
    use_bias: bool = True


class NeuralNetwork(Model):
    dropout: int = Field(default=0)
    layers: list[NNLayer] = Field(
        default_factory=lambda: [
            NNLayer(),
            NNLayer(),
            NNLayer(output_size=1, activation="sigmoid"),
        ]
    )
    loss_function: str = Field(default="mse")
    learning_rate: float = 1e-2
    optimizer: str = Field(default="sgd")
    batch_size: int = 64

    @validator("layers")
    @classmethod # DC_Change
    def validate_layers(cls, layers):
        if len(layers) == 0:
            raise Exception("layers must be non empty")

        # TODO other layer checks
        return layers


class Python(Model):
    module_name: str = ""
    class_name: str = ""
    venv_name: str = ""


class ONN(Model):
    max_num_hidden_layers: int = Field(default=10)
    qtd_neuron_hidden_layer: int = Field(default=32)
    n_classes: int
    b: float = Field(default=0.99)
    n: float = Field(default=0.01)
    s: float = Field(default=0.2)


class OVR(Model):
    n_classes: int
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)

    def get_model_config(self):
        params = super().get_model_config()
        params += self.base_model.get_model_config()
        return params

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model = globals()[model_configs[0]["algorithm"]].construct()
        self.base_model.set_params(model_configs)
        self.num_children = 1


class MultinomialNB(Model):
    n_classes: int
    alpha: float = Field(default=1.0)


class GaussianNB(Model):
    n_classes: int


class AdaptiveXGBoost(Model):
    n_classes: int
    learning_rate: float = Field(default=0.3)
    max_depth: int = Field(default=6)
    max_window_size: int = Field(default=1000)
    min_window_size: int = Field(default=0)
    max_buffer: int = Field(default=5)
    pre_train: int = Field(default=2)
    detect_drift: bool = Field(default=True)
    use_updater: bool = Field(default=True)
    trees_per_train: int = Field(default=1)
    percent_update_trees: float = Field(default=1.0)


class AdaptiveLGBM(Model):
    n_classes: int
    learning_rate: float = Field(default=0.3)
    max_depth: int = Field(default=6)
    max_window_size: int = Field(default=1000)
    min_window_size: int = Field(default=0)
    max_buffer: int = Field(default=5)
    pre_train: int = Field(default=2)
    detect_drift: bool = Field(default=True)
    use_updater: bool = Field(default=True)
    trees_per_train: int = Field(default=1)


class PreProcessor(Model):
    preprocessor_name: str
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)
    text_categories: list[int] = Field(default=[])
    image_sizes: list[int] = Field(default=[64, 64, 1])
    channel_first: bool = Field(default=False)
    gguf_model_id: GGUFModelId = Field(default=None)
    max_tokens_per_input: int = Field(default=512)

    def get_model_config(self):
        params = self.model_dump(by_alias=True)
        params["algorithm"] = "PreProcessor"
        params["preprocessor_name"] = self.preprocessor_name
        model_cfgs = [params] + self.base_model.get_model_config()
        return model_cfgs

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model.set_params(model_configs)
        self.num_children = 1


class MinMaxPreProcessor(Model):
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)

    def get_model_config(self):
        params = self.model_dump(by_alias=True)
        params["algorithm"] = "PreProcessor"
        params["preprocessor_name"] = "MinMax"
        model_cfgs = [params] + self.base_model.get_model_config()
        return model_cfgs

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model.set_params(model_configs)
        self.num_children = 1


class NormalPreProcessor(Model):
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)

    def get_model_config(self):
        params = self.model_dump(by_alias=True)
        params["algorithm"] = "PreProcessor"
        params["preprocessor_name"] = "Normal"
        model_cfgs = [params] + self.base_model.get_model_config()
        return model_cfgs

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model.set_params(model_configs)
        self.num_children = 1


class RobustPreProcessor(Model):
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)

    def get_model_config(self):
        params = self.model_dump(by_alias=True)
        params["algorithm"] = "PreProcessor"
        params["preprocessor_name"] = "Robust"
        model_cfgs = [params] + self.base_model.get_model_config()
        return model_cfgs

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model.set_params(model_configs)
        self.num_children = 1


class LabelPreProcessor(Model):
    text_categories: list[int] = Field(default=[])
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)

    def get_model_config(self):
        params = self.model_dump(by_alias=True)
        params["algorithm"] = "PreProcessor"
        params["preprocessor_name"] = "Label"
        model_cfgs = [params] + self.base_model.get_model_config()
        return model_cfgs

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model.set_params(model_configs)
        self.num_children = 1


class OneHotPreProcessor(Model):
    text_categories: list[int] = Field(default=[])
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)

    def get_model_config(self):
        params = self.model_dump(by_alias=True)
        params["algorithm"] = "PreProcessor"
        params["preprocessor_name"] = "OneHot"
        model_cfgs = [params] + self.base_model.get_model_config()
        return model_cfgs

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model.set_params(model_configs)
        self.num_children = 1


class TargetPreProcessor(Model):
    text_categories: list[int] = Field(default=[])
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)

    def get_model_config(self):
        params = self.model_dump(by_alias=True)
        params["algorithm"] = "PreProcessor"
        params["preprocessor_name"] = "Target"
        model_cfgs = [params] + self.base_model.get_model_config()
        return model_cfgs

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model.set_params(model_configs)
        self.num_children = 1


class FrequencyPreProcessor(Model):
    text_categories: list[int] = Field(default=[])
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)

    def get_model_config(self):
        params = self.model_dump(by_alias=True)
        params["algorithm"] = "PreProcessor"
        params["preprocessor_name"] = "Frequency"
        model_cfgs = [params] + self.base_model.get_model_config()
        return model_cfgs

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model.set_params(model_configs)
        self.num_children = 1


class BinaryPreProcessor(Model):
    text_categories: list[int] = Field(default=[])
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)

    def get_model_config(self):
        params = self.model_dump(by_alias=True)
        params["algorithm"] = "PreProcessor"
        params["preprocessor_name"] = "Binary"
        model_cfgs = [params] + self.base_model.get_model_config()
        return model_cfgs

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model.set_params(model_configs)
        self.num_children = 1


class ImageToNumericPreProcessor(Model):
    image_sizes: list[int] = Field(default=[64, 64, 1])
    channel_first: bool = Field(default=False)
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)

    def get_model_config(self):
        params = self.model_dump(by_alias=True)
        params["algorithm"] = "PreProcessor"
        params["preprocessor_name"] = "ImageToNumeric"
        model_cfgs = [params] + self.base_model.get_model_config()
        return model_cfgs

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model.set_params(model_configs)
        self.num_children = 1


class SNARIMAX(Model):
    horizon: int = Field(default=1)
    p: int = Field(default=1)
    d: int = Field(default=1)
    q: int = Field(default=1)
    m: int = Field(default=1)
    sp: int = Field(default=0)
    sd: int = Field(default=0)
    sq: int = Field(default=0)
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)

    @validator("base_model")
    def validate_base_model(cls, base_model):
        if not is_regressor(base_model):
            raise Exception("base_model must be a regressor model")
        return base_model

    def get_model_config(self):
        params = super().get_model_config()
        params += self.base_model.get_model_config()
        return params

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model = globals()[model_configs[0]["algorithm"]].construct()
        self.base_model.set_params(model_configs)
        self.num_children = 1


class HeteroLeveragingBaggingClassifier(Model):
    n_classes: int
    w: float = Field(default=6)
    bagging_method: str = Field(default="bag")
    seed: int = Field(default=0)
    base_models: list[Model] = Field(..., exclude=True)
    num_children: int = Field(default=0)

    @validator("base_models")
    
    def validate_base_models(cls, base_models):
        for base_model in base_models:
            if not is_classifier(base_model):
                raise Exception("all base_models must be classifier models")
        return base_models

    def get_model_config(self):
        params = super().get_model_config()
        for base_model in self.base_models:
            params += base_model.get_model_config()
        return params

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_models = []
        while len(model_configs):
            base_model = globals()[model_configs[0]["algorithm"]].construct()
            base_model.set_params(model_configs)
            self.base_models.append(base_model)
        self.update_num_children()

    def update_num_children(self):
        self.num_children = len(self.base_models)

    @model_validator(mode="before")
    @classmethod
    def set_num_children(cls, values):
        base_models = values.get("base_models", [])
        values["num_children"] = len(base_models)
        return values


class AdaBoostClassifier(Model):
    n_models: int = Field(default=10)
    n_classes: int
    seed: int = Field(default=0)
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=0)

    @validator("base_model")
    def validate_base_model(cls, base_model):
        if not is_classifier(base_model):
            raise Exception("base_model must be a classifier model")
        return base_model

    def get_model_config(self):
        params = super().get_model_config()
        params += self.base_model.get_model_config()
        return params

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model = globals()[model_configs[0]["algorithm"]].construct()
        self.base_model.set_params(model_configs)
        self.update_num_children()

    def update_num_children(self):
        self.num_children = 1 if self.base_model else 0

    @model_validator(mode="before")
    @classmethod
    def set_num_children(cls, values):
        values["num_children"] = 1 if values.get("base_model") else 0
        return values


class LeveragingBaggingClassifier(Model):
    n_models: int = Field(default=10)
    n_classes: int
    w: float = Field(default=6)
    bagging_method: str = Field(default="bag")
    seed: int = Field(default=0)
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=0)

    @validator("base_model")
    def validate_base_model(cls, base_model):
        if not is_classifier(base_model):
            raise Exception("base_model must be a classifier model")
        return base_model

    def get_model_config(self):
        params = super().get_model_config()
        params += self.base_model.get_model_config()
        return params

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model = globals()[model_configs[0]["algorithm"]].construct()
        self.base_model.set_params(model_configs)
        self.update_num_children()

    def update_num_children(self):
        self.num_children = 1 if self.base_model else 0

    @model_validator(mode="before")
    @classmethod
    def set_num_children(cls, values):
        values["num_children"] = 1 if values.get("base_model") else 0
        return values


class HeteroAdaBoostClassifier(Model):
    n_classes: int
    seed: int = Field(default=0)
    base_models: list[Model] = Field(..., exclude=True)
    num_children: int = Field(default=0)

    @validator("base_models")
    def validate_base_models(cls, base_models):
        for base_model in base_models:
            if not is_classifier(base_model):
                raise Exception("all base_models must be classifier models")
        return base_models

    def __init__(self, **data):
        super().__init__(**data)
        self.update_num_children()

    def update_num_children(self):
        self.num_children = len(self.base_models)

    def get_model_config(self):
        params = super().get_model_config()
        for base_model in self.base_models:
            params += base_model.get_model_config()
        return params

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_models = []
        while len(model_configs):
            base_model = globals()[model_configs[0]["algorithm"]].construct()
            base_model.set_params(model_configs)
            self.base_models.append(base_model)
        self.update_num_children()

    @model_validator(mode="before")
    @classmethod
    def set_num_children(cls, values):
        base_models = values.get("base_models", [])
        values["num_children"] = len(base_models)
        return values


class BanditModelSelection(Model):
    bandit: str = Field(default="EpsGreedy")
    metric_name: EvaluationMetrics = Field(default="WindowedMAE")
    base_models: list[Model] = Field(..., exclude=True)

    def get_model_config(self):
        params = super().get_model_config()
        for base_model in self.base_models:
            params += base_model.get_model_config()

        return params

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_models = []
        while len(model_configs):
            base_model = globals()[model_configs[0]["algorithm"]].construct()
            base_model.set_params(model_configs)
            self.base_models.append(base_model)


class ContextualBanditModelSelection(Model):
    contextualbandit: str = Field(default="LinTS")
    metric_name: EvaluationMetrics = Field(default="WindowedMAE")
    base_models: list[Model] = Field(..., exclude=True)

    def get_model_config(self):
        params = super().get_model_config()
        for base_model in self.base_models:
            params += base_model.get_model_config()

        return params

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_models = []
        while len(model_configs):
            base_model = globals()[model_configs[0]["algorithm"]].construct()
            base_model.set_params(model_configs)
            self.base_models.append(base_model)


class RandomProjectionEmbedding(Model):
    n_embeddings: int = Field(default=2)
    type_embedding: str = Field(default="Gaussian")


class ClipEmbeddingPreprocessor(Model):
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)
    gguf_model_id: GGUFModelId = Field(default=None)

    def get_model_config(self):
        params = self.model_dump(by_alias=True)
        params["algorithm"] = "PreProcessor"
        params["preprocessor_name"] = "ClipEmbeddingPreprocessor"
        model_cfgs = [params] + self.base_model.get_model_config()
        return model_cfgs

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model.set_params(model_configs)
        self.num_children = 1


class LlamaCppPreProcessor(Model):
    """
    LlamaCppPreProcessor is a preprocessor model that uses the LlamaCpp library to
    preprocess text fields into embeddings, passing them to the base model
    as numerical features.
    """

    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)
    gguf_model_id: GGUFModelId = Field(default=None)
    """
    A model id issued by `tb.llm.acquire_hf_model_as_gguf`.
    If this is not provided, our default BERT model will be used.
    """
    max_tokens_per_input: int = Field(default=512)
    """
    The maximum number of tokens to consider in the input text.
    Tokens beyond this limit will be truncated.
    """

    def get_model_config(self):
        params = self.model_dump(by_alias=True)
        params["algorithm"] = "PreProcessor"
        params["preprocessor_name"] = "LlamaCpp"
        model_cfgs = [params] + self.base_model.get_model_config()
        return model_cfgs

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model.set_params(model_configs)
        self.num_children = 1


class LlamaTextPreprocess(Model):
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=1)
    gguf_model_id: GGUFModelId = Field(default=None)

    def get_model_config(self):
        params = self.model_dump(by_alias=True)
        params["algorithm"] = "PreProcessor"
        params["preprocessor_name"] = "LlamaTextPreprocess"
        model_cfgs = [params] + self.base_model.get_model_config()
        return model_cfgs

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.base_model.set_params(model_configs)
        self.num_children = 1


class LLAMAEmbedding(Model):
    """
    LLAMAEmbedding is a model that uses the LlamaCpp library to preprocess text fields
    into embeddings, filling them into the `embeddings` field of the output.
    """

    gguf_model_id: GGUFModelId = Field(default=None)
    """
    A model id issued by `tb.llm.acquire_hf_model_as_gguf`.
    If this is not provided, our default BERT model will be used.
    """
    max_tokens_per_input: int = Field(default=512)
    """
    The maximum number of tokens to consider in the input text.
    Tokens beyond this limit will be truncated.
    """


class ClipEmbedding(Model):
    gguf_model_id: GGUFModelId = Field(default=None)


class LlamaText(Model):
    gguf_model_id: GGUFModelId = Field(default=None)


class EmbeddingModel(Model):
    embedding_model: Model = Field(..., exclude=True)
    base_model: Model = Field(..., exclude=True)
    num_children: int = Field(default=2)

    def get_model_config(self):
        params = super().get_model_config()
        params += self.embedding_model.get_model_config()
        params += self.base_model.get_model_config()
        return params

    def set_params(self, model_configs: list[dict]) -> None:
        super().set_params(model_configs)
        self.embedding_model = globals()[model_configs[0]["algorithm"]].construct()
        self.embedding_model.set_params(model_configs)
        self.base_model = globals()[model_configs[1]["algorithm"]].construct()
        self.base_model.set_params(model_configs[1:])
        self.num_children = 2


class RestAPIClient(Model):
    server_url: str = Field()
    max_retries: int = Field(default=3)
    connection_timeout: int = Field(default=10)
    max_request_time: int = Field(default=30)


class PythonEnsembleModel(Model):
    """
    PythonEnsembleModel manages an ensemble of Python-based models.
    """

    base_models: list[Model] = Field(..., exclude=True)
    module_name: Optional[str] = Field(default=None)
    class_name: Optional[str] = Field(default=None)
    venv_name: Optional[str] = Field(default=None)

    def get_model_config(self):
        ensemble_params = {
            "algorithm": "PythonEnsembleModel",
            "module_name": self.module_name,
            "class_name": self.class_name,
            "venv_name": self.venv_name or "",
        }
        configs = [ensemble_params]
        for base_model in self.base_models:
            configs.extend(base_model.get_model_config())
        return configs

    def set_params(self, model_configs: list[dict]) -> None:
        if not model_configs:
            raise ValueError("No configuration provided for PythonEnsembleModel.")

        # Extract ensemble-specific configuration
        ensemble_config = model_configs[0]
        if ensemble_config.get("algorithm") != "PythonEnsembleModel":
            raise ValueError("The first configuration must be for PythonEnsembleModel.")

        self.module_name = ensemble_config.get("module_name", "")
        self.class_name = ensemble_config.get("class_name", "")
        self.venv_name = ensemble_config.get("venv_name", "")

        # Initialize base models
        base_model_configs = model_configs[1:]  # Remaining configs are for base models
        if not base_model_configs:
            raise ValueError(
                "PythonEnsembleModel requires at least one base model configuration."
            )

        self.base_models = []
        for config in base_model_configs:
            algorithm = config.get("algorithm")
            if not algorithm:
                raise ValueError(
                    "Each base model configuration must include an 'algorithm' field."
                )
            model_class = globals().get(algorithm)
            if not model_class:
                raise ValueError(f"Unknown algorithm '{algorithm}' for base model.")
            base_model = model_class.construct()
            base_model.set_params([config])
            self.base_models.append(base_model)


class GRPCClient(Model):
    server_url: str = Field()
    max_retries: int = Field(default=3)
    connection_timeout: int = Field(default=10000)
    max_request_time: int = Field(default=30000)
