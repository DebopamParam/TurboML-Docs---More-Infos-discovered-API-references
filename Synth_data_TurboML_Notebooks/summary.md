# **TurboML: A Real-Time Machine Learning Platform - Detailed Summary**

TurboML is a platform designed for building, deploying, and managing real-time machine learning applications. It emphasizes streaming data and provides tools for the entire ML lifecycle, from data ingestion to model monitoring.

**1. Data Ingestion and Management:**

*   **Core Principle:** TurboML treats data as continuous streams, enabling real-time processing and updates.
*   **Ingestion Methods:**
    *   **Pull-based:**
        *   Uses pre-built connectors to continuously pull data from various sources.
        *   Supported sources include cloud storage (e.g., S3) and databases (e.g., Postgres). *While Kafka is used internally, the documentation doesn't explicitly present it as a direct pull-based source for end-users in the introductory sections.*
        *   Connectors are configured to handle data formats and connection details.
    *   **Push-based:**
        *   Allows direct data injection into the TurboML platform.
        *   Methods:
            *   **REST API:** Send data via HTTP requests using the `dataset/{dataset_id}/upload` endpoint.
            *   **Client SDKs:** More performant options for high-volume data. The Python SDK provides convenient methods for working with Pandas DataFrames.
            *   **gRPC API:** Upload data using Arrow Flight gRPC, providing the most performant option.
        *   Example (Pandas DataFrame):
            ```python
            transactions = tb.OnlineDataset.from_pd(
                id="qs_transactions",
                df=transactions_df,
                key_field="transactionID",
                load_if_exists=True,
            )
            ```
*   **Dataset Classes:**
    *   **`OnlineDataset`:**
        *   Represents a dataset managed by the TurboML platform.
        *   Supports continuous data ingestion (pull or push).
        *   Provides methods for feature engineering, model deployment, and monitoring.
        *   Can be created from Pandas DataFrames (`from_pd`), or loaded if it already exists (`load`).
        *   `add_pd()` method allows adding new data to an existing `OnlineDataset` (using Arrow Flight Protocol over gRPC). There are also `dataset/dataset_id/upload` REST API endpoint and direct gRPC API options for data upload.
        *   `sync_features()` method synchronizes materialized streaming features to the `OnlineDataset` object.  This is important after uploading new data or materializing features.
    *   **`LocalDataset`:**
        *   Represents an in-memory dataset, primarily for local experimentation and development.
        *   Can be created from Pandas DataFrames (`from_pd`).
        *   Useful for testing feature engineering logic before deploying to an `OnlineDataset`.
        *   Can be converted to an `OnlineDataset` using `to_online()`.
    * **`PandasDataset`:** *This class is present in the `intro` documentation, but its role and usage are less clearly defined compared to `OnlineDataset` and `LocalDataset`. It appears to be a less preferred way of interacting with data.*
* **Data Schema:**
    *   Datasets have a defined schema, specifying field names and data types.
    *   Schemas are automatically inferred from Pandas DataFrames.
    *   Schemas are managed by the platform and used for data validation and consistency.
    *   Supported data types include: INT32, INT64, FLOAT, DOUBLE, STRING, BOOL, BYTES.
* **Key Field:**
    *   Each dataset must have a primary key field (`key_field`) to uniquely identify records.
    *   Used for merging data, performing lookups, and ensuring data integrity.

**2. Feature Engineering:**

*   **Core Philosophy:** Define features once, test them locally, and then deploy them for continuous, real-time computation.
*   **Feature Definition Interfaces:**
    *   **SQL Features:**
        *   Define features using standard SQL expressions.
        *   Column names are enclosed in double quotes.
        *   Example:
            ```python
            transactions.feature_engineering.create_sql_features(
                sql_definition='"transactionAmount" + "localHour"',
                new_feature_name="my_sql_feat",
            )
            ```
    *   **Aggregate Features:**
        *   Define time-windowed aggregations (SUM, COUNT, AVG, MIN, MAX, etc.).
        *   Require a registered timestamp column.
        *   Specify:
            *   `column_to_operate`: The column to aggregate.
            *   `column_to_group`: The column(s) to group by.
            *   `operation`: The aggregation function (SUM, COUNT, etc.).
            *   `new_feature_name`: The name of the new feature column.
            *   `timestamp_column`: The column containing timestamps.
            *   `window_duration`: The length of the time window.
            *   `window_unit`: The unit of the window duration (seconds, minutes, hours, etc.).
        *   Example:
            ```python
            transactions.feature_engineering.register_timestamp(column_name="timestamp", format_type="epoch_seconds")

            transactions.feature_engineering.create_aggregate_features(
                column_to_operate="transactionAmount",
                column_to_group="accountID",
                operation="SUM",
                new_feature_name="my_sum_feat",
                timestamp_column="timestamp",
                window_duration=24,
                window_unit="hours",
            )
            ```
    *   **User-Defined Functions (UDFs):**
        *   Define features using custom Python code.
        *   **Simple UDFs:** Functions that take column values as input and return a single value.
            ```python
            myfunction_contents = """
            import numpy as np

            def myfunction(x):
                return np.sin(x)
            """
            transactions.feature_engineering.create_udf_features(
                new_feature_name="sine_of_amount",
                argument_names=["transactionAmount"],
                function_name="myfunction",
                function_file_contents=myfunction_contents,
                libraries=["numpy"],
            )

            ```
        *   **Rich UDFs:** Class-based UDFs that can maintain state and perform more complex operations (e.g., database lookups). Require an `__init__` method and a `func` method. The `dev_initializer_arguments` and `prod_initializer_arguments` are used for development and production environments, respectively.
        *   **User-Defined Aggregate Functions (UDAFs):** Custom aggregations defined in Python. Require implementing `create_state`, `accumulate`, `retract` (optional), `merge_states`, and `finish` methods.
            ```python
            function_file_contents = """
            def create_state():
                return 0, 0

            def accumulate(state, value, weight):
                if value is None or weight is None:
                    return state
                (s, w) = state
                s += value * weight
                w += weight
                return s, w

            def retract(state, value, weight):
                if value is None or weight is None:
                    return state
                (s, w) = state
                s -= value * weight
                w -= weight
                return s, w

            def merge_states(state_a, state_b):
                (s_a, w_a) = state_a
                (s_b, w_b) = state_b
                return s_a + s_b, w_a + w_b

            def finish(state):
                (sum, weight) = state
                if weight == 0:
                    return None
                else:
                    return sum / weight
            """
            transactions.feature_engineering.create_udaf_features(
                new_feature_name="weighted_avg",
                column_to_operate=["transactionAmount", "transactionTime"],
                function_name="weighted_avg",
                return_type="DOUBLE",
                function_file_contents=function_file_contents,
                column_to_group=["accountID"],
                timestamp_column="timestamp",
                window_duration=1,
                window_unit="hours",
            )
            ```
    *   **Ibis Features:**
        *   Define complex streaming features using the Ibis DataFrame API.
        *   Supports Apache Flink and RisingWave backends for execution.
        *   Allows for more sophisticated feature engineering than simple SQL or aggregations.
        *   Example:
            ```python
            fe = tb.IbisFeatureEngineering()
            transactions = fe.get_ibis_table("transactions_stream")
            # ... define features using Ibis expressions ...
            fe.materialize_features(
                transactions_with_frequency_score,
                "transactions_with_frequency_score",
                "transactionID",
                BackEnd.Flink, # Or BackEnd.Risingwave
                "transactions_stream",
            )
            ```

*   **Feature Materialization:**
    *   `materialize_features()`: Submits feature definitions to the platform for continuous computation. Features are computed in real-time as new data arrives.
    *   `materialize_ibis_features()`: Submits Ibis feature definitions.
*   **Feature Retrieval:**
    *   `get_features()`: Retrieves a *snapshot* of the raw data stream (for experimentation). *Note:* The returned data is not guaranteed to be in the same order or size on each call.
    *   `get_local_features()`: Returns a DataFrame with the locally computed features (for debugging and experimentation).
    *   `get_materialized_features()`: Retrieves the *continuously computed* features from the platform.
    *   `retrieve_features()`: Computes feature values on ad-hoc data (not part of the stream).
*   **Timestamp Handling:**
    *   `register_timestamp()`: Registers a column as the timestamp for the dataset. Required for time-windowed aggregations.
    *   `get_timestamp_formats()`: Returns a list of supported timestamp format strings.
    *   `convert_timestamp()`: *This function is an internal utility, not a directly exposed user-facing API. It's used within the feature engineering logic.*
* **Classes/Functions:**
    *   `FeatureEngineering`: Class for defining SQL and aggregation features on `OnlineDataset`.
    *   `LocalFeatureEngineering`: Class for defining features on `LocalDataset`.
    *   `IbisFeatureEngineering`: Class for defining features using the Ibis interface.
    *   `tb.register_source()`: Registers a data source configuration.
    *   `DataSource`: Defines where and how raw data is accessed.
        *    `FileSource`: Specifies a file-based data source (e.g., CSV, Parquet).
        *   `PostgresSource`: Specifies a PostgreSQL data source.
        *   `KafkaSource`: *Mentioned in the context, but not as a direct source for `DataSource` in the `intro`.*
        *   `FeatureGroupSource`: Specifies a feature group as a data source.
    *   `TimestampFormatConfig`: Configures timestamp format.
    *   `Watermark`: Defines watermark settings for streaming data.
    *   `TurboMLScalarFunction`: Base class for defining rich UDFs.

**3. ML Modeling:**

*   **Model Types:**
    *   **Supervised:** Models that learn from labeled data (e.g., classification, regression).
    *   **Unsupervised:** Models that learn from unlabeled data (e.g., anomaly detection).
*   **Input Specification:**
    *   Models require specifying the input features using:
        *   `numerical_fields`: List of numeric column names.
        *   `categorical_fields`: List of categorical column names.
        *   `textual_fields`: List of text column names.
        *   `imaginal_fields`: List of image column names (binary data).
        *   `time_field`: Name of the timestamp column (optional).
    *   Example:
        ```python
        numerical_fields = ["transactionAmount", "localHour"]
        categorical_fields = ["digitalItemCount", "physicalItemCount", "isProxyIP"]
        features = transactions.get_model_inputs(
            numerical_fields=numerical_fields, categorical_fields=categorical_fields
        )
        label = labels.get_model_labels(label_field="is_fraud")
        ```
* **Supported Algorithms (Examples):**
    *   **Classification:**
        *   `HoeffdingTreeClassifier`: Incremental decision tree for classification.
        *   `AMFClassifier`: Aggregated Mondrian Forest classifier.
        *   `FFMClassifier`: Field-aware Factorization Machine classifier.
        *   `SGTClassifier`: Stochastic Gradient Tree classifier.
        *   `MultinomialNB`: Multinomial Naive Bayes.
        *   `GaussianNB`: Gaussian Naive Bayes.
    *   **Regression:**
        *   `HoeffdingTreeRegressor`: Incremental decision tree for regression.
        *   `AMFRegressor`: Aggregated Mondrian Forest regressor.
        *   `FFMRegressor`: Field-aware Factorization Machine regressor.
        *   `SGTRegressor`: Stochastic Gradient Tree regressor.
        *   `SNARIMAX`: Time series forecasting model.
    *   **Anomaly Detection:**
        *   `RCF`: Random Cut Forest.
        *   `HST`: Half-Space Trees.
        *   `MStream`: Multi-aspect stream anomaly detection.
    *   **Ensemble Methods:**
        *   `LeveragingBaggingClassifier`: Bagging with ADWIN for concept drift.
        *   `HeteroLeveragingBaggingClassifier`: Bagging with different base models.
        *   `AdaBoostClassifier`: AdaBoost.
        *   `HeteroAdaBoostClassifier`: AdaBoost with different base models.
        *   `BanditModelSelection`: Model selection using bandit algorithms.
        *   `ContextualBanditModelSelection`: Model selection using contextual bandits.
        *   `RandomSampler`: Random sampling for imbalanced datasets.
    *   **General Purpose:**
        *   `NeuralNetwork`: Configurable neural network.
        *   `ONN`: Online Neural Network.
        *   `AdaptiveXGBoost`: XGBoost with concept drift handling.
        *   `AdaptiveLGBM`: LightGBM with concept drift handling.
        *   `ONNX`: Deploy models from other frameworks (PyTorch, TensorFlow, Scikit-learn) using ONNX format (static models).
        *   `Python`: Define custom models using Python classes.
        *   `PythonEnsembleModel`: Define custom ensemble models using Python classes.
        *   `RestAPIClient`: Use custom models via REST API.
        *   `GRPCClient`: Use custom models via gRPC API.
*   **Bring Your Own Models (BYOM):**
    *   **ONNX:** Deploy models trained in other frameworks.
    *   **Python:** Define custom models with `learn_one` and `predict_one` methods. Requires defining a class with `init_imports`, `learn_one`, and `predict_one` methods. A virtual environment (`venv`) can be set up to manage dependencies.
    *   **gRPC:** Integrate models via gRPC.
    *   **REST API:** Integrate models via REST API.
*   **Model Composition:**
    *   Combine models using ensemble methods (e.g., `LeveragingBaggingClassifier`).
    *   Chain preprocessors with models (e.g., `MinMaxPreProcessor` + `HoeffdingTreeClassifier`).
*   **Preprocessors:**
    *   Transform input data before it's passed to the model.
    *   Examples:
        *   `MinMaxPreProcessor`: Scales numerical features to [0, 1].
        *   `NormalPreProcessor`: Standardizes numerical features (zero mean, unit variance).
        *   `RobustPreProcessor`: Scales features using robust statistics (median, IQR).
        *   `LabelPreProcessor`: Converts strings to ordinal integers (textual fields).
        *   `OneHotPreProcessor`: Creates one-hot encoding for categorical features.
        *   `BinaryPreProcessor`: Creates binary encoding for categorical features.
        *   `FrequencyPreProcessor`: Encodes strings based on their frequency.
        *   `TargetPreProcessor`: Encodes strings based on the average target value.
        *   `LlamaCppPreProcessor`: Generates text embeddings using Llama.cpp models (GGUF format).
        *   `ClipEmbeddingPreprocessor`: Generates image embeddings using CLIP models (GGUF format).
        *   `ImageToNumericPreProcessor`: Converts binary image data to numerical data.
        *   `RandomProjectionEmbedding`: Dimensionality reduction using random projection.
        *   `LLAMAEmbedding`: Text embeddings using GGUF models.
        *   `ClipEmbedding`: Image embeddings using GGUF models.
        *   `EmbeddingModel`: Combines an embedding model with a base model.
        *   `OVR`: One-vs-the-rest multiclass strategy.
    *   Preprocessors are typically combined with a `base_model`.
* **Model Training:**
    *   **Batch Training:**
        *   Use the `learn()` method on a `Model` instance.
        *   Can be used to incrementally train a model on multiple batches of data.
        *   Example:
            ```python
            model = tb.HoeffdingTreeClassifier(n_classes=2)
            trained_model = model.learn(features, label)  # Train on initial data
            new_trained_model = trained_model.learn(new_features, new_label)  # Update with new data
            ```
    *   **Streaming Training:**
        *   Models are continuously updated as new data arrives.
        *   Enabled by default when deploying a model with `deploy()`.
        *   Can be configured with different update strategies:
            *   **Online:** Update on every data point.
            *   **Trigger-based:** Update based on volume, time, performance, or drift (*mentioned in `intro`, but specific configuration details are limited in the provided documentation*).
* **Model Deployment:**
    *   `model.deploy(name, input, labels, predict_only=False)`: Deploys a model to the TurboML platform.
        *   `name`: A unique name for the deployed model.
        *   `input`: An `OnlineInputs` object defining the input features.
        *   `labels`: An `OnlineLabels` object defining the target labels.
        *   `predict_only`: If `True`, the model will not be updated with new data (useful for batch-trained models).
    *   Returns a `DeployedModel` instance.
* **Model Retrieval:**
    *    `tb.retrieve_model(model_name: str)`: Fetches a reference to an already deployed model, allowing interaction in a new workspace/environment without redeployment.

**4. Model Evaluation and Monitoring (MLOps):**

*   **Evaluation Metrics:**
    *   **Built-in Metrics:**
        *   `WindowedAUC`: Area Under the ROC Curve (for classification).
        *   `WindowedAccuracy`: Accuracy (for classification).
        *   `WindowedMAE`: Mean Absolute Error (for regression).
        *   `WindowedMSE`: Mean Squared Error (for regression).
        *   `WindowedRMSE`: Root Mean Squared Error (for regression).
    *   **Custom Metrics:**
        *   Define custom aggregate metrics using Python classes that inherit from `ModelMetricAggregateFunction`.
        *   Implement `create_state`, `accumulate`, `retract` (optional), `merge_states`, and `finish` methods.
    *   **Continuous Evaluation:** Metrics are calculated continuously as new data arrives.
    *   **Functions:**
        *   `deployed_model.add_metric(metric_name)`: Registers a metric for a deployed model.
        *   `deployed_model.get_evaluation(metric_name, filter_expression="", window_size=1000, limit=100000)`: Retrieves evaluation results for a specific metric. The `filter_expression` allows filtering data based on SQL expressions.
        *   `tb.evaluation_metrics()`: Lists available built-in metrics.
        *   `tb.register_custom_metric(metric_name, metric_class)`: Registers a custom metric.
        *   `tb.compare_model_metrics(models, metric)`: Compares multiple models on a given metric (generates a Plotly plot).
*   **Drift Detection:**
    *   **Univariate Drift:** Detects changes in the distribution of individual features.
        *   Uses the Adaptive Windowing (ADWIN) method by default.
        *   Register with `dataset.register_univariate_drift(numerical_field, label=None)`.
        *   Retrieve with `dataset.get_univariate_drift(label=None, numerical_field=None)`.
    *   **Multivariate Drift:** Detects changes in the joint distribution of multiple features.
        *   Uses PCA-based reconstruction by default.
        *   Register with `dataset.register_multivariate_drift(label, numerical_fields)`.
        *   Retrieve with `dataset.get_multivariate_drift(label)`.
    *   **Model Drift (Target Drift):** Detects changes in the relationship between input features and the target variable.
        *   Register with `deployed_model.add_drift()`.
        *   Retrieve with `deployed_model.get_drifts()`.
*   **Model Explanations:**
    *   Integration with the `iXAI` library for incremental model explanations.
    *   Provides insights into feature importance and model behavior.  The example shows using `IncrementalPFI` from `iXAI`.
*   **Model Management:**
    *   `deployed_model.pause()`: Pauses a running model.
    *   `deployed_model.resume()`: Resumes a paused model.
    *   `deployed_model.delete(delete_output_topic=True)`: Deletes a model and optionally its associated output data.
    *   `deployed_model.get_endpoints()`: Retrieves the API endpoints for a deployed model. These endpoints can be used for synchronous inference.
    *   `deployed_model.get_logs()`: Retrieves logs for a deployed model.
* **Inference:**
    *   **Async:**
        *   The data streamed from the input source is continuously fed to the model, and the outputs are streamed to another source.
        *   ```python
            outputs = deployed_model.get_outputs()
            ```
    *   **API:**
        *   A request-response model is used for inference on a single data point synchronously.
        *   The `/model_name/predict` endpoint is exposed for each deployed model where a REST API call can be made to get the outputs.
            ```python
            import requests
            resp = requests.post(model_endpoints[0], json=model_query_datapoint, headers=tb.common.api.headers)
            ```
    *   **Batch:**
        *   When you have multiple records you’d like to perform inference on, you can use the get_inference method as follows.
            ```python
            outputs = deployed_model.get_inference(query_df)
            ```

**5. Advanced Features:**

*   **Hyperparameter Tuning:**
    *   `tb.hyperparameter_tuning()`: Performs grid search to find the best combination of hyperparameters for a given model and dataset.
*   **Algorithm Tuning:**
    *   `tb.algorithm_tuning()`: Compares different models on a given dataset to identify the best-performing algorithm.
* **Local Model:**
    * The `LocalModel` class provides direct access to TurboML's machine learning models in Python, allowing for local training and prediction without deploying to the platform.
    * Useful for offline experimentation and development.

**Code Structure and Key Modules:**

*   **`turboml`:** The main package, providing access to all core functionalities.
*   **`common`:**
    *   `api.py`: Handles communication with the TurboML backend API. Includes `ApiException`, `NotFoundException`, and retry logic.
    *   `dataloader.py`: Functions for data loading, serialization, and interaction with the Arrow server. Includes `get_proto_msgs`, `create_protobuf_from_row_tuple`, `upload_df`, and more.
    *   `datasets.py`: Defines dataset classes (`LocalDataset`, `OnlineDataset`, standard datasets like `FraudDetectionDatasetFeatures`).
    *   `feature_engineering.py`: Classes and functions for defining and managing features (`FeatureEngineering`, `LocalFeatureEngineering`, `IbisFeatureEngineering`).
    *   `internal.py`: Internal utilities (e.g., `TbPyArrow` for PyArrow helpers, `TbPandas` for Pandas helpers).
    *   `llm.py`: Functions for working with Large Language Models (LLMs), including acquiring models from Hugging Face and spawning LLM servers.
    *   `ml_algs.py`: Core ML algorithms, model deployment logic, and model classes (`Model`, `DeployedModel`, `LocalModel`). Includes functions like `ml_modelling`, `model_learn`, `model_predict`, `get_score_for_model`, and model classes for various algorithms.
    *   `models.py`: Pydantic models for data structures and API requests/responses. Defines core data structures like `DatasetSchema`, `InputSpec`, `ModelConfigStorageRequest`, etc.
    *   `namespaces.py`: Functions for managing namespaces.
    *   `pymodel.py`: Python bindings for the underlying C++ model implementation (using `turboml_bindings`).
    *   `pytypes.py`: Python type definitions (using `turboml_bindings`).
    *   `udf.py`: Base class for defining custom metric aggregation functions (`ModelMetricAggregateFunction`).
    *   `util.py`: General utility functions (e.g., type conversions, timestamp handling, Ibis utilities).
    *   `default_model_configs.py`: Default configurations for ML algorithms.
    *   `model_comparison.py`: Functions for comparing model performance (e.g., `compare_model_metrics`).
    *   `env.py`: Environment configuration (e.g., server addresses).
*   **`protos`:**
    *   Contains Protobuf definitions (`.proto` files) and generated Python code (`_pb2.py`, `_pb2.pyi`) for data structures used in communication with the backend. Key files include `config.proto`, `input.proto`, `output.proto`, `metrics.proto`.
*   **`wyo_models`:**
    *   Contains examples of how to implement custom models using Python.
*   **Subpackages for Algorithms:**
    *   `regression`, `classification`, `anomaly_detection`, `ensembles`, `general_purpose`, `forecasting`: Contain specific ML algorithm implementations.
*   **`pre_deployment_ml`:**
    *   Contains modules for pre-deployment tasks like hyperparameter tuning and algorithm tuning.
*   **`post_deployment_ml`:**
    *   Contains modules for post-deployment tasks like drift detection, model explanations, and custom metrics.
*   **`general_examples`:**
    *   Contains examples of using TurboML features.
*   **`non_numeric_inputs`:**
    *   Contains examples of using non-numeric inputs like images and text.
*   **`byo_models`:**
    *   Contains examples of bringing your own models.
*   **`llms`:**
    *   Contains modules and examples for using LLMs.

**Conclusion:**
TurboML provides a robust and flexible platform for building and deploying real-time machine learning applications. Its strengths lie in its streaming-first architecture, support for diverse data sources and ML algorithms, comprehensive feature engineering capabilities (including SQL, aggregations, UDFs, and Ibis integration), and built-in MLOps features like model monitoring, drift detection, and evaluation. The platform's Python SDK and support for BYOM (Bring Your Own Model) via ONNX, gRPC, REST API, and custom Python code make it adaptable to a wide range of use cases and existing workflows. The combination of ease of use, performance, and real-time capabilities makes TurboML a powerful tool for developing and managing data-driven applications.