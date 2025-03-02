# What is TurboML?
@ TurboML - page_link: https://docs.turboml.com/intro/
<page_content>
Introduction

TurboML is a machine learning platform that’s reinvented for real-time. What does that mean? All the steps in the ML lifecycle, from data ingestion, to feature engineering, to ML modelling to post deployment steps like monitoring, are all designed so that in addition to batch data, they can also handle real-time data.

## Data Ingestion [Permalink for this section](https://docs.turboml.com/intro/\#data-ingestion)

The first step is to bring your data to the TurboML platform. There are two major ways to ingest your data. Pull-based and Push-based.

### Pull-based ingestion [Permalink for this section](https://docs.turboml.com/intro/\#pull-based-ingestion)

With this approach, you use TurboML’s prebuilt connectors to connect to your data source. The connectors will continuously pull data from your data source, and ingest it into TurboML.

### Push-based ingestion [Permalink for this section](https://docs.turboml.com/intro/\#push-based-ingestion)

Sometimes, you might not want to send data via an intermediate data source, but rather directly send the data. Push-based ingestion can be used for this, where data can be send either via REST API calls, or using more performant client SDKs. Here’s an example with a Pandas DataFrame

```transactions = tb.PandasDataset(dataset_name="transactions",dataframe=df, upload=True)
transactions.configure_dataset(key_field="index")
```

## Feature Engineering [Permalink for this section](https://docs.turboml.com/intro/\#feature-engineering)

Feature engineering is perhaps the most important step for data scientists. TurboML provides several different interfaces to define features. We’ve designed the feature engineering experience in a way so that after you’ve defined a feature, you can see that feature computed for your local data. This should help debug and iterate faster. Once you’re confident about a feature definition, you can deploy it where it’ll be continuously computed on the real-time data. Once deployed, these features are automatically computed on the streaming data. And we have retrieval APIs to compute it for ad-hoc queries.

### SQL Features [Permalink for this section](https://docs.turboml.com/intro/\#sql-features)

Writing SQL queries is one of the most common way to define ML features. TurboML supports writing arbitrary SQL expressions to enable such features. Here’s an example with a simple SQL feature.

```transactions.feature_engineering.create_sql_features(
    sql_definition='"transactionAmount" + "localHour"',
    new_feature_name="my_sql_feat",
)
```

Notice that the column names are in quotes.

And here’s a more complicated example

```transactions.feature_engineering.create_sql_features(
    sql_definition='CASE WHEN "paymentBillingCountryCode" <> "ipCountryCode" THEN 1 ELSE 0 END ',
    new_feature_name="country_code_match",
)
```

### Aggregate Features [Permalink for this section](https://docs.turboml.com/intro/\#aggregate-features)

A common template for real-time features is aggregating some value over some time window. To define such time-windowed aggregations, you first need to register a timestamp column for your dataset. This can be done as follows,

```transactions.feature_engineering.register_timestamp(column_name="timestamp", format_type="epoch_seconds")
```

The supported formats can be found out using

```tb.get_timestamp_formats()
```

Once the timestamp is registered, we can create a feature using

```transactions.feature_engineering.create_aggregate_features(
    column_to_operate="transactionAmount",
    column_to_group="accountID",
    operation="SUM",
    new_feature_name="my_sum_feat",
    timestamp_column="timestamp",
    window_duration=24,
    window_unit="hours"
)
```

### User Defined Features [Permalink for this section](https://docs.turboml.com/intro/\#user-defined-features)

We understand why data scientists love Python - the simplicity, the ecosystem - is unmatchable. Guess what? You can use native Python, importing any library, [to define features](https://docs.turboml.com/feature_engineering/udf/)!

### IBIS Features [Permalink for this section](https://docs.turboml.com/intro/\#ibis-features)

For streaming features that are more complex than just windowed aggregations, can be defined using the [ibis interface](https://docs.turboml.com/feature_engineering/advanced/ibis_feature_engineering/). They can then be executed using Apache Flink or RisingWave.

### Feature Retrieval [Permalink for this section](https://docs.turboml.com/intro/\#feature-retrieval)

As mentioned before, once deployed, the feature computation is automatically added to the real-time streaming pipeline. However, feature values can also be retrieved on ad-hoc data using the retrieval API. Here’s an example

```features = tb.retrieve_features("transactions", query_df)
```

## ML Modelling - Basic concepts [Permalink for this section](https://docs.turboml.com/intro/\#ml-modelling---basic-concepts)

### Inputs and Labels [Permalink for this section](https://docs.turboml.com/intro/\#inputs-and-labels)

For each model, we need to specify the Inputs and the Labels.

### Types of fields [Permalink for this section](https://docs.turboml.com/intro/\#types-of-fields)

Different models can accept different types of input fields. The supported types of fields are, numeric, categoric, time series, text, and image.

### TurboML algorithms [Permalink for this section](https://docs.turboml.com/intro/\#turboml-algorithms)

TurboML provides several algorithms out of the box. These algorithms are optimized for online predictions and learning, and have been tested on real-world settings.

```model = tb.HoeffdingTreeClassifier(n_classes=2)
```

### Pytorch/TensorFlow/Scikit-learn [Permalink for this section](https://docs.turboml.com/intro/\#pytorchtensorflowscikit-learn)

We use ONNX to deploy trained models from [Pytorch](https://docs.turboml.com/byo_models/onnx_pytorch/), [TensorFlow](https://docs.turboml.com/byo_models/onnx_tensorflow/), [Scikit-learn](https://docs.turboml.com/byo_models/onnx_sklearn/) or other ONNX compatible frameworks. Example for these three frameworks can be found in the following notebooks.

Note: These models are static, and are not updated automatically.

### Python [Permalink for this section](https://docs.turboml.com/intro/\#python)

TurboML also supports writing arbitrary Python code to define your own algorithms, including any libraries. To add your own algorithms, you need to define a Python class with 2 methods defined with the following signature:

```class Model:
    def learn_one(self, features, label):
        pass

    def predict_one(self, features, output_data):
        pass
```

Examples of using an incremental learning algorithm, as well as a batch-like algorithm, can be found [here](https://docs.turboml.com/wyo_models/native_python_model/) from the river library.

### Combining models [Permalink for this section](https://docs.turboml.com/intro/\#combining-models)

Models can also be combined to create other models, e.g. ensembles. An example of an ensemble model is as follows

```model = tb.LeveragingBaggingClassifier(n_classes=2, base_model = tb.HoeffdingTreeClassifier(n_classes=2))
```

Preprocessors can also be chained and applied in a similar manner. E.g.

```model = tb.MinMaxPreProcessor(base_model = model)
```

## Model Training [Permalink for this section](https://docs.turboml.com/intro/\#model-training)

Once we’ve defined a model, it can be trained in different ways.

### Batch way [Permalink for this section](https://docs.turboml.com/intro/\#batch-way)

The simplest way is to train the model in a batch way. This is similar to sklearn’s fit() method. However, internally the training is performed in an incremental manner. So, you can update an already trained model on some new data too. Here’s an example

```old_trained_model = model.learn(old_features, old_label)
new_trained_model = old_trained_model.learn(new_features, new_label)
```

Any trained copy of the model can be then deployed to production.

```deployed_model = new_trained_model.deploy(name = "deployment_name", input=features, labels=label, predict_only=True)
```

Since this is a trained model, we can also invoke this model in a batch way to get predictions without actually deploying the mode.

```outputs = new_trained_model.predict(query_features)
```

### Streaming way [Permalink for this section](https://docs.turboml.com/intro/\#streaming-way)

This is where the model, after deployment, is continuously trained on new data. The user can choose how to update the model. The choices are online updates (where the model is updated on every new datapoint), or trigger-based updates which can be volume-based, time-based, performance-based or drift-based. The default option is online updates.

```deployed_model = model.deploy(name = "deployment_name", input=features, labels=label)
```

## Deployment and MLOps [Permalink for this section](https://docs.turboml.com/intro/\#deployment-and-mlops)

### Inference [Permalink for this section](https://docs.turboml.com/intro/\#inference)

Once you’ve deployed a mode, there are several different ways to perform inference.

#### Async [Permalink for this section](https://docs.turboml.com/intro/\#async)

The first one is the async method. The data that is streamed from the input source is continuously fed to the model, and the outputs are streamed to another source. This stream can be either be subscribed to directly be the end user application, or sinked to a database or other data sources.

```outputs = deployed_model.get_outputs()
```

#### API [Permalink for this section](https://docs.turboml.com/intro/\#api)

A request-response model is used for inference on a single data point synchronously. The `/model_name/predict` endpoint is exposed for each deployed model where a REST API call can be made to get the outputs.

#### Batch [Permalink for this section](https://docs.turboml.com/intro/\#batch)

When you have multiple records you’d like to perform inference on, you can use the get\_inference method as follows.

```outputs = deployed_model.get_inference(query_df)
```

### Evaluation [Permalink for this section](https://docs.turboml.com/intro/\#evaluation)

TurboML provides standard ML metrics out of the box to perform model evaluation. Multiple metrics can be registered for any deployed model. The metrics pipeline re-uses the labels extracted for model training.

```deployed_model.add_metric("WindowedAUC")
model_auc_scores = deployed_model.get_evaluation("WindowedAUC")
```

Last updated on January 24, 2025

[Quickstart](https://docs.turboml.com/quickstart/ "Quickstart")

What is TurboML? @ TurboML
</page_content>

# TurboML Quickstart 
@ TurboML - page_link: https://docs.turboml.com/quickstart/
<page_content>
Quickstart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/quickstart.ipynb)

```import turboml as tb
```

## Inspecting Data [Permalink for this section](https://docs.turboml.com/quickstart/\#inspecting-data)

TurboML is built for real-time machine learning, and as such, deals with streams of data. This can be achieved by using connectors to continuously pull data from your data source (like S3 or postgres), or use push-based approaches using REST API or Client SDKs.

For the purpose of this tutorial, we can use simulate real-time data generation, with a batch-like setting using pandas dataframes. Let's first load some pandas dataframes. In this example, we're using a credit card fraud detection dataset.

```transactions_df = tb.datasets.FraudDetectionDatasetFeatures().df
labels_df = tb.datasets.FraudDetectionDatasetLabels().df
```

```transactions_df
```

```labels_df
```

Our dataset has 201406 datapoints, each with a corresponding label. Since we don't have a natural primary key in the dataset that can uniquely identify each row, we'll use the inbuilt index that pandas provides.

```transactions_df.head()
```

```labels_df.head()
```

## Data Ingestion [Permalink for this section](https://docs.turboml.com/quickstart/\#data-ingestion)

We can now upload these dataframes to the TurboML platform, the **OnlineDataset** class can be used here. It takes in the dataframe, the primary key, and the name of the dataset that is to be created for the given dataframe as input.

```# Attempt to create and upload dataset
transactions = tb.OnlineDataset.from_pd(
    id="qs_transactions",
    df=transactions_df,
    key_field="transactionID",
    load_if_exists=True,
)
labels = tb.OnlineDataset.from_pd(
    id="qs_transaction_labels",
    df=labels_df,
    key_field="transactionID",
    load_if_exists=True,
)
```

## Feature Engineering [Permalink for this section](https://docs.turboml.com/quickstart/\#feature-engineering)

TurboML platform facilitates transformations on raw data to produce new features. You can use the jupyter notebook as a "playground" to explore different features. This involves 3 steps.

- **fetch data**: Experimentation is easier on static data. Since TurboML works with continuous data streams, to enable experimentation we fetch a snapshot or a subset of data in the jupyter notebook.
- **add feature definitions**: Now that we have a static dataset, we can define multiple different features, and see their values on this dataset. Since we can observe their values, we can perform simple experiments and validations like correlations, plots and other exploratory analysis.
- **submit feature definitions**: Once we're confident about the features we've defined, we can now submit the ones we want TurboML to compute continuously for the actual data stream.

### Fetch data [Permalink for this section](https://docs.turboml.com/quickstart/\#fetch-data)

We can use the **get\_features** function to get a snapshot or subset of the data stream.

**Note**: This size of the dataset returned by this function can change on each invocation. Also, the dataset is not guaranteed to be in the same order.

### Add feature definitions [Permalink for this section](https://docs.turboml.com/quickstart/\#add-feature-definitions)

To add feature definitions, we have a class from turboml package called **FeatureEngineering**. This allows us to define SQL-based and dynamic aggregation-based features.

The following cell shows how to define an SQL-based feature. The sql\_definition parameter in the **create\_sql\_features** function takes in the SQL expression to be used to prepare the feature. It returns a dataframe with all the original columns, and another column which, on a high-level is defined as `SELECT sql_definition AS new_feature_name FROM dataframe`.

```transactions.feature_engineering.create_sql_features(
    sql_definition='"transactionAmount" + "localHour"',
    new_feature_name="my_sql_feat",
)
```

```transactions.feature_engineering.get_local_features()
```

```tb.get_timestamp_formats()
```

```transactions.feature_engineering.register_timestamp(
    column_name="timestamp", format_type="epoch_seconds"
)
```

The following cell shows how to define an aggregation-based feature using the **create\_aggregate\_features** function. It returns a dataframe with all the original columns, and another column which, on a high-level is defined as `SELECT operation(column_to_operate) OVER (PARTITION BY column_to_group ORDER BY time_column RANGE BETWEEN INTERVAL window_duration PRECEDING AND CURRENT ROW) as new_feature_name from dataframe`.

```transactions.feature_engineering.create_aggregate_features(
    column_to_operate="transactionAmount",
    column_to_group="accountID",
    operation="SUM",
    new_feature_name="my_sum_feat",
    timestamp_column="timestamp",
    window_duration=24,
    window_unit="hours",
)
```

```transactions.feature_engineering.get_local_features()
```

### Submit feature definitions [Permalink for this section](https://docs.turboml.com/quickstart/\#submit-feature-definitions)

Now that we've seen the newly created features, and everything looks good, we can submit these feature definitions to the TurboML platform so that this can be computed continously for the input data stream.

We need to tell the platform to start computations for all pending features for the given dataset. This can be done by calling the **materialize\_features** function.

```transactions.feature_engineering.materialize_features(["my_sql_feat", "my_sum_feat"])
```

```df_transactions = transactions.feature_engineering.get_materialized_features()
df_transactions
```

## Machine Learning Modelling [Permalink for this section](https://docs.turboml.com/quickstart/\#machine-learning-modelling)

TurboML provides out of the box algorithms, optimized for real-time ML, and supports bringing your own models and algorithms as well. In this tutorial, we'll use the algorithms provided by TurboML.

### Check the available algorithms [Permalink for this section](https://docs.turboml.com/quickstart/\#check-the-available-algorithms)

You can check what are the available ML algorithms based on `tb.ml_algorithms(have_labels=True/False)` depending on supervised or unsupervised learning.

```tb.ml_algorithms(have_labels=False)
```

Let's use the RandomCutForest (RCF) algorithm.

### Create model [Permalink for this section](https://docs.turboml.com/quickstart/\#create-model)

Now that we've chosen an algorithm, we need to create a model.

```model = tb.RCF(number_of_trees=50)
```

### Run Streaming ML jobs [Permalink for this section](https://docs.turboml.com/quickstart/\#run-streaming-ml-jobs)

Now that we've instantiated the model, we can deploy it using the **deploy** function.
For an unsupervised ML job, we need to provide a dataset from which the model can consume inputs. For each record in this dataset, the model will make a prediction, produce the prediction to an output dataset, and then perform unsupervised updates using this record.

There are four types of fields that can be used by any ML algorithm:

- numerical\_fields: This represents fields that we want our algorithm to treat as real-valued fields.
- categorical\_fields: This represents fields that we want our algorithm to treat as categorical fields.
- time\_field: This is used for time-series applications to capture the timestamp field.
- textual\_fields: This represents fields that we want our algorithm to treat as text fields.

The input values from any of these fields are suitably converted to the desired type. String values are converted using the hashing trick.

Let's construct a model config using the following numerical fields, no categorical or time fields.

```numerical_fields = [\
    "transactionAmount",\
    "localHour",\
    "my_sum_feat",\
    "my_sql_feat",\
]
features = transactions.get_model_inputs(numerical_fields=numerical_fields)
label = labels.get_model_labels(label_field="is_fraud")
```

```deployed_model_rcf = model.deploy(name="demo_model_rcf", input=features, labels=label)
```

### Inspect model outputs [Permalink for this section](https://docs.turboml.com/quickstart/\#inspect-model-outputs)

We can now fetch the outputs that the model produced by calling the **get\_outputs** function.

**Note**: This size of the outputs returned by this function can change on each invocation, since the model is continuosly producing outputs.

```outputs = deployed_model_rcf.get_outputs()
```

```len(outputs)
```

```sample_output = outputs[-1]
sample_output
```

The above output corresponds to an input with the key, or index, sample\_output.key. Along with the anomaly score, the output also contains attributions to different features. We can see that the first numerical feature, i.e. 'transactionAmount' is around sample\_output.feature\_score\[0\]\*100% responsible for the anomaly score

```import matplotlib.pyplot as plt

plt.plot([output["record"].score for output in outputs])
```

### Model Endpoints [Permalink for this section](https://docs.turboml.com/quickstart/\#model-endpoints)

The above method of interacting with the model was asynchronous. We were adding our datapoints to an input dataset, and getting the corresponding model outputs in an output dataset. In some scenarios, we need a synchronous method to query the model. This is where we can use the model endpoints that TurboML exposes.

```model_endpoints = deployed_model_rcf.get_endpoints()
model_endpoints
```

Now that we know what endpoint to send the request to, we now need to figure out the right format. Let's try to make a prediction on the last row from our input dataset.

```model_query_datapoint = transactions_df.iloc[-1].to_dict()
model_query_datapoint
```

```import requests

resp = requests.post(
    model_endpoints[0], json=model_query_datapoint, headers=tb.common.api.headers
)
```

```resp.json()
```

### Batch Inference on Models [Permalink for this section](https://docs.turboml.com/quickstart/\#batch-inference-on-models)

While the above method is more suited for individual requests, we can also perform batch inference on the models. We use the **get\_inference** function for this purpose.

```outputs = deployed_model_rcf.get_inference(transactions_df)
outputs
```

## Model Evaluation [Permalink for this section](https://docs.turboml.com/quickstart/\#model-evaluation)

Similar to ML models, TurboML provides in-built metrics, and supports defining your own metrics. Let's see the available metrics.

```tb.evaluation_metrics()
```

We can select the AreaUnderCurve (AUC) metric to evaluate our anomaly detection model. The windowed prefix means we're evaluating these metrics over a rolling window. By default, the window size is `1000`.

```deployed_model_rcf.add_metric("WindowedAUC")
```

Similar to steps like feature engineering and ML modelling, model evaluation is also a continuosly running job. We can look at the snapshot of the model metrics at any given instance by using the **get\_evaluation** function.

**Note**: This size of the outputs returned by this function can change on each invocation, since we're continuously evaluating the model.

```model_auc_scores = deployed_model_rcf.get_evaluation("WindowedAUC")
model_auc_scores[-1]
```

```import matplotlib.pyplot as plt

plt.plot([model_auc_score.metric for model_auc_score in model_auc_scores])
```

### Model Evaluation with filter and custom window size [Permalink for this section](https://docs.turboml.com/quickstart/\#model-evaluation-with-filter-and-custom-window-size)

We support running evaluation on filtered model data using valid SQL expression along with custom window size.

```model_auc_scores = deployed_model_rcf.get_evaluation(
    "WindowedAUC",
    filter_expression="input_data.transactionCurrencyCode != 'USD' AND output_data.score > 0.6",
    window_size=200,
)
model_auc_scores[-1]
```

```import matplotlib.pyplot as plt

plt.plot([model_auc_score.metric for model_auc_score in model_auc_scores])
```

## Supervised Learning [Permalink for this section](https://docs.turboml.com/quickstart/\#supervised-learning)

Let's now take an example with a supervised learning algorithm. First, let's see what algorithms are supported out of the box.

```tb.ml_algorithms(have_labels=True)
```

We can use HoeffdingTreeClassifier to try to classify fraudulent and normal activity on the same dataset. First, we need to instantiate a model.

```htc_model = tb.HoeffdingTreeClassifier(n_classes=2)
```

We can use the same numerical fields in this model as well. However, let's add some categorical fields as well.

```categorical_fields = [\
    "digitalItemCount",\
    "physicalItemCount",\
    "isProxyIP",\
]
features = transactions.get_model_inputs(
    numerical_fields=numerical_fields, categorical_fields=categorical_fields
)
label = labels.get_model_labels(label_field="is_fraud")
```

### Run Supervised ML jobs [Permalink for this section](https://docs.turboml.com/quickstart/\#run-supervised-ml-jobs)

Same as before, we can deploy this model with the **deploy** function.

```deployed_model_htc = htc_model.deploy("demo_classifier", input=features, labels=label)
```

We can now inspect the outputs.

```outputs = deployed_model_htc.get_outputs()
```

```len(outputs)
```

```sample_output = outputs[-1]
sample_output
```

We notice that since this is a classification model, we have some new attributes in the output, specifically `class_probabilities` and `predicted_class`. We also have the `score` attribute which, for classification, just shows us the probability for the last class.

### Supervised Model Endpoints [Permalink for this section](https://docs.turboml.com/quickstart/\#supervised-model-endpoints)

Predict API for supervised models is exactly the same as unsupervised models.

```model_endpoints = deployed_model_htc.get_endpoints()
model_endpoints
```

```resp = requests.post(
    model_endpoints[0], json=model_query_datapoint, headers=tb.common.api.headers
)
resp.json()
```

### Supervised Model Evaluation [Permalink for this section](https://docs.turboml.com/quickstart/\#supervised-model-evaluation)

Let's now evaluate our supervised ML model. The process is exactly the same as for unsupervised model evaluation.

```deployed_model_htc.add_metric("WindowedAUC")
```

We can use the same **get\_evaluation** function to fetch the metrics for this model as well. Remember, this function retrieves the metric values present at that moment of time. So, if the number of records recieved seem low, just re-run this function.

```model_auc_scores = deployed_model_htc.get_evaluation("WindowedAUC")
model_auc_scores[-1]
```

```plt.plot([model_auc_score.metric for model_auc_score in model_auc_scores])
```

## Model Comparison [Permalink for this section](https://docs.turboml.com/quickstart/\#model-comparison)

Now that we have 2 models deployed, and we've registered metrics for both of them, we can compare them on real-time data. On each invocation, the following function will fetch the latest evaluations of the models and plot them.

```tb.compare_model_metrics(
    models=[deployed_model_rcf, deployed_model_htc], metric="WindowedAUC"
)
```

## Model Deletion [Permalink for this section](https://docs.turboml.com/quickstart/\#model-deletion)

We can delete the models like this, by default the generated output is deleted. If you want to retain the output generated by model, use `delete_output_topic=False`.

```deployed_model_rcf.delete()
```

Last updated on January 24, 2025

[Introduction](https://docs.turboml.com/intro/ "Introduction") [Batch API](https://docs.turboml.com/general_examples/batch_api/ "Batch API")
</page_content>

# String Encoding 
@ TurboML - page_link: https://docs.turboml.com/non_numeric_inputs/string_encoding/
<page_content>
Non-Numeric Inputs

String Encoding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/string_encoding.ipynb)

Textual data needs to be converted into numerical data to be used by ML models. For larger textual data like sentences and paragraphs, we saw in llm\_embedding notebook how embeddings from pre-trained languages models can be used. But what about smaller strings, like country name? How do we use such strings as features in our ML models? This notebook covers different encoding methods that TurboML provides for textual features.

```import turboml as tb
```

```transactions = tb.datasets.FraudDetectionDatasetFeatures().to_online(
    id="transactions", load_if_exists=True
)
labels = tb.datasets.FraudDetectionDatasetLabels().to_online(
    id="transaction_labels", load_if_exists=True
)
```

```numerical_fields = [\
    "transactionAmount",\
]
textual_fields = ["transactionCurrencyCode"]
features = transactions.get_model_inputs(
    numerical_fields=numerical_fields, textual_fields=textual_fields
)
label = labels.get_model_labels(label_field="is_fraud")
```

Notice that now we're extracting a textual feature called transactionCurrencyCode from our dataset. To make sure that the model finally works with numerical data, we can define preprocessors that transform the textual data to numerical data via some encoding methods. By default, TurboML uses the hashing trick ( [https://en.wikipedia.org/wiki/Feature\_hashing (opens in a new tab)](https://en.wikipedia.org/wiki/Feature_hashing)) to automatically hash and convert string data to numeric data. However, TurboML also supports popular encoding methods to handle strings including

- LabelPreProcessor
- OneHotPreProcessor
- TargetPreProcessor
- FrequencyPreProcessor
- BinaryPreProcessor

We'll try an example using FrequencyPreProcessor. For these pre-processors, we need to specify in advance the cardinality of our data, which can be computed as follows.

```htc_model = tb.HoeffdingTreeClassifier(n_classes=2)
```

```import pandas as pd

demo_classifier = tb.FrequencyPreProcessor(
    text_categories=[\
        len(pd.unique(transactions.preview_df[col])) for col in textual_fields\
    ],
    base_model=htc_model,
)
```

```deployed_model = demo_classifier.deploy(
    "demo_classifier_htc", input=features, labels=label
)
```

```outputs = deployed_model.get_outputs()
```

```sample_output = outputs[-1]
sample_output
```

```
```

Last updated on January 24, 2025

[LLM Tutorial](https://docs.turboml.com/llms/turboml_llm_tutorial/ "LLM Tutorial") [Image Input](https://docs.turboml.com/non_numeric_inputs/image_input/ "Image Input")
</page_content>

# AMF Regressor 
@ TurboML - page_link: https://docs.turboml.com/regression/amfregressor/
<page_content>
Regression

AMF Regressor

**Aggregated Mondrian Forest** regressor for online learning.

This algorithm is truly online, in the sense that a single pass is performed, and that predictions can be produced anytime.

Each node in a tree predicts according to the average of the labels it contains. The prediction for a sample is computed as the aggregated predictions of all the subtrees along the path leading to the leaf node containing the sample. The aggregation weights are exponential weights with learning rate `step` using a squared loss when `use_aggregation` is `True`.

This computation is performed exactly thanks to a context tree weighting algorithm. More details can be found in the original paper[1](https://docs.turboml.com/regression/amfregressor/#user-content-fn-1).

The final predictions are the average of the predictions of each of the `n_estimators` trees in the forest.

## Parameters [Permalink for this section](https://docs.turboml.com/regression/amfregressor/\#parameters)

- **n\_estimators**( `int`, Default: `10`) → The number of trees in the forest.

- **step** ( `float`, Default: `1.0`) → Step-size for the aggregation weights.

- **use\_aggregation**( `bool`, Default: `True`) → Controls if aggregation is used in the trees. It is highly recommended to leave it as `True`.

- **seed**( `int` \| `None`, Default: `None`) → Random seed for reproducibility.


## Example Usage [Permalink for this section](https://docs.turboml.com/regression/amfregressor/\#example-usage)

We can create an instance of the AMF Regressor model like this.

```import turboml as tb
amf_model = tb.AMFRegressor()
```

## Footnotes [Permalink for this section](https://docs.turboml.com/regression/amfregressor/\#footnote-label)

1. Mourtada, J., Gaïffas, S., & Scornet, E. (2021). AMF: Aggregated Mondrian forests for online learning. Journal of the Royal Statistical Society Series B: Statistical Methodology, 83(3), 505-533. [↩](https://docs.turboml.com/regression/amfregressor/#user-content-fnref-1)


Last updated on January 24, 2025

[Random Cut Forest](https://docs.turboml.com/anomaly_detection/rcf/ "Random Cut Forest") [FFM Regressor](https://docs.turboml.com/regression/ffmregressor/ "FFM Regressor")
</page_content>

# MultinomialNB 
@ TurboML - page_link: https://docs.turboml.com/classification/multinomialnb/
<page_content>
Classification

Multinomial Naive Bayes


Naive Bayes classifier for multinomial models.

Multinomial Naive Bayes model learns from occurrences between features such as word counts and discrete classes. The input vector must contain positive values, such as counts or TF-IDF values.

## Parameters [Permalink for this section](https://docs.turboml.com/classification/multinomialnb/\#parameters)

- **n\_classes**( `int`) → The number of classes for the classifier.

- **alpha**(Default: `1.0`) → Additive (Laplace/Lidstone) smoothing parameter (use 0 for no smoothing).


## Example Usage [Permalink for this section](https://docs.turboml.com/classification/multinomialnb/\#example-usage)

We can create an instance and deploy Multinomial NB model like this.

```import turboml as tb
model = tb.MultinomialNB(n_classes=2)
```

Last updated on January 24, 2025

[Gaussian Naive Bayes](https://docs.turboml.com/classification/gaussiannb/ "Gaussian Naive Bayes") [Hoeffding Tree Classifier](https://docs.turboml.com/classification/hoeffdingtreeclassifier/ "Hoeffding Tree Classifier")
</page_content>

# PreProcessors
@ TurboML - page_link: https://docs.turboml.com/pipeline_components/preprocessors/
<page_content>
Pipeline Components

PreProcessors

Since our preprocessors must also work with streaming data, we define preprocessors by combining them with a base model. Under the hood, we apply the transformation by the preprocessor, and pass the transformed inputs to the base model. This concept is similar to `Pipelines` in Scikit-Learn.

## MinMaxPreProcessor [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#minmaxpreprocessor)

Works on numerical fields of the input. Scales them between 0 and 1, by maintaining running min and max for all numerical features.

### Parameters [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#parameters)

- **base\_model**( `Model`) → The model to call after transforming the input.

### Example Usage [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#example-usage)

We can create an instance of the MinMaxPreProcessor model like this.

```import turboml as tb
embedding = tb.MinMaxPreProcessor(base_model=tb.HoeffdingTreeClassifier(n_classes=2))
```

## NormalPreProcessor [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#normalpreprocessor)

Works on numerical fields of the input. Scales the data so that it has zero mean and unit variance, by maintaining running mean and variance for all numerical features.

### Parameters [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#parameters-1)

- **base\_model**( `Model`) → The model to call after transforming the input.

### Example Usage [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#example-usage-1)

We can create an instance of the NormalPreProcessor model like this.

```import turboml as tb
embedding = tb.NormalPreProcessor(base_model=tb.HoeffdingTreeClassifier(n_classes=2))
```

## RobustPreProcessor [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#robustpreprocessor)

Works on numerical fields of the input. Scales the data using statistics that are robust to outliers, by removing the running median and scaling by running interquantile range.

### Parameters [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#parameters-2)

- **base\_model**( `Model`) → The model to call after transforming the input.

### Example Usage [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#example-usage-2)

We can create an instance of the RobustPreProcessor model like this.

```import turboml as tb
embedding = tb.RobustPreProcessor(base_model=tb.HoeffdingTreeClassifier(n_classes=2))
```

## LabelPreProcessor [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#labelpreprocessor)

Works on textual fields of the input. For each textual feature, we need to know in advance the cardinality of that feature. Converts the strings into ordinal integers. The resulting numbers are appended to the numerical features.

### Parameters [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#parameters-3)

- **base\_model**( `Model`) → The model to call after transforming the input.

- **text\_categories**( `List[int]`) → List of cardinalities for each textual feature.


### Example Usage [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#example-usage-3)

We can create an instance of the LabelPreProcessor model like this.

```import turboml as tb
embedding = tb.LabelPreProcessor(text_categories=[5, 10], base_model=tb.HoeffdingTreeClassifier(n_classes=2))
```

## OneHotPreProcessor [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#onehotpreprocessor)

Works on textual fields of the input. For each textual feature, we need to know in advance the cardinality of that feature. Converts the strings into one-hot encoding. The resulting numbers are appended to the numerical features.

### Parameters [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#parameters-4)

- **base\_model**( `Model`) → The model to call after transforming the input.

- **text\_categories**( `List[int]`) → List of cardinalities for each textual feature.


### Example Usage [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#example-usage-4)

We can create an instance of the OneHotPreProcessor model like this.

```import turboml as tb
embedding = tb.OneHotPreProcessor(text_categories=[5, 10], base_model=tb.HoeffdingTreeClassifier(n_classes=2))
```

## BinaryPreProcessor [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#binarypreprocessor)

Works on textual fields of the input. For each textual feature, we need to know in advance the cardinality of that feature. Converts the strings into binary encoding. The resulting numbers are appended to the numerical features.

### Parameters [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#parameters-5)

- **base\_model**( `Model`) → The model to call after transforming the input.

- **text\_categories**( `List[int]`) → List of cardinalities for each textual feature.


### Example Usage [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#example-usage-5)

We can create an instance of the BinaryPreProcessor model like this.

```import turboml as tb
embedding = tb.BinaryPreProcessor(text_categories=[5, 10], base_model=tb.HoeffdingTreeClassifier(n_classes=2))
```

## FrequencyPreProcessor [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#frequencypreprocessor)

Works on textual fields of the input. For each textual feature, we need to know in advance the cardinality of that feature. Converts the strings into their frequency based on the values seen so far. The resulting numbers are appended to the numerical features.

### Parameters [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#parameters-6)

- **base\_model**( `Model`) → The model to call after transforming the input.

- **text\_categories**( `List[int]`) → List of cardinalities for each textual feature.


### Example Usage [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#example-usage-6)

We can create an instance of the FrequencyPreProcessor model like this.

```import turboml as tb
embedding = tb.FrequencyPreProcessor(text_categories=[5, 10], base_model=tb.HoeffdingTreeClassifier(n_classes=2))
```

## TargetPreProcessor [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#targetpreprocessor)

Works on textual fields of the input. For each textual feature, we need to know in advance the cardinality of that feature. Converts the strings into average target value seen for them so far. The resulting numbers are appended to the numerical features.

### Parameters [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#parameters-7)

- **base\_model**( `Model`) → The model to call after transforming the input.

- **text\_categories**( `List[int]`) → List of cardinalities for each textual feature.


### Example Usage [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#example-usage-7)

We can create an instance of the TargetPreProcessor model like this.

```import turboml as tb
embedding = tb.TargetPreProcessor(text_categories=[5, 10], base_model=tb.HoeffdingTreeClassifier(n_classes=2))
```

## LlamaCppPreProcessor [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#llamacpppreprocessor)

Works on textual fields of the input. Converts the text features into their embeddings obtained from a pre-trained language model. The resulting embeddings are appended to the numerical features.

### Parameters [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#parameters-8)

- **base\_model**( `Model`) → The model to call after transforming the input.

- **gguf\_model\_id**( `List[int]`) → A model id issued by `tb.acquire_hf_model_as_gguf`.

- **max\_tokens\_per\_input**( `int`) → The maximum number of tokens to consider in the input text. Tokens beyond this limit will be truncated. Default is 512.


### Example Usage [Permalink for this section](https://docs.turboml.com/pipeline_components/preprocessors/\#example-usage-8)

We can create an instance of the LlamaCppPreProcessor model like this.

```import turboml as tb
embedding = tb.LlamaCppPreProcessor(gguf_model_id=tb.acquire_hf_model_as_gguf("BAAI/bge-small-en-v1.5", "f16"), max_tokens_per_input=512, base_model=tb.HoeffdingTreeClassifier(n_classes=2))
```

Last updated on January 24, 2025

[One-Vs-Rest](https://docs.turboml.com/pipeline_components/ovr/ "One-Vs-Rest") [Embedding Model](https://docs.turboml.com/pipeline_components/embeddingmodel/ "Embedding Model")
</page_content>

# Python Model: Batch Example
@ TurboML - page_link: https://docs.turboml.com/wyo_models/batch_python_model/
<page_content>
Write Your Own Models

Batch Python Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/batch_python_model.ipynb)

In this example we emulate batch training of custom models defined using TurboML's `Python` model.

```import turboml as tb
```

```import pandas as pd
import numpy as np
```

## Model Definition [Permalink for this section](https://docs.turboml.com/wyo_models/batch_python_model/\#model-definition)

Here we define `MyBatchModel` with buffers to store the input features and labels until we exceed our buffer limit. Then, the model can be brained all at once on the buffered samples.

We use `Scikit-Learn`'s `Perceptron` for this task.

```from sklearn.linear_model import Perceptron
import turboml.common.pytypes as types


class MyBatchModel:
    def __init__(self):
        self.model = Perceptron()
        self.X_buffer = []
        self.y_buffer = []
        self.batch_size = 64
        self.trained = False

    def init_imports(self):
        from sklearn.linear_model import Perceptron
        import numpy as np

    def learn_one(self, input: types.InputData):
        self.X_buffer.append(input.numeric)
        self.y_buffer.append(input.label)

        if len(self.X_buffer) >= self.batch_size:
            self.model = self.model.partial_fit(
                np.array(self.X_buffer), np.array(self.y_buffer), classes=[0, 1]
            )

            self.X_buffer = []
            self.y_buffer = []

            self.trained = True

    def predict_one(self, input: types.InputData, output: types.OutputData):
        if self.trained:
            prediction = self.model.predict(np.array(input.numeric).reshape(1, -1))[0]

            output.set_predicted_class(prediction)
        else:
            output.set_score(0.0)
```

Now, we define a custom virtual environment with the correct list of dependencies which the model will be using, and link our model to this `venv`.

```venv = tb.setup_venv("my_batch_python_venv", ["scikit-learn", "numpy<2"])
venv.add_python_class(MyBatchModel)
```

## Model Deployment [Permalink for this section](https://docs.turboml.com/wyo_models/batch_python_model/\#model-deployment)

Once the virtual environment is ready, we prepare the dataset to be used in this task and deploy the model with its features and labels.

```batch_model = tb.Python(class_name=MyBatchModel.__name__, venv_name=venv.name)
```

```transactions = tb.datasets.FraudDetectionDatasetFeatures().to_online(
    "transactions", load_if_exists=True
)
labels = tb.datasets.FraudDetectionDatasetLabels().to_online(
    "transaction_labels", load_if_exists=True
)
```

```numerical_fields = [\
    "transactionAmount",\
    "localHour",\
    "isProxyIP",\
    "digitalItemCount",\
    "physicalItemCount",\
]
features = transactions.get_model_inputs(numerical_fields=numerical_fields)
label = labels.get_model_labels(label_field="is_fraud")
```

```deployed_batch_model = batch_model.deploy("batch_model", input=features, labels=label)
```

## Evaluation [Permalink for this section](https://docs.turboml.com/wyo_models/batch_python_model/\#evaluation)

```import matplotlib.pyplot as plt

deployed_batch_model.add_metric("WindowedRMSE")
model_auc_scores = deployed_batch_model.get_evaluation("WindowedRMSE")
plt.plot([model_auc_score.metric for model_auc_score in model_auc_scores])
```

Last updated on January 24, 2025

[Ensemble Python Model](https://docs.turboml.com/wyo_models/ensemble_python_model/ "Ensemble Python Model") [PySAD Example](https://docs.turboml.com/wyo_models/pysad_example/ "PySAD Example")
</page_content>

# Image Processing (MNIST Example)

@ TurboML - page_link: https://docs.turboml.com/llms/image_embeddings/
<page_content>
LLMs

Image Embeddings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/image_embeddings.ipynb)

```import turboml as tb
```

```import pandas as pd
from torchvision import datasets, transforms
import io
from PIL import Image
```

```class PILToBytes:
    def __init__(self, format="JPEG"):
        self.format = format

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError(f"Input should be a PIL Image, but got {type(img)}.")
        buffer = io.BytesIO()
        img.save(buffer, format=self.format)
        return buffer.getvalue()


transform = transforms.Compose(
    [\
        transforms.Resize((28, 28)),\
        PILToBytes(format="PNG"),\
    ]
)
```

## Data Inspection [Permalink for this section](https://docs.turboml.com/llms/image_embeddings/\#data-inspection)

Downloading the MNIST dataset to be used in ML modelling.

```mnist_dataset_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
mnist_dataset_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
```

```images_train = []
images_test = []
labels_train = []
labels_test = []

for image, label in mnist_dataset_train:
    images_train.append(image)
    labels_train.append(label)

for image, label in mnist_dataset_test:
    images_test.append(image)
    labels_test.append(label)
```

Transforming the lists into Pandas DataFrames.

```image_dict_train = {"images": images_train}
label_dict_train = {"labels": labels_train}
image_df_train = pd.DataFrame(image_dict_train)
label_df_train = pd.DataFrame(label_dict_train)

image_dict_test = {"images": images_test}
label_dict_test = {"labels": labels_test}
image_df_test = pd.DataFrame(image_dict_test)
label_df_test = pd.DataFrame(label_dict_test)
```

Adding index columns to the DataFrames to act as primary keys for the datasets.

```image_df_train.reset_index(inplace=True)
label_df_train.reset_index(inplace=True)

image_df_test.reset_index(inplace=True)
label_df_test.reset_index(inplace=True)
```

```image_df_train.head()
```

```image_df_test.head()
```

```image_df_test = image_df_test[:5].reset_index(drop=True)
label_df_test = label_df_test[:5].reset_index(drop=True)
```

Using `LocalDataset` class for compatibility with the TurboML platform.

```images_train = tb.LocalDataset.from_pd(df=image_df_train, key_field="index")
labels_train = tb.LocalDataset.from_pd(df=label_df_train, key_field="index")

images_test = tb.LocalDataset.from_pd(df=image_df_test, key_field="index")
labels_test = tb.LocalDataset.from_pd(df=label_df_test, key_field="index")
```

Extracting the features and the targets from the TurboML-compatible datasets.

```imaginal_fields = ["images"]

features_train = images_train.get_model_inputs(imaginal_fields=imaginal_fields)
targets_train = labels_train.get_model_labels(label_field="labels")

features_test = images_test.get_model_inputs(imaginal_fields=imaginal_fields)
targets_test = labels_test.get_model_labels(label_field="labels")
```

## Clip Model Initialization [Permalink for this section](https://docs.turboml.com/llms/image_embeddings/\#clip-model-initialization)

We Simply create a ClipEmbedding model with gguf\_model. The CLIP model is pulled from the Huggingface repository. As it is already quantized, we can directly pass the model file name in 'select\_model\_file' parameter.

```gguf_model = tb.llm.acquire_hf_model_as_gguf(
    "xtuner/llava-llama-3-8b-v1_1-gguf", "auto", "llava-llama-3-8b-v1_1-mmproj-f16.gguf"
)
gguf_model
```

```model = tb.ClipEmbedding(gguf_model_id=gguf_model)
```

## Model Training [Permalink for this section](https://docs.turboml.com/llms/image_embeddings/\#model-training)

Setting the model combined with the `ImageToNumeric PreProcessor` to learn on the training data.

```model = model.learn(features_train, targets_train)
```

## Model Inference [Permalink for this section](https://docs.turboml.com/llms/image_embeddings/\#model-inference)

Performing inference on the trained model using the test data.

```outputs_test = model.predict(features_test)
```

```outputs_test
```

Last updated on January 24, 2025

[LLM Embeddings](https://docs.turboml.com/llms/llm_embedding/ "LLM Embeddings") [LLM Tutorial](https://docs.turboml.com/llms/turboml_llm_tutorial/ "LLM Tutorial")
</page_content>

# Stream Dataset to Deployed Models
@ TurboML - page_link: https://docs.turboml.com/general_examples/stream_dataset_online/
<page_content>
General

Stream Dataset Online

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/stream_dataset_online.ipynb)

This notebook demonstrates how to upload data to an already registered dataset with a deployed model.

```import turboml as tb
```

```import pandas as pd
import requests
import time

from tqdm.notebook import tqdm
```

```# Add helper functions
def do_retry(
    operation,
    return_on: lambda result: True,
    retry_count=3,
    sleep_seconds=3,
):
    attempt = 1
    while attempt <= retry_count:
        print(f"## Attempt {attempt} of {retry_count}.")
        result = operation()
        if return_on(result):
            print(f"## Finished in {attempt} attempt.")
            return result
        else:
            time.sleep(sleep_seconds)
            attempt += 1
            continue
    print(f"## Exiting after {attempt} attempts.")


def simulate_realtime_stream(df: pd.DataFrame, chunk_size: int, delay: float):
    # Number of chunks to yield
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)

    for i in range(num_chunks):
        # Yield the chunk of DataFrame
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        yield chunk

        # Simulate real-time delay
        time.sleep(delay)
```

## Inspecting Data [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#inspecting-data)

```transactions_df = tb.datasets.FraudDetectionDatasetFeatures().df
labels_df = tb.datasets.FraudDetectionDatasetLabels().df
```

We will only use a subset of the dataset for initial model deployment.

```sub_transactions_df = transactions_df.iloc[0:20000]
sub_transactions_df = sub_transactions_df
sub_transactions_df
```

```sub_labels_df = labels_df.iloc[0:20000]
sub_labels_df = sub_labels_df
sub_labels_df
```

```sub_transactions_df.head()
```

```sub_labels_df.head()
```

## Data Ingestion [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#data-ingestion)

```input_dataset_id = "transactions_stream_online"

transactions = tb.OnlineDataset.from_pd(
    df=sub_transactions_df,
    id=input_dataset_id,
    key_field="transactionID",
    load_if_exists=True,
)
```

```label_dataset_id = "transaction_stream_labels"
labels = tb.OnlineDataset.from_pd(
    df=sub_labels_df,
    id=label_dataset_id,
    key_field="transactionID",
    load_if_exists=True,
)
```

## Feature Engineering [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#feature-engineering)

### Fetch data [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#fetch-data)

```tb.get_features(dataset_id=input_dataset_id)
```

### Add feature definitions [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#add-feature-definitions)

```transactions.feature_engineering.create_sql_features(
    sql_definition='"transactionAmount" + "localHour"',
    new_feature_name="my_sql_feat",
)
```

```transactions.feature_engineering.get_local_features()
```

```tb.get_timestamp_formats()
```

```transactions.feature_engineering.register_timestamp(
    column_name="timestamp", format_type="epoch_seconds"
)
```

```transactions.feature_engineering.create_aggregate_features(
    column_to_operate="transactionAmount",
    column_to_group="accountID",
    operation="SUM",
    new_feature_name="my_sum_feat",
    timestamp_column="timestamp",
    window_duration=24,
    window_unit="hours",
)
```

```transactions.feature_engineering.get_local_features()
```

### Submit feature definitions [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#submit-feature-definitions)

```transactions.feature_engineering.materialize_features(["my_sql_feat", "my_sum_feat"])
```

```materialized_features = transactions.feature_engineering.get_materialized_features()
materialized_features
```

## Supervised Learning [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#supervised-learning)

```htc_model = tb.HoeffdingTreeClassifier(n_classes=2)
```

```numerical_fields = [\
    "transactionAmount",\
    "localHour",\
    "my_sum_feat",\
    "my_sql_feat",\
]
categorical_fields = [\
    "digitalItemCount",\
    "physicalItemCount",\
    "isProxyIP",\
]
features = transactions.get_model_inputs(
    numerical_fields=numerical_fields, categorical_fields=categorical_fields
)
label = labels.get_model_labels(label_field="is_fraud")
```

### Run Supervised ML jobs [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#run-supervised-ml-jobs)

We will deploy a HoeffdingTreeClassifier Model trained on a subset of our dataset.

```deployed_model_htc = htc_model.deploy(
    "demo_classifier_htc_stream_model", input=features, labels=label
)
```

```outputs = do_retry(
    deployed_model_htc.get_outputs, return_on=(lambda result: len(result) > 0)
)
```

```outputs[-1]
```

### Supervised Model Endpoints [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#supervised-model-endpoints)

```model_endpoints = deployed_model_htc.get_endpoints()
model_endpoints
```

```model_query_datapoint = transactions_df.iloc[765].to_dict()
model_query_datapoint
```

```resp = requests.post(
    model_endpoints[0], json=model_query_datapoint, headers=tb.common.api.headers
)
resp.json()
```

### Supervised Model Evaluation [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#supervised-model-evaluation)

```deployed_model_htc.add_metric("WindowedAUC")
```

```model_auc_scores = do_retry(
    lambda: deployed_model_htc.get_evaluation("WindowedAUC"),
    return_on=(lambda result: len(result) > 0),
)
```

```import matplotlib.pyplot as plt

plt.plot([model_auc_score.metric for model_auc_score in model_auc_scores])
```

## Upload to dataset with online model [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#upload-to-dataset-with-online-model)

We will upload data to the registered dataset, which will be used for training and inference by the respective deployed model in realtime.

We use a helper function `simulate_realtime_stream` to simulate realtime streaming data from dataframe.

### Upload using SDK [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#upload-using-sdk)

Here we use the **upload\_df** method provided by the **OnlineDataset** class to upload data to a registered dataset. This method internally uploads the data using the **Arrow Flight Protocol** over gRPC.

```sub_transactions_df = transactions_df.iloc[20000:100000]
sub_transactions_df = sub_transactions_df
sub_transactions_df
```

```sub_labels_df = labels_df.iloc[20000:100000]
sub_labels_df = sub_labels_df
sub_labels_df
```

Set the chunk size and delay for the `simulate_realtime_stream` helper function

```chunk_size = 10 * 1024
delay = 0.1
```

Here we zip the two stream generators to get a batch of dataframe for input and label datasets and we upload them.

```sub_labels_df
```

```# lets normalize these dfs to replace any nan's with 0-values for the corresponding type
sub_transactions_df = tb.datasets.PandasHelpers.normalize_df(sub_transactions_df)
sub_labels_df = tb.datasets.PandasHelpers.normalize_df(sub_labels_df)

realtime_input_stream = simulate_realtime_stream(sub_transactions_df, chunk_size, delay)
realtime_label_stream = simulate_realtime_stream(sub_labels_df, chunk_size, delay)

with tqdm(
    total=len(sub_transactions_df), desc="Progress", unit="rows", unit_scale=True
) as pbar:
    for input_stream, label_stream in zip(
        realtime_input_stream, realtime_label_stream, strict=True
    ):
        start = time.perf_counter()
        transactions.add_pd(input_stream)
        labels.add_pd(label_stream)
        end = time.perf_counter()

        pbar.update(len(input_stream))
        print(
            f"# Uploaded {len(input_stream)} input, label rows for processing in {end - start:.6f} seconds."
        )
```

#### Check Updated Dataset and Model [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#check-updated-dataset-and-model)

```tb.get_features(dataset_id=input_dataset_id)
```

We can use the **sync\_features** method to sync the materialized streaming features to the **OnlineDataset** object.

```time.sleep(1)
transactions.sync_features()
```

Calling **get\_materialized\_features** method will show that newly uploaded data is properly materialized.

```materialized_features = transactions.feature_engineering.get_materialized_features()
materialized_features
```

The **get\_ouputs** method will return the latest processed ouput.

```outputs = do_retry(
    deployed_model_htc.get_outputs, return_on=(lambda result: len(result) > 0)
)
outputs[-1]
```

```model_auc_scores = deployed_model_htc.get_evaluation("WindowedAUC")
print(len(model_auc_scores))
plt.plot([model_auc_score.metric for model_auc_score in model_auc_scores])
```

```resp = requests.post(
    model_endpoints[0], json=model_query_datapoint, headers=tb.common.api.headers
)
resp.json()
```

### Upload using REST API [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#upload-using-rest-api)

Here we use the **dataset/dataset\_id/upload** REST API endpoint to upload data to a registered dataset. This endpoint will directly upload the data to the registered **dataset kafka topic**.

```sub_transactions_df = transactions_df.iloc[100000:170000]
sub_transactions_df = sub_transactions_df
sub_transactions_df
```

```sub_labels_df = labels_df.iloc[100000:170000]
sub_labels_df = sub_labels_df
sub_labels_df
```

```from turboml.common.api import api
import json
```

We use the turboml api module to initiate the HTTP call, since auth is already configured for it.

```def rest_upload_df(dataset_id: str, df: pd.DataFrame):
    row_list = json.loads(df.to_json(orient="records"))
    api.post(f"dataset/{dataset_id}/upload", json=row_list)
```

```realtime_input_stream = simulate_realtime_stream(sub_transactions_df, chunk_size, delay)
realtime_label_stream = simulate_realtime_stream(sub_labels_df, chunk_size, delay)

with tqdm(
    total=len(sub_transactions_df), desc="Progress", unit="rows", unit_scale=True
) as pbar:
    for input_stream, label_stream in zip(
        realtime_input_stream, realtime_label_stream, strict=True
    ):
        start = time.perf_counter()
        rest_upload_df(input_dataset_id, input_stream)
        rest_upload_df(label_dataset_id, label_stream)
        end = time.perf_counter()

        pbar.update(len(input_stream))
        print(
            f"# Uploaded {len(input_stream)} input, label rows for processing in {end - start:.6f} seconds."
        )
```

#### Check Updated Dataset and Model [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#check-updated-dataset-and-model-1)

```time.sleep(1)
transactions.sync_features()
```

```materialized_features = transactions.feature_engineering.get_materialized_features()
materialized_features
```

```outputs = do_retry(
    deployed_model_htc.get_outputs, return_on=(lambda result: len(result) > 0)
)
```

```outputs[-1]
```

```model_auc_scores = deployed_model_htc.get_evaluation("WindowedAUC")
print(len(model_auc_scores))
plt.plot([model_auc_score.metric for model_auc_score in model_auc_scores])
```

```deployed_model_htc.get_inference(transactions_df)
```

### Upload using gRPC API [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#upload-using-grpc-api)

This example shows how to directly upload data to the registered dataset using Arrow Flight gRPC.

```sub_transactions_df = transactions_df.iloc[170000:]
sub_transactions_df = sub_transactions_df
sub_transactions_df
```

```sub_labels_df = labels_df.iloc[170000:]
sub_labels_df = sub_labels_df
sub_labels_df
```

```import pyarrow
import struct
import itertools
from functools import partial
from pyarrow.flight import FlightDescriptor

from turboml.common.env import CONFIG as tb_config
from turboml.common import get_protobuf_class, create_protobuf_from_row_tuple
```

Here we have defined a helper function `write_batch` to write pyarrow record batch given a pyarrow flight client instance.

```def write_batch(writer, df, proto_gen_partial_func):
    row_iter = df.itertuples(index=False, name=None)
    batch_size = 1024
    while True:
        batch = list(
            map(
                proto_gen_partial_func,
                itertools.islice(row_iter, batch_size),
            )
        )

        if not batch:
            break

        batch = pyarrow.RecordBatch.from_arrays([batch], ["value"])
        writer.write(batch)
```

We initiate connection for the pyarrow flight client to the TurboML arrow server with the required configs.

```arrow_server_grpc_endpoint = tb_config.ARROW_SERVER_ADDRESS

# Note: SchemaId prefix is required for proper kafka protobuf serialization.
input_proto_gen_func = partial(
    create_protobuf_from_row_tuple,
    fields=sub_transactions_df.columns.tolist(),
    proto_cls=transactions.protobuf_cls,
    prefix=struct.pack("!xIx", transactions.registered_schema.id),
)

label_proto_gen_func = partial(
    create_protobuf_from_row_tuple,
    fields=sub_labels_df.columns.tolist(),
    proto_cls=labels.protobuf_cls,
    prefix=struct.pack("!xIx", labels.registered_schema.id),
)

client = pyarrow.flight.connect(arrow_server_grpc_endpoint)
# Note: Expected arrow schema is a column named 'value' with serialized protobuf binary message.
pa_schema = pyarrow.schema([("value", pyarrow.binary())])

input_stream_writer, _ = client.do_put(
    FlightDescriptor.for_command(f"produce:{input_dataset_id}"),
    pa_schema,
    options=pyarrow.flight.FlightCallOptions(headers=api.arrow_headers),
)

label_stream_writer, _ = client.do_put(
    FlightDescriptor.for_command(f"produce:{label_dataset_id}"),
    pa_schema,
    options=pyarrow.flight.FlightCallOptions(headers=api.arrow_headers),
)
```

Now, we use the stream generator and pass the data to the `write_batch` function along with **pyarrow client write handler** for for both input and label data writes respectively.

```realtime_input_stream = simulate_realtime_stream(sub_transactions_df, chunk_size, delay)
realtime_label_stream = simulate_realtime_stream(sub_labels_df, chunk_size, delay)

with tqdm(
    total=len(sub_transactions_df), desc="Progress", unit="rows", unit_scale=True
) as pbar:
    for input_stream, label_stream in zip(
        realtime_input_stream, realtime_label_stream, strict=True
    ):
        start = time.perf_counter()
        write_batch(input_stream_writer, input_stream, input_proto_gen_func)
        write_batch(label_stream_writer, label_stream, label_proto_gen_func)
        end = time.perf_counter()

        pbar.update(len(input_stream))
        print(
            f"# Uploaded {len(input_stream)} input, label rows for processing in {end - start:.6f} seconds."
        )
```

Close the pyarrow client write handlers.

```input_stream_writer.close()
label_stream_writer.close()
```

#### Check Updated Dataset and Model [Permalink for this section](https://docs.turboml.com/general_examples/stream_dataset_online/\#check-updated-dataset-and-model-2)

```time.sleep(1)
transactions.sync_features()
```

```materialized_features = transactions.feature_engineering.get_materialized_features()
materialized_features
```

```outputs = do_retry(
    deployed_model_htc.get_outputs, return_on=(lambda result: len(result) > 0)
)
```

```outputs[-1]
```

```model_auc_scores = deployed_model_htc.get_evaluation("WindowedAUC")
print(len(model_auc_scores))
plt.plot([model_auc_score.metric for model_auc_score in model_auc_scores])
```

Last updated on January 24, 2025

[Local Model](https://docs.turboml.com/general_examples/local_model/ "Local Model") [ONNX - Pytorch](https://docs.turboml.com/byo_models/onnx_pytorch/ "ONNX - Pytorch")
</page_content>

# LeveragingBaggingClassifier
@ TurboML - page_link: https://docs.turboml.com/ensembles/leveragingbaggingclassifier/
<page_content>
Ensembles

Leveraging Bagging Classifier

Leveraging Bagging is an improvement over the `Oza Bagging algorithm`. The bagging performance is leveraged by increasing the re-sampling. It uses a poisson distribution to simulate the re-sampling process. To increase re-sampling it uses a higher w value of the Poisson distribution (agerage number of events), 6 by default, increasing the input space diversity, by attributing a different range of weights to the data samples.

To deal with concept drift, Leveraging Bagging uses the `ADWIN` algorithm to monitor the performance of each member of the enemble If concept drift is detected, the worst member of the ensemble (based on the error estimation by ADWIN) is replaced by a new (empty) classifier.

## Parameters [Permalink for this section](https://docs.turboml.com/ensembles/leveragingbaggingclassifier/\#parameters)

- **model**( `Model`) → The classifier to bag.

- **n\_models**( `int`, Default: `10`) → The number of models in the ensemble.

- **w**( `float`, Default: `6`) → Indicates the average number of events. This is the lambda parameter of the Poisson distribution used to compute the re-sampling weight.

- **bagging\_method**( `str`, Default: `bag`) → The bagging method to use. Can be one of the following:
  - `bag` \- Leveraging Bagging using ADWIN.
  - `me` \- Assigns if sample is misclassified, otherwise.
  - `half` \- Use resampling without replacement for half of the instances.
  - `wt` \- Resample without taking out all instances.
  - `subag` \- Resampling without replacement.
- **seed**( `int` \| `None`, Default: `None`) → Random number generator seed for reproducibility.


## Example Usage [Permalink for this section](https://docs.turboml.com/ensembles/leveragingbaggingclassifier/\#example-usage)

We can create an instance and deploy LBC model like this.

```import turboml as tb
htc_model = tb.HoeffdingTreeClassifier(n_classes=2)
lbc_model = tb.LeveragingBaggingClassifier(n_classes=2, base_model = htc_model)
```

Last updated on January 24, 2025

[Contextual Bandit Model Selection](https://docs.turboml.com/ensembles/contextualbanditmodelselection/ "Contextual Bandit Model Selection") [Heterogeneous Leveraging Bagging Classifier](https://docs.turboml.com/ensembles/heteroleveragingbaggingclassifier/ "Heterogeneous Leveraging Bagging Classifier")
</page_content>

# TF-IDF embedding example using gRPC Client
@ TurboML - page_link: https://docs.turboml.com/byo_models/tfidf_example/
<page_content>
Bring Your Own Models

TF-IDF Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/tfidf_example.ipynb)

This example demonstrates using our gRPC API client to generate TF-IDF embedding.

```import turboml as tb
```

```!pip install nltk grpcio
```

### Start gRPC server for tfdif embedding from jupyter-notebook [Permalink for this section](https://docs.turboml.com/byo_models/tfidf_example/\#start-grpc-server-for-tfdif-embedding-from-jupyter-notebook)

```import pandas as pd
from utils.tfidf_grpc_server import serve
import threading


def run_server_in_background(url):
    serve(url)  # This will start the gRPC server


# Start the server in a separate thread
url = "0.0.0.0:50047"
server_thread = threading.Thread(
    target=run_server_in_background, args=(url,), daemon=True
)
server_thread.start()

print("gRPC server is running in the background...")
```

### Load text dataset [Permalink for this section](https://docs.turboml.com/byo_models/tfidf_example/\#load-text-dataset)

```import re
import urllib.request

with urllib.request.urlopen(
    "https://raw.githubusercontent.com/TurboML-Inc/colab-notebooks/refs/heads/main/data/tfidf_test_data.txt"
) as file:
    text = file.read().decode()

sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
labels = [0] * len(sentences)
```

```text_dict_test = {"text": sentences}
label_dict_test = {"labels": labels}
text_df_test = pd.DataFrame(text_dict_test)
label_df_test = pd.DataFrame(label_dict_test)
text_df_test.reset_index(inplace=True)
label_df_test.reset_index(inplace=True)
```

```text_df_test = text_df_test.reset_index(drop=True)
label_df_test = label_df_test.reset_index(drop=True)
```

```text_train = tb.LocalDataset.from_pd(df=text_df_test, key_field="index")
labels_train = tb.LocalDataset.from_pd(df=label_df_test, key_field="index")

text_test = tb.LocalDataset.from_pd(df=text_df_test, key_field="index")
labels_test = tb.LocalDataset.from_pd(df=label_df_test, key_field="index")
```

```textual_fields = ["text"]
features_train = text_train.get_model_inputs(textual_fields=textual_fields)
targets_train = labels_train.get_model_labels(label_field="labels")

features_test = text_test.get_model_inputs(textual_fields=textual_fields)
targets_test = labels_test.get_model_labels(label_field="labels")
```

### Using TurboML Client to request gRPC server [Permalink for this section](https://docs.turboml.com/byo_models/tfidf_example/\#using-turboml-client-to-request-grpc-server)

```grpc_model = tb.GRPCClient(
    server_url="0.0.0.0:50047",
    connection_timeout=10000,
    max_request_time=10000,
    max_retries=1,
)
```

```model_trained = grpc_model.learn(features_train, targets_train)
```

```outputs_test = model_trained.predict(features_test)
```

```outputs_test
```

Last updated on January 24, 2025

[ONNX - Tensorflow](https://docs.turboml.com/byo_models/onnx_tensorflow/ "ONNX - Tensorflow") [ResNet Example](https://docs.turboml.com/byo_models/resnet_example/ "ResNet Example")
</page_content>

# EmbeddingModel
@ TurboML - page_link: https://docs.turboml.com/pipeline_components/embeddingmodel/
<page_content>
Pipeline Components

Embedding Model

An embedding is a numerical representation of a piece of information, for example, text, documents, images, audio, etc. The representation captures the semantic meaning of what is being embedded. This is a meta-model which takes in a `embedding_model` and a `base_model`. The embeddings are computed by the embedding\_model, and passed as numerical input to the base\_model for training/prediction.

## Parameters [Permalink for this section](https://docs.turboml.com/pipeline_components/embeddingmodel/\#parameters)

- **embedding\_model**( `embedding_model`) → The embedding model to used.
- **base\_model**( `base_model`) → The base classifier or regressor model.

## Example Usage [Permalink for this section](https://docs.turboml.com/pipeline_components/embeddingmodel/\#example-usage)

We can use and deploy the EmbeddingModel as such.

```import turboml as tb
embedding = tb.RandomProjectionEmbedding(n_embeddings = 4)
htc_model = tb.HoeffdingTreeClassifier(n_classes=2)
embedding_model = tb.EmbeddingModel(embedding_model = embedding, base_model = htc_model)
```

Last updated on January 24, 2025

[PreProcessors](https://docs.turboml.com/pipeline_components/preprocessors/ "PreProcessors") [Random Projection Embedding](https://docs.turboml.com/pipeline_components/randomprojectionembedding/ "Random Projection Embedding")
</page_content>

# BanditModelSelection
@ TurboML - page_link: https://docs.turboml.com/ensembles/banditmodelselection/
<page_content>
Ensembles

Bandit Model Selection

Bandit-based model selection.

Each model is associated with an arm, in a multi-arm bandit scenario. The bandit algorithm is used to decide which models to update on new data. The reward of the bandit is the performance of the model on that sample. Fo prediction, we always use the current best model based on the bandit.

## Parameters [Permalink for this section](https://docs.turboml.com/ensembles/banditmodelselection/\#parameters)

- **bandit**(Default: `EpsGreedy`) → The underlying bandit algorithm. Options are: EpsGreedy, UCB, and GaussianTS.

- **metric\_name**(Default: `WindowedMAE`) → The metric to use to evaluate models. Options are: WindowedAUC, WindowedAccuracy, WindowedMAE, WindowedMSE, and WindowedRMSE.

- **base\_models**( `list[Model]`) → The list of models over which to perform model selection.


## Example Usage [Permalink for this section](https://docs.turboml.com/ensembles/banditmodelselection/\#example-usage)

We can create an instance and deploy BanditModel like this.

```import turboml as tb
htc_model = tb.HoeffdingTreeRegressor()
amf_model = tb.AMFRegressor()
ffm_model = tb.FFMRegressor()
bandit_model = tb.BanditModelSelection(base_models = [htc_model, amf_model, ffm_model])
```

Last updated on January 24, 2025

[AdaBoost Classifier](https://docs.turboml.com/ensembles/adaboostclassifer/ "AdaBoost Classifier") [Contextual Bandit Model Selection](https://docs.turboml.com/ensembles/contextualbanditmodelselection/ "Contextual Bandit Model Selection")
</page_content>

# HeteroLeveragingBaggingClassifier
@ TurboML - page_link: https://docs.turboml.com/ensembles/heteroleveragingbaggingclassifier/
<page_content>
Ensembles

Heterogeneous Leveraging Bagging Classifier

Similar to LeveragingBaggingClassifier, but instead of multiple copies of the same model, it can work with different base models.

## Parameters [Permalink for this section](https://docs.turboml.com/ensembles/heteroleveragingbaggingclassifier/\#parameters)

- **base\_models**( `list[Model]`) → The list of classifier models.

- **n\_classes**( `int`) → The number of classes for the classifier.

- **w**( `float`, Default: `6`) → Indicates the average number of events. This is the lambda parameter of the Poisson distribution used to compute the re-sampling weight.

- **bagging\_method**( `str`, Default: `bag`) → The bagging method to use. Can be one of the following:
  - `bag` \- Leveraging Bagging using ADWIN.
  - `me` \- Assigns if sample is misclassified, otherwise.
  - `half` \- Use resampling without replacement for half of the instances.
  - `wt` \- Resample without taking out all instances.
  - `subag` \- Resampling without replacement.
- **seed**( `int` \| `None`, Default: `None`) → Random number generator seed for reproducibility.


## Example Usage [Permalink for this section](https://docs.turboml.com/ensembles/heteroleveragingbaggingclassifier/\#example-usage)

We can create an instance and deploy LBC model like this.

```import turboml as tb
model = tb.HeteroLeveragingBaggingClassifier(n_classes=2, base_models = [tb.HoeffdingTreeClassifier(n_classes=2), tb.AMFClassifier(n_classes=2)])
```

Last updated on January 24, 2025

[Leveraging Bagging Classifier](https://docs.turboml.com/ensembles/leveragingbaggingclassifier/ "Leveraging Bagging Classifier") [Heterogeneous AdaBoost Classifier](https://docs.turboml.com/ensembles/heteroadaboostclassifer/ "Heterogeneous AdaBoost Classifier")
</page_content>

# FFM Regressor
@ TurboML - page_link: https://docs.turboml.com/regression/ffmregressor/
<page_content>
Regression

FFM Regressor


**Field-aware Factorization Machine** [1](https://docs.turboml.com/regression/ffmregressor/#user-content-fn-1) for regression.

The model equation is defined by:
Where is the latent vector corresponding to feature for field, and is the latent vector corresponding to feature for field.
`$$ \sum_{f1=1}^{F} \sum_{f2=f1+1}^{F} \mathbf{w_{i1}} \cdot \mathbf{w_{i2}}, \text{where } i1 = \Phi(v_{f1}, f1, f2), \quad i2 = \Phi(v_{f2}, f2, f1) $$`
Our implementation automatically applies MinMax scaling to the inputs, use normal distribution for latent initialization and squared loss for optimization.

## Parameters [Permalink for this section](https://docs.turboml.com/regression/ffmregressor/\#parameters)

- **n\_factors**( `int`, Default: `10`) → Dimensionality of the factorization or number of latent factors.

- **l1\_weight**( `int`, Default: `0.0`) → Amount of L1 regularization used to push weights towards 0.

- **l2\_weight**( `int`, Default: `0.0`) → Amount of L2 regularization used to push weights towards 0.

- **l1\_latent**( `int`, Default: `0.0`) → Amount of L1 regularization used to push latent weights towards 0.

- **l2\_latent**( `int`, Default: `0.0`) → Amount of L2 regularization used to push latent weights towards 0.

- **intercept**( `int`, Default: `0.0`) → Initial intercept value.

- **intercept\_lr**( `float`, Default: `0.01`) → Learning rate scheduler used for updating the intercept. No intercept will be used if this is set to 0.

- **clip\_gradient**(Default: `1000000000000.0`) → Clips the absolute value of each gradient value.

- **seed**( `int` \| `None`, Default: `None`) → Randomization seed used for reproducibility.


## Example Usage [Permalink for this section](https://docs.turboml.com/regression/ffmregressor/\#example-usage)

We can create an instance of the FFM model like this.

```import turboml as tb
ffm_model = tb.FFMRegressor()
```

## Footnotes [Permalink for this section](https://docs.turboml.com/regression/ffmregressor/\#footnote-label)

1. Juan, Y., Zhuang, Y., Chin, W.S. and Lin, C.J., 2016, September. Field-aware factorization machines for CTR prediction. In Proceedings of the 10th ACM Conference on Recommender Systems (pp. 43-50). [↩](https://docs.turboml.com/regression/ffmregressor/#user-content-fnref-1)


Last updated on January 24, 2025

[AMF Regressor](https://docs.turboml.com/regression/amfregressor/ "AMF Regressor") [Hoeffding Tree Regressor](https://docs.turboml.com/regression/hoeffdingtreeregressor/ "Hoeffding Tree Regressor")
</page_content>

# Hyperparameter Tuning
@ TurboML - page_link: https://docs.turboml.com/pre_deployment_ml/hyperparameter_tuning/
<page_content>
Pre-Deployment ML

Hyperparameter Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/hyperparameter_tuning.ipynb)

Hyperparameter Tuning uses grid search to scan through a given hyperparameter space for a model and find out the best combination of hyperparameters with respect to a given performance metric.

```import turboml as tb
```

```from sklearn import metrics
```

## Dataset [Permalink for this section](https://docs.turboml.com/pre_deployment_ml/hyperparameter_tuning/\#dataset)

We use our standard `FraudDetection` dataset for this example, exposed through the `LocalDataset` interface that can be used for tuning, and also configure the dataset to indicate the column with the primary key.

For this example, we use the first 100k rows.

```transactions = tb.datasets.FraudDetectionDatasetFeatures()
labels = tb.datasets.FraudDetectionDatasetLabels()

transactions_100k = transactions[:100000]
labels_100k = labels[:100000]
```

```numerical_fields = ["transactionAmount", "localHour"]
categorical_fields = ["digitalItemCount", "physicalItemCount", "isProxyIP"]
inputs = transactions_100k.get_model_inputs(
    numerical_fields=numerical_fields, categorical_fields=categorical_fields
)
label = labels_100k.get_model_labels(label_field="is_fraud")
```

## Training/Tuning [Permalink for this section](https://docs.turboml.com/pre_deployment_ml/hyperparameter_tuning/\#trainingtuning)

We will be using the `AdaBoost Classifier` with `Hoeffding Tree Classifier` being the base model as an example.

```model_to_tune = tb.AdaBoostClassifier(
    n_classes=2, base_model=tb.HoeffdingTreeClassifier(n_classes=2)
)
```

Since a particular model object can include other base models and PreProcessors as well, the `hyperparameter_tuning` function accepts a list of hyperparameter spaces for all such models as part of the `model` parameter, and tests all possible combinations across the different spaces.

In this example, the first dictionary in the list corresponds to the hyperparameters of `AdaBoostClassifier` while the second dictionary is the hyperparameter space for the `HoeffdingTreeClassifier`.

It is not necessary to include all possible hyperparameters in the space; default values are taken for those not specified

```model_score_list = tb.hyperparameter_tuning(
    metric_to_optimize="accuracy",
    model=model_to_tune,
    hyperparameter_space=[\
        {"n_models": [2, 3]},\
        {\
            "delta": [1e-7, 1e-5, 1e-3],\
            "tau": [0.05, 0.01, 0.1],\
            "grace_period": [200, 100, 500],\
            "n_classes": [2],\
            "leaf_pred_method": ["mc"],\
            "split_method": ["gini", "info_gain", "hellinger"],\
        },\
    ],
    input=inputs,
    labels=label,
)
best_model, best_score = model_score_list[0]
best_model
```

```features = transactions.get_model_inputs(
    numerical_fields=numerical_fields, categorical_fields=categorical_fields
)

outputs = best_model.predict(features)
```

```labels_df = labels.df
print(
    "Accuracy: ",
    metrics.accuracy_score(labels_df["is_fraud"], outputs["predicted_class"]),
)
print("F1: ", metrics.f1_score(labels_df["is_fraud"], outputs["predicted_class"]))
```

Last updated on January 24, 2025

[Algorithm Tuning](https://docs.turboml.com/pre_deployment_ml/algorithm_tuning/ "Algorithm Tuning") [Performance Improvements](https://docs.turboml.com/pre_deployment_ml/performance_improvements/ "Performance Improvements")
</page_content>

# Algorithm Tuning
@ TurboML - page_link: https://docs.turboml.com/pre_deployment_ml/algorithm_tuning/
<page_content>
Pre-Deployment ML

Algorithm Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/algorithm_tuning.ipynb)

Algorithm Tuning allows us to test different models on a given dataset, and helps to figure out which particular model gives the highest value of a user-defined performance metric on that particular dataset.

```import turboml as tb
```

```import pandas as pd
from sklearn import metrics
```

## Dataset [Permalink for this section](https://docs.turboml.com/pre_deployment_ml/algorithm_tuning/\#dataset)

We use our standard `FraudDetection` dataset for this example, exposed through the `LocalDataset` interface that can be used for tuning, and also configure the dataset to indicate the column with the primary key.

For this example, we use the first 100k rows.

```transactions = tb.datasets.FraudDetectionDatasetFeatures()
labels = tb.datasets.FraudDetectionDatasetLabels()
```

```transactions_100k = transactions[:100000]
labels_100k = labels[:100000]

numerical_fields = [\
    "transactionAmount",\
]
categorical_fields = ["digitalItemCount", "physicalItemCount", "isProxyIP"]
inputs = transactions_100k.get_model_inputs(
    numerical_fields=numerical_fields, categorical_fields=categorical_fields
)
label = labels_100k.get_model_labels(label_field="is_fraud")
```

## Training/Tuning [Permalink for this section](https://docs.turboml.com/pre_deployment_ml/algorithm_tuning/\#trainingtuning)

We will be comparing the `Neural Network` and `Hoeffding Tree Classifier`, and the metric we will be optimizing is `accuracy`.

Configuring the NN according to the dataset.

```new_layer = tb.NNLayer(output_size=2)

nn = tb.NeuralNetwork()
nn.layers.append(new_layer)
```

The `algorithm_tuning` function takes in the models being tested as a list along with the metric to test against, and returns an object for the model which had the highest score for the given metric.

```model_score_list = tb.algorithm_tuning(
    models_to_test=[\
        tb.HoeffdingTreeClassifier(n_classes=2),\
        nn,\
    ],
    metric_to_optimize="accuracy",
    input=inputs,
    labels=label,
)
best_model, best_score = model_score_list[0]
best_model
```

## Testing

After finding out the best performing model, we can use it normally for inference on the entire dataset and testing on more performance metrics.

```transactions_test = transactions[100000:]
labels_test = labels[100000:]
```

```features = transactions_test.get_model_inputs(
    numerical_fields=numerical_fields, categorical_fields=categorical_fields
)

outputs = best_model.predict(features)
```

```print(
    "Accuracy: ",
    metrics.accuracy_score(labels_test.df["is_fraud"], outputs["predicted_class"]),
)
print("F1: ", metrics.f1_score(labels_test.df["is_fraud"], outputs["predicted_class"]))
```

Last updated on January 24, 2025

[Ibis Feature Engineering](https://docs.turboml.com/feature_engineering/advanced/ibis_feature_engineering/ "Ibis Feature Engineering") [Hyperparameter Tuning](https://docs.turboml.com/pre_deployment_ml/hyperparameter_tuning/ "Hyperparameter Tuning")
</page_content>

# Drift Detection
@ TurboML - page_link: https://docs.turboml.com/post_deployment_ml/drift/
<page_content>
Post-Deployment ML

Drift

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/drift.ipynb)

Drift detection is a crucial part of ML observability. As is the case with other components, drift detection in TurboML is a continuous streaming process. In this notebook, we'll see how to compute data drift (univariate and multivariate) and model drift.

For univariate drift detection, by default we're using Adaptive Windowing method, and for multivariate drift detection, by default we're using PCA based reconstruction method.

```import turboml as tb
```

```transactions = tb.datasets.FraudDetectionDatasetFeatures().to_online(
    "transactions", load_if_exists=True
)
labels = tb.datasets.FraudDetectionDatasetLabels().to_online(
    "transaction_labels", load_if_exists=True
)
```

```model = tb.RCF(number_of_trees=50)
```

```numerical_fields = [\
    "transactionAmount",\
    "localHour",\
    "isProxyIP",\
    "physicalItemCount",\
    "digitalItemCount",\
]
features = transactions.get_model_inputs(numerical_fields=numerical_fields)
label = labels.get_model_labels(label_field="is_fraud")
```

```deployed_model = model.deploy(name="drift_demo", input=features, labels=label)
```

We can register univariate drift by using `numerical_field` and optionally a `label`. By default, label is same as `numerical_field`.

```transactions.register_univariate_drift(numerical_field="transactionAmount")
```

```transactions.register_univariate_drift(
    label="demo_uv_drift", numerical_field="physicalItemCount"
)
```

For multivariate drift, providing `label` is required.

```transactions.register_multivariate_drift(
    label="demo_mv_drift", numerical_fields=numerical_fields
)
```

```deployed_model.add_drift()
```

```import matplotlib.pyplot as plt


def plot_drift(drifts):
    plt.plot([drift["record"].score for drift in drifts])
```

We can use either `label` or `numerical_field(s)` to fetch drift results.

```plot_drift(transactions.get_univariate_drift(numerical_field="transactionAmount"))
```

```plot_drift(transactions.get_univariate_drift(label="demo_uv_drift"))
```

```plot_drift(transactions.get_multivariate_drift(label="demo_mv_drift"))
```

```plot_drift(deployed_model.get_drifts())
```

Last updated on January 24, 2025

[Performance Improvements](https://docs.turboml.com/pre_deployment_ml/performance_improvements/ "Performance Improvements") [Model Explanations](https://docs.turboml.com/post_deployment_ml/model_explanations/ "Model Explanations")
</page_content>


# MSTREAM

@ TurboML - page_link: https://docs.turboml.com/anomaly_detection/mstream/
<page_content>
Anomaly Detection

MStream

MSTREAM [1](https://docs.turboml.com/anomaly_detection/mstream/#user-content-fn-1) can detect unusual group anomalies as they occur,
in a dynamic manner. MSTREAM has the following properties:

- (a) it detects anomalies in multi-aspect data including both categorical and
numeric attributes;
- (b) it is online, thus processing each record in
constant time and constant memory;
- (c) it can capture the correlation
between multiple aspects of the data.

![mstream](https://docs.turboml.com/_next/static/media/mstream.b86655c2.png)

## Parameters [Permalink for this section](https://docs.turboml.com/anomaly_detection/mstream/\#parameters)

- **num\_rows**( `int`, Default: `2`) → Number of Hash Functions.

- **num\_buckets**( `int`, Default: `factor`) → Number of Buckets for hashing.

- **factor**( `float`, Default: `0.8`) → Temporal Decay Factor.


## Example Usage [Permalink for this section](https://docs.turboml.com/anomaly_detection/mstream/\#example-usage)

We can create an instance and deploy LBC model like this.

```import turboml as tb
model = tb.MStream()
```

## Footnotes [Permalink for this section](https://docs.turboml.com/anomaly_detection/mstream/\#footnote-label)

1. Bhatia, S., Jain, A., Li, P., Kumar, R., & Hooi, B. (2021, April). Mstream: Fast anomaly detection in multi-aspect streams. In Proceedings of the Web Conference 2021 (pp. 3371-3382). [↩](https://docs.turboml.com/anomaly_detection/mstream/#user-content-fnref-1)


Last updated on January 24, 2025

[Half Space Trees](https://docs.turboml.com/anomaly_detection/hst/ "Half Space Trees") [Random Cut Forest](https://docs.turboml.com/anomaly_detection/rcf/ "Random Cut Forest")
</page_content>

# RandomProjectionEmbedding
@ TurboML - page_link: https://docs.turboml.com/pipeline_components/randomprojectionembedding/
<page_content>
Pipeline Components

Random Projection Embedding

This model supports two methods of generating embeddings.

## Sparse Random Projection [Permalink for this section](https://docs.turboml.com/pipeline_components/randomprojectionembedding/\#sparse-random-projection)

Reduces the dimensionality of inputs by projecting them onto a sparse random projection matrix using a density (ratio of non-zero components in the matrix) of 0.1.

## Gaussian Random Projection [Permalink for this section](https://docs.turboml.com/pipeline_components/randomprojectionembedding/\#gaussian-random-projection)

Reduces the dimensionality of inputs through Gaussian random projection where the components of the random projections matrix are drawn from `N(0, 1/n_embeddings)`.

## Parameters [Permalink for this section](https://docs.turboml.com/pipeline_components/randomprojectionembedding/\#parameters)

- **n\_embeddings**(Default: `2`) → Number of components to project the data onto.

- **type\_embedding**(Default: `Gaussian`) → Method to use for random projection. Options are "Gaussian" and "Sparse".


## Example Usage [Permalink for this section](https://docs.turboml.com/pipeline_components/randomprojectionembedding/\#example-usage)

We can create an instance of the RandomProjectionEmbedding model like this.

```import turboml as tb
embedding = tb.RandomProjectionEmbedding(n_embeddings = 4)
```

Last updated on January 24, 2025

[Embedding Model](https://docs.turboml.com/pipeline_components/embeddingmodel/ "Embedding Model") [Random Sampler](https://docs.turboml.com/pipeline_components/randomsampler/ "Random Sampler")
</page_content>

# Gaussian Naive Bayes
@ TurboML - page_link: https://docs.turboml.com/classification/gaussiannb/
<page_content>
Classification

Gaussian Naive Bayes

**Gaussian Naive Bayes**, A Gaussian `$$G_{cf}$$` distribution is maintained for each class `c` and each feature `f` . Each Gaussian is updated using the amount associated with each feature. The joint log-likelihood is then obtained by summing the log probabilities of each feature associated with each class.

## Parameters [Permalink for this section](https://docs.turboml.com/classification/gaussiannb/\#parameters)

- **n\_classes**( `int`) → The number of classes for classification.

## Example Usage [Permalink for this section](https://docs.turboml.com/classification/gaussiannb/\#example-usage)

We can create an instance of GaussianNB model in this format.

```import turboml as tb
gauNB = tb.GaussianNB(n_classes=2)
```

[FFM Classifier](https://docs.turboml.com/classification/ffmclassifier/ "FFM Classifier") [Multinomial Naive Bayes](https://docs.turboml.com/classification/multinomialnb/ "Multinomial Naive Bayes")
</page_content>

# Feature Engineering - Python UDAF
@ TurboML - page_link: https://docs.turboml.com/feature_engineering/udaf/
<page_content>
Feature Engineering

UDAF

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/UDAF.ipynb)

```import turboml as tb
```

```transactions = tb.datasets.FraudDetectionDatasetFeatures()[:100].to_online(
    id="udaf_transactions", load_if_exists=True
)
```

### User Defined Aggregation function [Permalink for this section](https://docs.turboml.com/feature_engineering/udaf/\#user-defined-aggregation-function)

To create a UDAF, you need to implement the following essential functions in a separate python file containing the function. These functions manage the lifecycle of the aggregation process, from initialization to final result computation:

#### State Initialization (create\_state): [Permalink for this section](https://docs.turboml.com/feature_engineering/udaf/\#state-initialization-create_state)

Purpose: This function sets up the initial state for the UDAF.
Requirement: The state should represent the data structure that will store intermediate results (e.g., sum, count, or any other aggregated values).

#### Accumulation (accumulate): [Permalink for this section](https://docs.turboml.com/feature_engineering/udaf/\#accumulation-accumulate)

Purpose: This function updates the state with new values as they are processed.
Requirement: It should handle null or missing values gracefully and update the intermediate state based on the value and any additional parameters.

#### Retraction (retract): [Permalink for this section](https://docs.turboml.com/feature_engineering/udaf/\#retraction-retract)

Purpose: This function "retracts" or removes a previously accumulated value from the state.
Requirement: It should reverse the effect of the accumulate function for cases where data needs to be removed (e.g., when undoing a previous operation).

#### Merging States (merge\_states): [Permalink for this section](https://docs.turboml.com/feature_engineering/udaf/\#merging-states-merge_states)

Purpose: This function merges two states together.
Requirement: Combine the intermediate results from two states into one. This is essential for distributed aggregations.

#### Final Result Computation (finish): [Permalink for this section](https://docs.turboml.com/feature_engineering/udaf/\#final-result-computation-finish)

Purpose: This function computes the final result once all values have been accumulated.
Requirement: It should return the final output of the aggregation based on the state. Handle edge cases such as empty datasets (e.g., return None if no valid values were processed).

```function_file_contents = """
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
```

```transactions.feature_engineering.register_timestamp(
    column_name="timestamp", format_type="epoch_seconds"
)
```

```transactions.feature_engineering.create_udaf_features(
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

```transactions.feature_engineering.materialize_features(["weighted_avg"])
```

```transactions.feature_engineering.get_materialized_features()
```

Last updated on January 24, 2025

[UDF](https://docs.turboml.com/feature_engineering/udf/ "UDF") [Ibis Quickstart](https://docs.turboml.com/feature_engineering/advanced/ibis_quickstart/ "Ibis Quickstart")
</page_content>

# TurboML LLM Tutorial
@ TurboML - page_link: https://docs.turboml.com/llms/turboml_llm_tutorial/
<page_content>
LLMs

LLM Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/turboml_llm_tutorial.ipynb)

TurboML can spin up LLM servers with an OpenAI-compatible API. We currently support models
in the GGUF format, but also support non-GGUF models that can be converted to GGUF. In the latter
case you get to decide the quantization type you want to use.

```import turboml as tb
```

```LlamaServerRequest = tb.llm.LlamaServerRequest
HuggingFaceSpec = LlamaServerRequest.HuggingFaceSpec
ServerParams = LlamaServerRequest.ServerParams
```

## Choose a model [Permalink for this section](https://docs.turboml.com/llms/turboml_llm_tutorial/\#choose-a-model)

Let's use a Llama 3.2 quant already in the GGUF format.

```hf_spec = HuggingFaceSpec(
    hf_repo_id="hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
    select_gguf_file="llama-3.2-1b-instruct-q4_k_m.gguf",
)
```

## Spawn a server [Permalink for this section](https://docs.turboml.com/llms/turboml_llm_tutorial/\#spawn-a-server)

On spawning a server, you get a `server_id` to reference it later as well as `server_relative_url` you can
use to reach it. This method is synchronous, so it can take a while to yield as we retrieve (and convert) your model.

```response = tb.llm.spawn_llm_server(
    LlamaServerRequest(
        source_type=LlamaServerRequest.SourceType.HUGGINGFACE,
        hf_spec=hf_spec,
        server_params=ServerParams(
            threads=-1,
            seed=-1,
            context_size=0,
            flash_attention=False,
        ),
    )
)
response
```

```server_id = response.server_id
```

### Interacting with the LLM [Permalink for this section](https://docs.turboml.com/llms/turboml_llm_tutorial/\#interacting-with-the-llm)

Our LLM is exposed with an OpenAI-compatible API, so we can use the OpenAI SDK, or any
other tool compatible tool to use it.

```%pip install openai
```

```from openai import OpenAI

base_url = tb.common.env.CONFIG.TURBOML_BACKEND_SERVER_ADDRESS
server_url = f"{base_url}/{response.server_relative_url}"

client = OpenAI(base_url=server_url, api_key="-")


response = client.chat.completions.create(
    messages=[\
        {\
            "role": "user",\
            "content": "Hello there how are you doing today?",\
        }\
    ],
    model="-",
)

print(response)
```

```embeddings = (
    client.embeddings.create(input=["Hello there how are you doing today?"], model="-")
    .data[0]
    .embedding
)
len(embeddings), embeddings[:5]
```

## Stop the server [Permalink for this section](https://docs.turboml.com/llms/turboml_llm_tutorial/\#stop-the-server)

```tb.llm.stop_llm_server(server_id)
```

Last updated on January 24, 2025

[Image Embeddings](https://docs.turboml.com/llms/image_embeddings/ "Image Embeddings") [String Encoding](https://docs.turboml.com/non_numeric_inputs/string_encoding/ "String Encoding")
</page_content>

# AMF Classifier
@ TurboML - page_link: https://docs.turboml.com/classification/amfclassifier/
<page_content>
Classification

AMF Classifier

**Aggregated Mondrian Forest** classifier for online learning.
This implementation is truly online, in the sense that a single pass is performed, and that predictions can be produced anytime.

Each node in a _tree_ predicts according to the distribution of the labels it contains. This distribution is regularized using a **Jeffreys** prior with parameter `dirichlet`. For each class with count labels in the node and n\_samples samples in it, the prediction of a node is given by

The prediction for a sample is computed as the aggregated predictions of all the subtrees along the path leading to the leaf node containing the sample. The aggregation weights are exponential weights with learning rate step and log-loss when use\_aggregation is True.

This computation is performed exactly thanks to a context tree weighting algorithm. More details can be found in the paper cited in the reference[1](https://docs.turboml.com/classification/amfclassifier/#user-content-fn-1) below.

The final predictions are the average class probabilities predicted by each of the n\_estimators trees in the forest.

## Parameters [Permalink for this section](https://docs.turboml.com/classification/amfclassifier/\#parameters)

- **n\_classes**( `int`) → The number of classes for classification.

- **n\_estimators**( `int`, Default: `10`) → The number of trees in the forest.

- **step** ( `float`, Default: `1.0`) → Step-size for the aggregation weights. Default is 1 for classification with the log-loss, which is usually the best choice.

- **use\_aggregation**( `bool`, Default: `True`) → Controls if aggregation is used in the trees. It is highly recommended to leave it as True.

- **dirichlet** ( `float`, Default: `0.5`) → Regularization level of the class frequencies used for predictions in each node. A rule of thumb is to set this to 1 / n\_classes, where n\_classes is the expected number of classes which might appear. Default is dirichlet = 0.5, which works well for binary classification problems.

- **split\_pure**( `bool`, Default: `False`) → Controls if nodes that contains only sample of the same class should be split ("pure" nodes). Default is False, namely pure nodes are not split, but True can be sometimes better.

- **seed**( `int` \| `None`, Default: `None`) → Random seed for reproducibility.


## Example Usage [Permalink for this section](https://docs.turboml.com/classification/amfclassifier/\#example-usage)

We can simply use the below syntax to invoke the list of algorithms preconfigured in TurboML, here `have_labels=True` means supervised models.

```import turboml as tb
amf_model = tb.AMFClassifier(n_classes=2)
```

ℹ

Only log\_loss used for the computation of the aggregation weights is supported for now, namely the log-loss for multi-class classification.

## Footnotes [Permalink for this section](https://docs.turboml.com/classification/amfclassifier/\#footnote-label)

1. Mourtada, J., Gaïffas, S., & Scornet, E. (2021). AMF: Aggregated Mondrian forests for online learning. Journal of the Royal Statistical Society Series B: Statistical Methodology, 83(3), 505-533. [↩](https://docs.turboml.com/classification/amfclassifier/#user-content-fnref-1)


Last updated on January 24, 2025

[SGT Regressor](https://docs.turboml.com/regression/sgtregressor/ "SGT Regressor") [FFM Classifier](https://docs.turboml.com/classification/ffmclassifier/ "FFM Classifier")
</page_content>

# Image Processing (MNIST Example)
@ TurboML - page_link: https://docs.turboml.com/non_numeric_inputs/image_input/
<page_content>
Non-Numeric Inputs

Image Input

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/image_input.ipynb)

```import turboml as tb
```

```import pandas as pd
from torchvision import datasets, transforms
import io
from PIL import Image
```

```class PILToBytes:
    def __init__(self, format="JPEG"):
        self.format = format

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError(f"Input should be a PIL Image, but got {type(img)}.")
        buffer = io.BytesIO()
        img.save(buffer, format=self.format)
        return buffer.getvalue()


transform = transforms.Compose(
    [\
        transforms.Resize((28, 28)),\
        PILToBytes(format="PNG"),\
    ]
)
```

## Data Inspection [Permalink for this section](https://docs.turboml.com/non_numeric_inputs/image_input/\#data-inspection)

Downloading the MNIST dataset to be used in ML modelling.

```mnist_dataset_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
mnist_dataset_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
```

```images_train = []
images_test = []
labels_train = []
labels_test = []

for image, label in mnist_dataset_train:
    images_train.append(image)
    labels_train.append(label)

for image, label in mnist_dataset_test:
    images_test.append(image)
    labels_test.append(label)
```

Transforming the lists into Pandas DataFrames.

```image_dict_train = {"images": images_train}
label_dict_train = {"labels": labels_train}
image_df_train = pd.DataFrame(image_dict_train)
label_df_train = pd.DataFrame(label_dict_train)

image_dict_test = {"images": images_test}
label_dict_test = {"labels": labels_test}
image_df_test = pd.DataFrame(image_dict_test)
label_df_test = pd.DataFrame(label_dict_test)
```

Adding index columns to the DataFrames to act as primary keys for the datasets.

```image_df_train.reset_index(inplace=True)
label_df_train.reset_index(inplace=True)

image_df_test.reset_index(inplace=True)
label_df_test.reset_index(inplace=True)
```

```image_df_train.head()
```

```label_df_train.head()
```

Using `LocalDataset` class for compatibility with the TurboML platform.

```images_train = tb.LocalDataset.from_pd(df=image_df_train, key_field="index")
labels_train = tb.LocalDataset.from_pd(df=label_df_train, key_field="index")

images_test = tb.LocalDataset.from_pd(df=image_df_test, key_field="index")
labels_test = tb.LocalDataset.from_pd(df=label_df_test, key_field="index")
```

Extracting the features and the targets from the TurboML-compatible datasets.

```imaginal_fields = ["images"]

features_train = images_train.get_model_inputs(imaginal_fields=imaginal_fields)
targets_train = labels_train.get_model_labels(label_field="labels")

features_test = images_test.get_model_inputs(imaginal_fields=imaginal_fields)
targets_test = labels_test.get_model_labels(label_field="labels")
```

## Model Initialization [Permalink for this section](https://docs.turboml.com/non_numeric_inputs/image_input/\#model-initialization)

Defining a Neural Network (NN) to be used on the MNIST data.

The `output_size` of the final layer in the NN is `10` in the case of MNIST.

Since this is a classification task, `Cross Entropy` loss is used with the `Adam` optimizer.

```final_layer = tb.NNLayer(output_size=10, activation="none")

model = tb.NeuralNetwork(
    loss_function="cross_entropy", optimizer="adam", learning_rate=0.01
)
model.layers[-1] = final_layer
```

## ImageToNumeric PreProcessor [Permalink for this section](https://docs.turboml.com/non_numeric_inputs/image_input/\#imagetonumeric-preprocessor)

Since we are dealing with images as input to the model, we select the `ImageToNumeric PreProcessor` to accordingly convert the binary images into numerical data useful to the NN.

```model = tb.ImageToNumericPreProcessor(base_model=model, image_sizes=[28, 28, 1])
```

## Model Training [Permalink for this section](https://docs.turboml.com/non_numeric_inputs/image_input/\#model-training)

Setting the model combined with the `ImageToNumeric PreProcessor` to learn on the training data.

```model = model.learn(features_train, targets_train)
```

## Model Inference [Permalink for this section](https://docs.turboml.com/non_numeric_inputs/image_input/\#model-inference)

Performing inference on the trained model using the test data.

```outputs_test = model.predict(features_test)
```

```outputs_test
```

## Model Testing [Permalink for this section](https://docs.turboml.com/non_numeric_inputs/image_input/\#model-testing)

Testing the trained model's performance on the test data.

```from sklearn import metrics
```

```labels_test_list = labels_test.input_df["labels"].to_list()
```

```print(
    "Accuracy: ",
    metrics.accuracy_score(labels_test_list, outputs_test["predicted_class"]),
)
print(
    "F1: ",
    metrics.f1_score(
        labels_test_list, outputs_test["predicted_class"], average="macro"
    ),
)
print(
    "Precision: ",
    metrics.precision_score(
        labels_test_list, outputs_test["predicted_class"], average="macro"
    ),
)
```

Last updated on January 24, 2025

[String Encoding](https://docs.turboml.com/non_numeric_inputs/string_encoding/ "String Encoding") [Half Space Trees](https://docs.turboml.com/anomaly_detection/hst/ "Half Space Trees")
</page_content>

# Feature Engineering - Python UDFs
@ TurboML - page_link: https://docs.turboml.com/feature_engineering/udf/
<page_content>
Feature Engineering

UDF

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/UDF.ipynb)

```import turboml as tb
```

```transactions = tb.datasets.FraudDetectionDatasetFeatures()[:100].to_online(
    id="udf_transactions", load_if_exists=True
)
```

### Simple User Defined function [Permalink for this section](https://docs.turboml.com/feature_engineering/udf/\#simple-user-defined-function)

For creating a user defined function first create a separate python file containing the function along with the imports used by it; the function should process the data and return a value. In the below example we have shown a simple example of a function that takes a value and then returns its sine value.

```myfunction_contents = """
import numpy as np


def myfunction(x):
    return np.sin(x)
"""
```

### User Defined Functions - Multiple Input example [Permalink for this section](https://docs.turboml.com/feature_engineering/udf/\#user-defined-functions---multiple-input-example)

We saw that the above user defined function is very simple. We can also create a more complicated function with multiple inputs, we can perform string processing etc

```my_complex_function_contents = """
def my_complex_function(x, y):
    if x.lower() == y.lower():
        return 1
    else:
        return 0
"""
```

### Rich User Defined Functions [Permalink for this section](https://docs.turboml.com/feature_engineering/udf/\#rich-user-defined-functions)

```%pip install psycopg_pool psycopg['binary'] psycopg2-binary
```

```my_rich_function_contents = """
from turboml.common.feature_engineering import TurboMLScalarFunction
from psycopg_pool import ConnectionPool


class PostgresLookup(TurboMLScalarFunction):
    def __init__(self, user, password, host, port, dbname):
        conninfo = (
            f"user={user} password={password} host={host} port={port} dbname={dbname}"
        )
        self.connPool = ConnectionPool(conninfo=conninfo)

    def func(self, index: str):
        with self.connPool.connection() as risingwaveConn:
            with risingwaveConn.cursor() as cur:
                query = 'SELECT "model_length" FROM r2dt_models WHERE id = %s'
                cur.execute(query, (index,))
                result = cur.fetchone()
        return result[0] if result else 0
"""
```

We can create a rich UDF and materialize it.

```transactions.feature_engineering.create_rich_udf_features(
    new_feature_name="lookup_feature",
    argument_names=["index"],
    function_name="lookup",
    class_file_contents=my_rich_function_contents,
    libraries=["psycopg_pool", "psycopg[binary]", "psycopg2-binary"],
    class_name="PostgresLookup",
    dev_initializer_arguments=["reader", "NWDMCE5xdipIjRrp", "hh-pgsql-public.ebi.ac.uk", "5432", "pfmegrnargs"],
    prod_initializer_arguments=["reader", "NWDMCE5xdipIjRrp", "hh-pgsql-public.ebi.ac.uk", "5432", "pfmegrnargs"],
)

transactions.feature_engineering.materialize_features(["lookup_feature"])
```

## Feature Engineering using User Defined Functions (UDF) [Permalink for this section](https://docs.turboml.com/feature_engineering/udf/\#feature-engineering-using-user-defined-functions-udf)

Make sure the libraries that are specified are pip installable and hence named appropriately, for example, if the UDF uses a sklearn function, then the library to be installed should be "scikit-learn" (and not "sklearn")

```transactions.feature_engineering.create_udf_features(
    new_feature_name="sine_of_amount",
    argument_names=["transactionAmount"],
    function_name="myfunction",
    function_file_contents=myfunction_contents,
    libraries=["numpy"],
)
```

```transactions.feature_engineering.create_udf_features(
    new_feature_name="transaction_location_overlap",
    argument_names=["ipCountryCode", "paymentBillingCountryCode"],
    function_name="my_complex_function",
    function_file_contents=my_complex_function_contents,
    libraries=[],
)
```

```transactions.feature_engineering.get_local_features()
```

```transactions.feature_engineering.materialize_features(
    ["sine_of_amount", "transaction_location_overlap"]
)
```

```transactions.feature_engineering.get_materialized_features()
```

Last updated on January 24, 2025

[PySAD Example](https://docs.turboml.com/wyo_models/pysad_example/ "PySAD Example") [UDAF](https://docs.turboml.com/feature_engineering/udaf/ "UDAF")
</page_content>

# Native Python Models
@ TurboML - page_link: https://docs.turboml.com/wyo_models/native_python_model/
<page_content>
Write Your Own Models

Native Python Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/native_python_model.ipynb)

While TurboML offers a wide array of algorithms implemented with performant machine-native code, we also
give you the flexibility to use your own models in Python when necessary, allowing the use of any public
library from PyPi. Lets walk through some simple examples for model based on [River (opens in a new tab)](https://riverml.xyz/latest/)
and [scikit-learn (opens in a new tab)](https://scikit-learn.org/stable/).

```import turboml as tb
```

```!pip install river
```

```import pandas as pd
```

## Prepare an Evaluation Dataset [Permalink for this section](https://docs.turboml.com/wyo_models/native_python_model/\#prepare-an-evaluation-dataset)

We choose a standard Credit Card Fraud dataset that ships with River to evaluate our models on.

```features = tb.datasets.CreditCardsDatasetFeatures()
labels = tb.datasets.CreditCardsDatasetLabels()

features
```

```features.df.loc[0]
```

### And finally load them as datasets in the TurboML Platform [Permalink for this section](https://docs.turboml.com/wyo_models/native_python_model/\#and-finally-load-them-as-datasets-in-the-turboml-platform)

```features = tb.OnlineDataset.from_local_dataset(
    features, "cc_features", load_if_exists=True
)
labels = tb.OnlineDataset.from_local_dataset(labels, "cc_labels", load_if_exists=True)
```

### Isolate features [Permalink for this section](https://docs.turboml.com/wyo_models/native_python_model/\#isolate-features)

```numerical_cols = features.preview_df.columns.tolist()
numerical_cols.remove("index")
input_features = features.get_model_inputs(numerical_fields=numerical_cols)
label = labels.get_model_labels(label_field="score")
```

## Structure of User Defined Models [Permalink for this section](https://docs.turboml.com/wyo_models/native_python_model/\#structure-of-user-defined-models)

A custom Python model must implement 3 instance methods - `learn_one`, `predict_one` and `init_imports`.
The interface and usage is described below and explored further in the examples contained in this notebook.

```class CustomModel:
    def init_imports(self):
        """
        Import any external symbols/modules used in this class
        """
        pass

    def learn_one(self, input: types.InputData):
        """
        Receives labelled data for the model to learn from
        """
        pass

    def predict_one(self, input: types.InputData, output: types.OutputData):
        """
        Receives input features for a prediction, must pass output to the
        output object
        """
        pass
```

## Example - Leveraging [River (opens in a new tab)](https://riverml.xyz/) [Permalink for this section](https://docs.turboml.com/wyo_models/native_python_model/\#example---leveraging-river)

River is a popular ML library for online machine learning, river comes with an inbuilt functionality for `learn_one` and `predict_one` out of the box, however it is important to note the differences in input to the User Defined models and the input of river model, which takes a dictionary and label as inputs for a supervised algorithm. In this example we create a custom model using river according to the standards mentioned above and put it in a separate python module.

```from river import linear_model
import turboml.common.pytypes as types


class MyLogisticRegression:
    def __init__(self):
        self.model = linear_model.LogisticRegression()

    def init_imports(self):
        from river import linear_model

    def learn_one(self, input: types.InputData):
        self.model.learn_one(dict(enumerate(input.numeric)), input.label)

    def predict_one(self, input: types.InputData, output: types.OutputData):
        score = float(self.model.predict_one(dict(enumerate(input.numeric))))
        output.set_score(score)

        # example: setting embeddings
        # output.resize_embeddings(3)
        # mut = output.embeddings()
        # mut[0] = 1
        # mut[1] = 2
        # mut[2] = 3

        # example: appending to feature scores
        # this api is an alternative to resize + set as above,
        # but less efficient
        # output.append_feature_score(0.5)
```

Since python packages can have multiple external dependencies we can make use of `tb.setup_venv(name_of_venv, [List of packages])`. This can create a virtual environment that enables interaction with the platform and the installation of external dependencies with ease.

```venv = tb.setup_venv("my_river_venv", ["river", "numpy"])
venv.add_python_class(MyLogisticRegression)
```

```river_model = tb.Python(class_name=MyLogisticRegression.__name__, venv_name=venv.name)
```

```deployed_model_river = river_model.deploy(
    "river_model", input=input_features, labels=label
)
```

```import matplotlib.pyplot as plt

deployed_model_river.add_metric("WindowedRMSE")
model_auc_scores = deployed_model_river.get_evaluation("WindowedRMSE")
plt.plot([model_auc_score.metric for model_auc_score in model_auc_scores])
```

## Example - An Online Model with Sci-Kit Learn [Permalink for this section](https://docs.turboml.com/wyo_models/native_python_model/\#example---an-online-model-with-sci-kit-learn)

Using Scikit learn you can implement online learning something similar to the code example below using `partial_fit()`.

```!pip install scikit-learn
```

```from sklearn.linear_model import Perceptron
import numpy as np
import turboml.common.pytypes as types


class MyPerceptron:
    def __init__(self):
        self.model = Perceptron()
        self.fitted = False

    def init_imports(self):
        from sklearn.linear_model import Perceptron

    def learn_one(self, input: types.InputData):
        if not self.fitted:
            self.model.partial_fit(
                np.array(input.numeric).reshape(1, -1),
                np.array(input.label).reshape(-1),
                classes=[0, 1],
            )
            self.fitted = True
        else:
            self.model.partial_fit(
                np.array(input.numeric).reshape(1, -1),
                np.array(input.label).reshape(-1),
            )

    def predict_one(self, input: types.InputData, output: types.OutputData):
        if self.fitted:
            score = self.model.predict(np.array(input.numeric).reshape(1, -1))[0]
            output.set_score(score)
        else:
            output.set_score(0.0)
```

```venv = tb.setup_venv("my_sklearn_venv", ["scikit-learn"])
venv.add_python_class(MyPerceptron)
```

```sklearn_model = tb.Python(class_name=MyPerceptron.__name__, venv_name=venv.name)
```

```deployed_model_sklearn = sklearn_model.deploy(
    "sklearn_model", input=input_features, labels=label
)
```

```import matplotlib.pyplot as plt

deployed_model_sklearn.add_metric("WindowedRMSE")
model_auc_scores = deployed_model_sklearn.get_evaluation("WindowedRMSE")
plt.plot([model_auc_score.metric for model_auc_score in model_auc_scores])
```

## Example - Leveraging [Vowpal Wabbit (opens in a new tab)](https://vowpalwabbit.org/) [Permalink for this section](https://docs.turboml.com/wyo_models/native_python_model/\#example---leveraging-vowpal-wabbit)

Vowpal Wabbit provides fast, efficient, and flexible online machine learning techniques for reinforcement learning, supervised learning, and more.

In this example we use the new `vowpal-wabbit-next` Python bindings. Note that we need to transform our input to Vowpal's native text format.

```!pip install vowpal-wabbit-next
```

```import vowpal_wabbit_next as vw
import turboml.common.pytypes as types


class MyVowpalModel:
    def __init__(self):
        self.vw_workspace = vw.Workspace()
        self.vw_parser = vw.TextFormatParser(self.vw_workspace)

    def init_imports(self):
        import vowpal_wabbit_next as vw

    def to_vw_format(self, features, label=None):
        "Convert a feature vector into the Vowpal Wabbit format"
        label_place = f"{label} " if label is not None else ""
        vw_text = f"{label_place}| {' '.join([f'{idx}:{feat}' for idx, feat in enumerate(features, start=1)])}\n"
        return self.vw_parser.parse_line(vw_text)

    def predict_one(self, input: types.InputData, output: types.OutputData):
        vw_format = self.to_vw_format(input.numeric)
        output.set_score(self.vw_workspace.predict_one(vw_format))

    def learn_one(self, input: types.InputData):
        vw_format = self.to_vw_format(input.numeric, input.label)
        self.vw_workspace.learn_one(vw_format)
```

In the below cell we make use of the custom virtual environment created before to install new packages in this case vowpalwabbit. We have to ensure that the name of the virtual environment remains the same and we can reuse the virtual environment multiple times.

```venv = tb.setup_venv("my_vowpal_venv", ["vowpal-wabbit-next"])
venv.add_python_class(MyVowpalModel)
```

```vw_model = tb.Python(class_name=MyVowpalModel.__name__, venv_name=venv.name)
```

```deployed_model_vw = vw_model.deploy("vw_model", input=input_features, labels=label)
```

```import matplotlib.pyplot as plt

deployed_model_vw.add_metric("WindowedRMSE")
model_auc_scores = deployed_model_vw.get_evaluation("WindowedRMSE")
plt.plot([model_auc_score.metric for model_auc_score in model_auc_scores])
```

Last updated on January 24, 2025

[OCR](https://docs.turboml.com/byo_models/ocr_example/ "OCR") [Ensemble Python Model](https://docs.turboml.com/wyo_models/ensemble_python_model/ "Ensemble Python Model")
</page_content>

# ONNX tutorial with Scikit-Learn
@ TurboML - page_link: https://docs.turboml.com/byo_models/onnx_sklearn/
<page_content>
Bring Your Own Models

ONNX - Scikit-Learn

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/onnx_sklearn.ipynb)

```import turboml as tb
```

```!pip install onnx==1.14.1 scikit-learn skl2onnx
```

## Scikit Learn - Standard Model Training [Permalink for this section](https://docs.turboml.com/byo_models/onnx_sklearn/\#scikit-learn---standard-model-training)

The following blocks of code define a standard sklearn training code. This is completely independent of TurboML.

```import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
import matplotlib.pyplot as plt
```

```transactions = tb.datasets.FraudDetectionDatasetFeatures()
labels = tb.datasets.FraudDetectionDatasetLabels()
```

```joined_df = pd.merge(transactions.df, labels.df, on="transactionID", how="right")
joined_df
```

```X = joined_df.drop("is_fraud", axis=1)
y = joined_df["is_fraud"]
```

```numerical_fields = [\
    "transactionAmount",\
    "localHour",\
    "isProxyIP",\
    "digitalItemCount",\
    "physicalItemCount",\
]
X = X[numerical_fields]
```

```X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
```

```y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Export model to ONNX format [Permalink for this section](https://docs.turboml.com/byo_models/onnx_sklearn/\#export-model-to-onnx-format)

Exporting a model to ONNX format depends on the framework. Tutorials for different frameworks can be found at [https://github.com/onnx/tutorials#converting-to-onnx-format (opens in a new tab)](https://github.com/onnx/tutorials#converting-to-onnx-format)

```initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
onx = convert_sklearn(
    clf, initial_types=initial_type, options={type(clf): {"zipmap": False}}
)
onx = select_model_inputs_outputs(onx, outputs=["probabilities"])
```

## Create an ONNX model with TurboML [Permalink for this section](https://docs.turboml.com/byo_models/onnx_sklearn/\#create-an-onnx-model-with-turboml)

Now that we've converted the model to ONNX format, we can deploy it with TurboML.

```transactions = transactions.to_online(id="transactions", load_if_exists=True)
labels = labels.to_online(id="transaction_labels", load_if_exists=True)

features = transactions.get_model_inputs(numerical_fields=numerical_fields)
label = labels.get_model_labels(label_field="is_fraud")
```

```tb.set_onnx_model("randomforest", onx.SerializeToString())
onnx_model = tb.ONNX(model_save_name="randomforest")
```

```deployed_model = onnx_model.deploy("onnx_model", input=features, labels=label)
```

```deployed_model.add_metric("WindowedAUC")
```

```model_auc_scores = deployed_model.get_evaluation("WindowedAUC")
plt.plot([model_auc_score.metric for model_auc_score in model_auc_scores])
```

```
```

Last updated on January 24, 2025

[ONNX - Pytorch](https://docs.turboml.com/byo_models/onnx_pytorch/ "ONNX - Pytorch") [ONNX - Tensorflow](https://docs.turboml.com/byo_models/onnx_tensorflow/ "ONNX - Tensorflow")
</page_content>

# Adaptive LightGBM
@ TurboML - page_link: https://docs.turboml.com/general_purpose/adaptivelgbm/
<page_content>
General Purpose

Adaptive LightGBM

LightGBM implementation to handle concept drift based on Adaptive XGBoost for Evolving Data Streams[1](https://docs.turboml.com/general_purpose/adaptivelgbm/#user-content-fn-1).

## Parameters [Permalink for this section](https://docs.turboml.com/general_purpose/adaptivelgbm/\#parameters)

- **n\_classes**( `int`) → The `num_class` parameter from XGBoost.

- **learning\_rate**(Default: `0.3`) → The `eta` parameter from XGBoost.

- **max\_depth**(Default: `6`) → The `max_depth` parameter from XGBoost.

- **max\_window\_size**(Default: `1000`) → Max window size for drift detection.

- **min\_window\_size**(Default: `0`) → Min window size for drift detection.

- **max\_buffer**(Default: `5`) → Buffers after which to stop growing and start replacing.

- **pre\_train**(Default: `2`) → Buffers to wait before the first XGBoost training.

- **detect\_drift**(Default: `True`) → If set will use a drift detector (ADWIN).

- **use\_updater**(Default: `True`) → Uses `refresh` updated for XGBoost.

- **trees\_per\_train**(Default: `1`) → The number of trees for each training run.


## Example Usage [Permalink for this section](https://docs.turboml.com/general_purpose/adaptivelgbm/\#example-usage)

We can create an instance and deploy AdaptiveLGBM model like this.

```import turboml as tb
model = tb.AdaptiveLGBM(n_classes=2)
```

## Footnotes [Permalink for this section](https://docs.turboml.com/general_purpose/adaptivelgbm/\#footnote-label)

1. J. Montiel, R. Mitchell, E. Frank, B. Pfahringer, T. Abdessalem and A. Bifet [Adaptive XGBoost for Evolving Data Streams (opens in a new tab)](https://arxiv.org/abs/2005.07353) [↩](https://docs.turboml.com/general_purpose/adaptivelgbm/#user-content-fnref-1)


Last updated on January 24, 2025

[Adaptive XGBoost](https://docs.turboml.com/general_purpose/adaptivexgboost/ "Adaptive XGBoost") [Neural Networks](https://docs.turboml.com/general_purpose/neuralnetwork/ "Neural Networks")
</page_content>

# HoeffdingTreeRegressor
@ TurboML - page_link: https://docs.turboml.com/regression/hoeffdingtreeregressor/
<page_content>
Regression

Hoeffding Tree Regressor

The **Hoeffding Tree Regressor** (HTR) is an adaptation of the incremental tree algorithm of the same name for classification. Similarly to its classification counterpart, HTR uses the Hoeffding bound to control its split decisions. Differently from the classification algorithm, HTR relies on calculating the reduction of variance in the target space to decide among the split candidates. The smallest the variance at its leaf nodes, the more homogeneous the partitions are. At its leaf nodes, HTR fits either linear models or uses the target average as the predictor.

## Parameters [Permalink for this section](https://docs.turboml.com/regression/hoeffdingtreeregressor/\#parameters)

- **grace\_period**( `int`, Default: `200`) → Number of instances a leaf should observe between split attempts.

- **delta**( `float`, Default: `1e-07`) → Significance level to calculate the Hoeffding bound. The significance level is given by `1 - delta`. Values closer to zero imply longer split decision delays.

- **tau**( `float`,Default: `0.05`) → Threshold below which a split will be forced to break ties.

- **leaf\_prediction**( `str`,Default: `mean`) → Prediction mechanism used at leafs. For now, only Target mean ( `mean`) is supported.


## Example Usage [Permalink for this section](https://docs.turboml.com/regression/hoeffdingtreeregressor/\#example-usage)

We can create an instance of the Hoeffding Tree Regressor model like this.

```import turboml as tb
htc_model = tb.HoeffdingTreeRegressor()
```

Last updated on January 24, 2025

[FFM Regressor](https://docs.turboml.com/regression/ffmregressor/ "FFM Regressor") [SGT Regressor](https://docs.turboml.com/regression/sgtregressor/ "SGT Regressor")
</page_content>

# Python Model: PySAD Example
@ TurboML - page_link: https://docs.turboml.com/wyo_models/pysad_example/
<page_content>
Write Your Own Models

PySAD Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/pysad_example.ipynb)

In this example, we use the `PySAD` package to monitor anomalies in our streaming data.

```import turboml as tb
```

We start off by installing and importing the `pysad` package along with its dependencies.

```!pip install pysad mmh3==2.5.1
```

```import pandas as pd
import numpy as np
from pysad.models import xStream
```

## Model Definition [Permalink for this section](https://docs.turboml.com/wyo_models/pysad_example/\#model-definition)

TurboML's inbuilt `PythonModel` can be used to define custom models which are compatible with TurboML.

Here we define `PySADModel` as a wrapper using `PySAD`'s `xStream` model, making sure to properly implement the required instance methods for a Python model.

```import turboml.common.pytypes as types


class PySADModel:
    def __init__(self):
        self.model = xStream()

    def init_imports(self):
        from pysad.models import xStream
        import numpy as np

    def learn_one(self, input: types.InputData):
        self.model = self.model.fit_partial(np.array(input.numeric))

    def predict_one(self, input: types.InputData, output: types.OutputData):
        score = self.model.score_partial(np.array(input.numeric))
        output.set_score(score)
```

Now, we create a custom `venv` so that the custom model defined above has access to all the required dependencies. PySAD required mmh3==2.5.1 as per their docs.

```venv = tb.setup_venv("my_pysad_venv", ["mmh3==2.5.1", "pysad", "numpy"])
venv.add_python_class(PySADModel)
```

## Dataset [Permalink for this section](https://docs.turboml.com/wyo_models/pysad_example/\#dataset)

We choose our standard `FraudDetection` dataset, using the `to_online` method to push it to the platform.

```transactions = tb.datasets.FraudDetectionDatasetFeatures().to_online(
    id="transactions", load_if_exists=True
)
labels = tb.datasets.FraudDetectionDatasetLabels().to_online(
    id="transaction_labels", load_if_exists=True
)
```

```numerical_fields = [\
    "transactionAmount",\
    "localHour",\
    "isProxyIP",\
    "digitalItemCount",\
    "physicalItemCount",\
]
```

```features = transactions.get_model_inputs(numerical_fields=numerical_fields)
label = labels.get_model_labels(label_field="is_fraud")
```

## Model Deployment [Permalink for this section](https://docs.turboml.com/wyo_models/pysad_example/\#model-deployment)

Now, we deploy our model and extract its outputs.

```pysad_model = tb.Python(class_name=PySADModel.__name__, venv_name=venv.name)
```

```deployed_model_pysad = pysad_model.deploy("pysad_model", input=features, labels=label)
```

```outputs = deployed_model_pysad.get_outputs()
```

```len(outputs)
```

## Evaluation [Permalink for this section](https://docs.turboml.com/wyo_models/pysad_example/\#evaluation)

Finally, we use any of `PySAD`'s metrics for giving a numerical value to the degree of the presence of anomalies in our data.

```from pysad.evaluation import AUROCMetric
```

```auroc = AUROCMetric()
labels_df = labels.preview_df
for output, y in zip(
    outputs, labels_df["is_fraud"].tolist()[: len(outputs)], strict=False
):
    auroc.update(y, output.score)
```

```auroc.get()
```

Last updated on January 24, 2025

[Batch Python Model](https://docs.turboml.com/wyo_models/batch_python_model/ "Batch Python Model") [UDF](https://docs.turboml.com/feature_engineering/udf/ "UDF")
</page_content>

# Batch APIs
@ TurboML - page_link: https://docs.turboml.com/general_examples/batch_api/
<page_content>
General

Batch API

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/batch_api.ipynb)

The main mode of operation in TurboML is streaming, with continuous updates to different components with fresh data. However, TurboML also supports the good ol' fashioned batch APIs. We've already seen examples of this for feature engineering in the quickstart notebook. In this notebook, we'll focus primarily on batch APIs for ML modelling.

To make this more interesting, we'll show how we can still have incremental training on batch data.

```import turboml as tb
```

```import pandas as pd
from sklearn import metrics
```

## Dataset [Permalink for this section](https://docs.turboml.com/general_examples/batch_api/\#dataset)

We'll use our standard `FraudDetection` dataset again, but this time without pushing it to the platform. Interfaces like feature engineering and feature selection work in the exact same ways, just without being linked
to a platform-managed dataset.

```transactions = tb.datasets.FraudDetectionDatasetFeatures()
labels = tb.datasets.FraudDetectionDatasetLabels()

transactions_p1 = transactions[:100000]
labels_p1 = labels[:100000]

transactions_p2 = transactions[100000:]
labels_p2 = labels[100000:]
```

```numerical_fields = [\
    "transactionAmount",\
    "localHour",\
]
categorical_fields = [\
    "digitalItemCount",\
    "physicalItemCount",\
    "isProxyIP",\
]
features = transactions_p1.get_model_inputs(
    numerical_fields=numerical_fields, categorical_fields=categorical_fields
)
label = labels_p1.get_model_labels(label_field="is_fraud")
```

## Training [Permalink for this section](https://docs.turboml.com/general_examples/batch_api/\#training)

With the features and label defined, we can train a model in a batch way using the learn method.

```model = tb.HoeffdingTreeClassifier(n_classes=2)
```

```model_trained_100K = model.learn(features, label)
```

We've trained a model on the first 100K rows. Now, to update this model on the remaining data, we can create another batch dataset and call the `learn` method. Note that this time, learn is called on a trained model.

```features = transactions_p2.get_model_inputs(
    numerical_fields=numerical_fields, categorical_fields=categorical_fields
)
label = labels_p2.get_model_labels(label_field="is_fraud")
```

```model_fully_trained = model_trained_100K.learn(features, label)
```

## Inference [Permalink for this section](https://docs.turboml.com/general_examples/batch_api/\#inference)

We've seen batch inference on deployed models in the quickstart notebook. We can also perform batch inference on these models using the `predict` method.

```outputs = model_trained_100K.predict(features)
print(metrics.roc_auc_score(labels_p2.df["is_fraud"], outputs["score"]))
outputs = model_fully_trained.predict(features)
print(metrics.roc_auc_score(labels_p2.df["is_fraud"], outputs["score"]))
```

## Deployment [Permalink for this section](https://docs.turboml.com/general_examples/batch_api/\#deployment)

So far, we've only trained a model. We haven't deployed it yet. Deploying a batch trained model is exactly like any other model deployment, except we'll set the `predict_only` option to be True. This means the model won't be updated automatically.

```transactions = transactions.to_online(id="transactions10", load_if_exists=True)
labels = labels.to_online(id="transaction_labels", load_if_exists=True)
```

```features = transactions.get_model_inputs(
    numerical_fields=numerical_fields, categorical_fields=categorical_fields
)
label = labels.get_model_labels(label_field="is_fraud")
```

```deployed_model = model_fully_trained.deploy(
    name="predict_only_model", input=features, labels=label, predict_only=True
)
```

```outputs = deployed_model.get_outputs()
outputs[-1]
```

## Next Steps [Permalink for this section](https://docs.turboml.com/general_examples/batch_api/\#next-steps)

In this notebook, we discussed how to train models in a batch paradigm and deploy them. In a separate notebook we'll cover two different statregies to update models, (i) starting from a batch trained model and using continual learning, (ii) training models incrementally in a batch paradigm and updating the deployment with newer versions.

Last updated on January 24, 2025

[Quickstart](https://docs.turboml.com/quickstart/ "Quickstart") [Local Model](https://docs.turboml.com/general_examples/local_model/ "Local Model")
</page_content>

# ContextualBanditModelSelection
@ TurboML - page_link: https://docs.turboml.com/ensembles/contextualbanditmodelselection/
<page_content>
Ensembles

Contextual Bandit Model Selection

Contextual Bandit-based model selection.

Similar to BanditModelSelection, but now the algorithm uses a contextual bandit leading to more fine-grained model selection.

## Parameters [Permalink for this section](https://docs.turboml.com/ensembles/contextualbanditmodelselection/\#parameters)

- **contextualbandit**(Default: `LinTS`) → The underlying bandit algorithm. Options are: LinTS, and LinUCB.

- **metric\_name**(Default: `WindowedMAE`) → The metric to use to evaluate models. Options are: WindowedAUC, WindowedAccuracy, WindowedMAE, WindowedMSE, and WindowedRMSE.

- **base\_models**( `list[Model]`) → The list of models over which to perform model selection.


## Example Usage [Permalink for this section](https://docs.turboml.com/ensembles/contextualbanditmodelselection/\#example-usage)

We can create an instance and deploy BanditModel like this.

```import turboml as tb
htc_model = tb.HoeffdingTreeRegressor()
amf_model = tb.AMFRegressor()
ffm_model = tb.FFMRegressor()
bandit_model = tb.ContextualBanditModelSelection(base_models = [htc_model, amf_model, ffm_model])
```

Last updated on January 24, 2025

[Bandit Model Selection](https://docs.turboml.com/ensembles/banditmodelselection/ "Bandit Model Selection") [Leveraging Bagging Classifier](https://docs.turboml.com/ensembles/leveragingbaggingclassifier/ "Leveraging Bagging Classifier")
</page_content>

# ONNX tutorial with TensorFlow
@ TurboML - page_link: https://docs.turboml.com/byo_models/onnx_tensorflow/
<page_content>
Bring Your Own Models

ONNX - Tensorflow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/onnx_tensorflow.ipynb)

```import turboml as tb
```

```!pip install -U tensorflow tf2onnx onnx==1.14.1 scikit-learn
```

## Tensorflow - Standard Model Training [Permalink for this section](https://docs.turboml.com/byo_models/onnx_tensorflow/\#tensorflow---standard-model-training)

The following blocks of code define a standard tensorflow training code. This is completely independent of TurboML.

```import pandas as pd
import tf2onnx.convert
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

```transactions = tb.datasets.FraudDetectionDatasetFeatures()
labels = tb.datasets.FraudDetectionDatasetLabels()
```

```joined_df = pd.merge(transactions.df, labels.df, on="transactionID", how="right")
joined_df
```

```X = joined_df.drop("is_fraud", axis=1)
numerical_fields = [\
    "transactionAmount",\
    "localHour",\
    "isProxyIP",\
    "digitalItemCount",\
    "physicalItemCount",\
]

feats = X[numerical_fields]
targets = joined_df["is_fraud"].astype(int)
```

```from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    feats, targets, test_size=0.2, random_state=42
)
```

```model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation="relu"))
model.add(Dense(2, activation="softmax"))
```

```model.summary()
```

```model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

num_epochs = 10
model.fit(X_train, y_train, epochs=num_epochs, batch_size=64, verbose=1)
```

```_, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)
```

## Export model to ONNX format [Permalink for this section](https://docs.turboml.com/byo_models/onnx_tensorflow/\#export-model-to-onnx-format)

Exporting a model to ONNX format depends on the framework. Tutorials for different frameworks can be found at [https://github.com/onnx/tutorials#converting-to-onnx-format (opens in a new tab)](https://github.com/onnx/tutorials#converting-to-onnx-format)

```onnx_model_path = "tensorflow_model.onnx"
input_signature = [\
    tf.TensorSpec([1, len(numerical_fields)], tf.float32, name="keras_tensor")\
]
model.output_names = ["output"]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

onnx_model_bytes = onnx_model.SerializeToString()
```

## Create an ONNX model with TurboML [Permalink for this section](https://docs.turboml.com/byo_models/onnx_tensorflow/\#create-an-onnx-model-with-turboml)

Now that we've converted the model to ONNX format, we can deploy it with TurboML.

```transactions = transactions.to_online(id="transactions", load_if_exists=True)
labels = labels.to_online(id="transaction_labels", load_if_exists=True)

features = transactions.get_model_inputs(numerical_fields=numerical_fields)
label = labels.get_model_labels(label_field="is_fraud")
```

```tb.set_onnx_model("tensorflowmodel", onnx_model_bytes)
onnx_model = tb.ONNX(model_save_name="tensorflowmodel")
```

```deployed_model = onnx_model.deploy("onnx_model_tf", input=features, labels=label)
```

```deployed_model.add_metric("WindowedAUC")
```

```model_auc_scores = deployed_model.get_evaluation("WindowedAUC")
plt.plot([model_auc_score.metric for model_auc_score in model_auc_scores])
```

```
```

Last updated on January 24, 2025

[ONNX - Scikit-Learn](https://docs.turboml.com/byo_models/onnx_sklearn/ "ONNX - Scikit-Learn") [TF-IDF Example](https://docs.turboml.com/byo_models/tfidf_example/ "TF-IDF Example")
</page_content>

# Local Model
@ TurboML - page_link: https://docs.turboml.com/general_examples/local_model/
<page_content>
General

Local Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/local_model.ipynb)

LocalModel is our Python interface that gives direct access to TurboML's machine learning models.

We will use the transactions.csv and labels.csv datasets for our experiments.

```import turboml as tb
```

```import pandas as pd
from turboml import LocalModel
from turboml.common.models import InputSpec
import numpy as np
from sklearn import metrics
import time
import base64
```

## Load Datasets [Permalink for this section](https://docs.turboml.com/general_examples/local_model/\#load-datasets)

```transactions = tb.datasets.FraudDetectionDatasetFeatures()
labels = tb.datasets.FraudDetectionDatasetLabels()

transactions_train = transactions[:100000]
labels_train = labels[:100000]

transactions_test = transactions[100000:120000]
labels_test = labels[100000:120000]
```

## Define Input Specification [Permalink for this section](https://docs.turboml.com/general_examples/local_model/\#define-input-specification)

```numerical_fields = [\
    "transactionAmount",\
    "localHour",\
]

categorical_fields = [\
    "digitalItemCount",\
    "physicalItemCount",\
    "isProxyIP",\
]

input_spec = InputSpec(
    key_field="index", # If this is mentioned - "index" - it uses the "key_field", present in the Dataset(LocalDataset/OnlineDataset) in TurboML. If you want to override, set the specific column name  
    numerical_fields=numerical_fields,
    categorical_fields=categorical_fields,
    textual_fields=[],
    imaginal_fields=[],
    time_field="",
    label_field="is_fraud",
)
```

## Prepare Input and Label Data [Permalink for this section](https://docs.turboml.com/general_examples/local_model/\#prepare-input-and-label-data)

```train_features = transactions_train.get_model_inputs(
    numerical_fields=numerical_fields, categorical_fields=categorical_fields
)
train_labels = labels_train.get_model_labels(label_field="is_fraud")

test_features = transactions_test.get_model_inputs(
    numerical_fields=numerical_fields, categorical_fields=categorical_fields
)
test_labels = labels_test.get_model_labels(label_field="is_fraud")
```

## Define Model Configurations [Permalink for this section](https://docs.turboml.com/general_examples/local_model/\#define-model-configurations)

```hoeffding_tree = tb.HoeffdingTreeClassifier(
    delta=1e-7,
    tau=0.05,
    grace_period=200,
    n_classes=2,
    leaf_pred_method="mc",
    split_method="gini",
)

amf_classifier = tb.AMFClassifier(
    n_classes=2,
    n_estimators=10,
    step=1,
    use_aggregation=True,
    dirichlet=0.5,
    split_pure=False,
)

multinomial_nb = tb.MultinomialNB(n_classes=2, alpha=1.0)
```

```# Convert each Model instance to LocalModel
hoeffding_tree_local = hoeffding_tree.to_local_model(input_spec)
amf_classifier_local = amf_classifier.to_local_model(input_spec)
multinomial_nb_local = multinomial_nb.to_local_model(input_spec)
```

## Training and Evaluation Function [Permalink for this section](https://docs.turboml.com/general_examples/local_model/\#training-and-evaluation-function)

```# Store trained models and predictions
model_trained_100K = {}
initial_results = {}

models_to_train = [\
    ("HoeffdingTree", hoeffding_tree_local),\
    ("AMF", amf_classifier_local),\
    ("MultinomialNB", multinomial_nb_local),\
]
```

```for name, model in models_to_train:
    try:
        print(f"Training {name} model on first 100K records...")
        model.learn(train_features, train_labels)

        predictions = model.predict(test_features)
        roc_auc = metrics.roc_auc_score(
            test_labels.dataframe["is_fraud"], predictions["score"]
        )
        accuracy = metrics.accuracy_score(
            test_labels.dataframe["is_fraud"], predictions["predicted_class"]
        )

        print(f"{name} Model Results:")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Accuracy Score: {accuracy:.4f}")

        # Store results
        model_trained_100K[name] = model
        initial_results[name] = predictions

    except Exception as e:
        print(f"Error with {name} model: {str(e)}")
```

## Further Training in Batches [Permalink for this section](https://docs.turboml.com/general_examples/local_model/\#further-training-in-batches)

We will continue training the Hoeffding Tree model with additional data in batches.

```model_hoeffding_tree = model_trained_100K.get("HoeffdingTree")
start = 100000
step = 100
stop = 102000

if model_hoeffding_tree is not None:
    # Split the dataset into 10 parts for batch training
    pos = start
    i = 0
    while pos < stop - step:
        print(f"\nPreparing batch {i + 1}...")
        feat_batch = transactions[pos : pos + step].get_model_inputs(
            numerical_fields=numerical_fields,
            categorical_fields=categorical_fields,
        )
        label_batch = labels[pos : pos + step].get_model_labels(label_field="is_fraud")
        pos = pos + step
        i += 1

        print(f"Training batch {i + 1}...")
        start_time = time.time()
        model_hoeffding_tree.learn(feat_batch, label_batch)
        end_time = time.time()
        print(
            f"Batch {i + 1} training completed in {end_time - start_time:.2f} seconds."
        )
else:
    print("Hoeffding Tree model not found in trained models.")
```

## ONNX Model [Permalink for this section](https://docs.turboml.com/general_examples/local_model/\#onnx-model)

```!pip install onnx==1.14.1 scikit-learn skl2onnx river
```

```from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Prepare features and target
transactions_df = transactions.df
labels_df = labels.df

X = transactions_df[numerical_fields + categorical_fields + ["transactionID"]]
y = labels_df["is_fraud"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train sklearn model
clf = RandomForestClassifier()
clf.fit(X_train[numerical_fields + categorical_fields], y_train)
```

```# Convert to ONNX format
initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
onx = convert_sklearn(
    clf, initial_types=initial_type, options={type(clf): {"zipmap": False}}
)

# Get the serialized ONNX model
onnx_model_data = onx.SerializeToString()
# Base64-encode the ONNX model data
model_data_base64 = base64.b64encode(onnx_model_data).decode("utf-8")
```

```# Create ONNX model config with the encoded model data
onnx_model_config = [\
    {\
        "algorithm": "ONNX",\
        "onnx_config": {\
            "model_save_name": "randomforest",\
            "model_data": model_data_base64,\
        },\
    }\
]


onnx_input_spec = InputSpec(
    key_field="index",  # If this is mentioned - "index" - it uses the "key_field", present in the Dataset(LocalDataset/OnlineDataset) in TurboML. If you want to override, set the specific column name 
    numerical_fields=numerical_fields + categorical_fields,
    categorical_fields=[],
    textual_fields=[],
    imaginal_fields=[],
    time_field="",
    label_field="is_fraud",
)

local_onnx_model = LocalModel(
    model_configs=onnx_model_config,
    input_spec=onnx_input_spec,
)
```

```# train data
train_input_data = tb.LocalDataset.from_pd(
    df=X_train, key_field="transactionID"
).get_model_inputs(numerical_fields=numerical_fields + categorical_fields)


train_label_data = tb.LocalDataset.from_pd(
    df=pd.DataFrame({"transactionID": X_train.transactionID, "is_fraud": y_train}),
    key_field="transactionID",
).get_model_labels(label_field="is_fraud")
```

```# Create test input data
test_input_data = tb.LocalDataset.from_pd(
    df=X_test, key_field="transactionID"
).get_model_inputs(numerical_fields=numerical_fields + categorical_fields)


test_label_data = tb.LocalDataset.from_pd(
    df=pd.DataFrame({"transactionID": X_test.transactionID, "is_fraud": y_test}),
    key_field="transactionID",
).get_model_labels(label_field="is_fraud")
```

```def onnx_model():
    try:
        # Get predictions
        predictions = local_onnx_model.predict(test_input_data)

        # Calculate metrics
        roc_auc = metrics.roc_auc_score(
            test_label_data.dataframe["is_fraud"],
            predictions["score"],
        )
        accuracy = metrics.accuracy_score(
            test_label_data.dataframe["is_fraud"],
            predictions["predicted_class"],
        )

        print("ONNX Model Results:")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Accuracy Score: {accuracy:.4f}")

        return predictions

    except Exception as e:
        print(f"Error testing ONNX model: {str(e)}")
        return None


# Run the test
predictions = onnx_model()

if predictions is not None:
    sklearn_preds = clf.predict(X_test)
    onnx_preds = predictions["predicted_class"]

    match_rate = (sklearn_preds == onnx_preds).mean()
    print("\nPrediction Comparison:")
    print(f"Sklearn vs ONNX prediction match rate: {match_rate:.4f}")
```

## Python Model Testing [Permalink for this section](https://docs.turboml.com/general_examples/local_model/\#python-model-testing)

```python_model_code = """
from river import linear_model
import turboml.common.pytypes as types

class MyLogisticRegression:

    def init_imports(self):
        from river import linear_model
        import turboml.common.pytypes as types

    def __init__(self):
        self.model = linear_model.LogisticRegression()

    def learn_one(self, input):
        # Combine numerical and categorical features into a dictionary
        features = {}
        features.update({f'num_{i}': val for i, val in enumerate(input.numeric)})
        features.update({f'cat_{i}': val for i, val in enumerate(input.categ)})
        self.model.learn_one(features, input.label)

    def predict_one(self, input, output):
        # Combine numerical and categorical features into a dictionary
        features = {}
        features.update({f'num_{i}': val for i, val in enumerate(input.numeric)})
        features.update({f'cat_{i}': val for i, val in enumerate(input.categ)})
        proba = self.model.predict_proba_one(features)
        score = float(proba.get(True, 0))
        output.set_score(score)
        output.set_predicted_class(int(score >= 0.5))
"""
```

```# Define the model configuration
python_model_config = {
    "algorithm": "Python",
    "python_config": {
        "class_name": "MyLogisticRegression",
        "code": python_model_code,
    },
}

# Create the LocalModel instance
local_python_model = LocalModel(
    model_configs=[python_model_config],
    input_spec=input_spec,
)
```

```# Train the model
local_python_model.learn(train_input_data, train_label_data)

# Make predictions
predictions = local_python_model.predict(test_input_data)

# Evaluate the model
roc_auc = metrics.roc_auc_score(
    test_label_data.dataframe["is_fraud"], predictions["score"]
)
accuracy = metrics.accuracy_score(
    test_label_data.dataframe["is_fraud"], predictions["predicted_class"]
)

print(f"Python Model ROC AUC Score: {roc_auc:.4f}")
print(f"Python Model Accuracy Score: {accuracy:.4f}")
```

## Python Ensemble Model [Permalink for this section](https://docs.turboml.com/general_examples/local_model/\#python-ensemble-model)

```# Base models (already defined and trained)
hoeffding_tree_model = model_trained_100K["HoeffdingTree"]
amf_classifier_model = model_trained_100K["AMF"]
multinomial_nb_model = model_trained_100K["MultinomialNB"]

# Extract base model configurations
base_model_configs = [\
    hoeffding_tree_model.model_configs[0],\
    amf_classifier_model.model_configs[0],\
    multinomial_nb_model.model_configs[0],\
]
```

```# Prepare ensemble model code
ensemble_model_code = """
import turboml.common.pymodel as model
from typing import List

class MyEnsembleModel:
    def __init__(self, base_models: List[model.Model]):
        if not base_models:
            raise ValueError("PythonEnsembleModel requires at least one base model.")
        self.base_models = base_models

    def init_imports(self):
        import turboml.common.pytypes as types
        from typing import List

    def learn_one(self, input):
        for model in self.base_models:
            model.learn_one(input)

    def predict_one(self, input, output):
        total_score = 0.0
        for model in self.base_models:
            model_output = model.predict_one(input)
            total_score += model_output.score()
        average_score = total_score / len(self.base_models)
        output.set_score(average_score)
        output.set_predicted_class(int(average_score >= 0.5))
"""
```

```# Define the ensemble model configuration
ensemble_model_config = {
    "algorithm": "PythonEnsembleModel",
    "python_ensemble_config": {
        "class_name": "MyEnsembleModel",
        "code": ensemble_model_code,
    },
}

# Combine the ensemble model config and base model configs
model_configs = [ensemble_model_config] + base_model_configs

# Create the ensemble LocalModel instance
ensemble_model = tb.LocalModel(
    model_configs=model_configs,
    input_spec=input_spec,
)
```

```# Train the ensemble model
ensemble_model.learn(train_input_data, train_label_data)

# Make predictions with the ensemble model
ensemble_predictions = ensemble_model.predict(test_input_data)

# Evaluate the ensemble model
roc_auc = metrics.roc_auc_score(
    test_label_data.dataframe["is_fraud"], ensemble_predictions["score"]
)
accuracy = metrics.accuracy_score(
    test_label_data.dataframe["is_fraud"], ensemble_predictions["predicted_class"]
)

print(f"Ensemble Model ROC AUC Score: {roc_auc:.4f}")
print(f"Ensemble Model Accuracy Score: {accuracy:.4f}")
```

Last updated on January 24, 2025

[Batch API](https://docs.turboml.com/general_examples/batch_api/ "Batch API") [Stream Dataset Online](https://docs.turboml.com/general_examples/stream_dataset_online/ "Stream Dataset Online")
</page_content>

# Neural Network
@ TurboML - page_link: https://docs.turboml.com/general_purpose/neuralnetwork/
<page_content>
General Purpose

Neural Networks

To parameterize a neural network, we use a configuration based on [ludwig (opens in a new tab)](https://ludwig.ai/). The main building block for neural networks is `tb.NNLayer()` which implements a fully connected layer. The parameters of NNLayer are,

- **output\_size**(Default `64`) → Output size of a fully connected layer.
- **residual\_connections**(list\[int\]) → List of indices of for layers with which to establish residual\_connections.
- **activation**(Default: `relu`) → Default activation function applied to the output of the fully connected layers. Options are `elu`, `leakyRelu`, `logSigmoid`, `relu`, `sigmoid`, `tanh`, and `softmax`.
- **dropout**(Default `0.3`) → Default dropout rate applied to fully connected layers. Increasing dropout is a common form of regularization to combat overfitting. The dropout is expressed as the probability of an element to be zeroed out (0.0 means no dropout)
- **use\_bias**(Default: `True`) → Whether the layer uses a bias vector. Options are True and False.

## Parameters [Permalink for this section](https://docs.turboml.com/general_purpose/neuralnetwork/\#parameters)

- **dropout**(Default `0`) → Dropout value to use for the overall model.
- **layers**(list\[NNLayer\]) → Neural Network layers. By default, we pass 3 layers. `[tb.NNLayer(), tb.NNLayer(), tb.NNLayer(output_size=1, activation="sigmoid")]`
- **loss\_function**(Default: `mse`) → Which loss function to optimize. Options are `l1`, `mse`, `cross_entropy`, `nll`, `poisson_nll` and `bce`.
- **learning\_rate**(Default: `1e-2`) → Initial learning rate.
- **optimizer**(Default: `sgd`) → Which optimizer to use. Options are `sgd` and `adam`.

## Example Usage [Permalink for this section](https://docs.turboml.com/general_purpose/neuralnetwork/\#example-usage)

We can create an instance and deploy Neural Network model like this.

```import turboml as tb
model = tb.NeuralNetwork(layers=[tb.NNLayer(), tb.NNLayer(), tb.NNLayer(output_size=1, activation="sigmoid")])
```

Last updated on January 24, 2025

[Adaptive LightGBM](https://docs.turboml.com/general_purpose/adaptivelgbm/ "Adaptive LightGBM") [Online Neural Networks](https://docs.turboml.com/general_purpose/onlineneuralnetwork/ "Online Neural Networks")
</page_content>

# OCR example using RestAPI Client
@ TurboML - page_link: https://docs.turboml.com/byo_models/ocr_example/
<page_content>
Bring Your Own Models

OCR


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/ocr_example.ipynb)

This example demonstrates using our REST API client for OCR processing.

```import turboml as tb
```

```!pip install surya-ocr
```

```import os
from PIL import Image
import pandas as pd
```

### Launching our FastAPI application with OCR model from jupyter-notebook [Permalink for this section](https://docs.turboml.com/byo_models/ocr_example/\#launching-our-fastapi-application-with-ocr-model-from-jupyter-notebook)

```import subprocess
import threading


def run_uvicorn_server(cmd, ready_event):
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    for line in process.stdout:
        print(line, end="")
        # Check for the message indicating the server has started
        if "Uvicorn running on" in line:
            ready_event.set()
    process.wait()


cmd = "uvicorn utils.ocr_server_app:app --port 5379 --host 0.0.0.0"

server_ready_event = threading.Event()
server_thread = threading.Thread(
    target=run_uvicorn_server, args=(cmd, server_ready_event)
)
server_thread.start()
```

### Loading a dataset of Images [Permalink for this section](https://docs.turboml.com/byo_models/ocr_example/\#loading-a-dataset-of-images)

```import io
import base64

image_dir = "./data/test_images/"
images_test = []
labels_test = []
widths_test = []
heights_test = []

for filename in os.listdir(image_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
        image_path = os.path.join(image_dir, filename)

        # Open and process the image
        with Image.open(image_path) as pil_image:
            pil_image = pil_image.convert("RGB")

            # Get image dimensions
            width, height = pil_image.size

            # Save the image to a bytes buffer
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="JPEG")
            binary_image = img_byte_arr.getvalue()

            # Encode the binary image data to base64
            base64_image = base64.b64encode(binary_image).decode("utf-8")

        images_test.append(base64_image)
        labels_test.append(0)  # Assigning a default label of 0
        widths_test.append(width)
        heights_test.append(height)

image_dict_test = {"images": images_test, "width": widths_test, "height": heights_test}
label_dict_test = {"labels": labels_test}
image_df_test = pd.DataFrame(image_dict_test)
label_df_test = pd.DataFrame(label_dict_test)
image_df_test.reset_index(inplace=True)
label_df_test.reset_index(inplace=True)

print(f"Processed {len(images_test)} images.")
print(f"Image DataFrame shape: {image_df_test.shape}")
print(f"Label DataFrame shape: {label_df_test.shape}")
```

```image_df_test = image_df_test.reset_index(drop=True)
label_df_test = label_df_test.reset_index(drop=True)
```

```images_train = tb.LocalDataset.from_pd(df=image_df_test, key_field="index")
labels_train = tb.LocalDataset.from_pd(df=label_df_test, key_field="index")

images_test = tb.LocalDataset.from_pd(df=image_df_test, key_field="index")
labels_test = tb.LocalDataset.from_pd(df=label_df_test, key_field="index")
```

```imaginal_fields = ["images"]
categorical_fields = ["width", "height"]
features_train = images_train.get_model_inputs(
    imaginal_fields=imaginal_fields, categorical_fields=categorical_fields
)
targets_train = labels_train.get_model_labels(label_field="labels")

features_test = images_test.get_model_inputs(
    imaginal_fields=imaginal_fields, categorical_fields=categorical_fields
)
targets_test = labels_test.get_model_labels(label_field="labels")
```

### Using TurboML to make a request to OCR Server [Permalink for this section](https://docs.turboml.com/byo_models/ocr_example/\#using-turboml-to-make-a-request-to-ocr-server)

```request_model = tb.RestAPIClient(
    server_url="http://0.0.0.0:5379/predict",
    connection_timeout=10000,
    max_request_time=10000,
    max_retries=1,
)
```

```server_ready_event.wait(timeout=100)
```

```model_trained = request_model.learn(features_train, targets_train)
```

```outputs_test = model_trained.predict(features_test)
```

```outputs_test
```

Last updated on January 24, 2025

[ResNet Example](https://docs.turboml.com/byo_models/resnet_example/ "ResNet Example") [Native Python Model](https://docs.turboml.com/wyo_models/native_python_model/ "Native Python Model")
</page_content>

# HeteroAdaBoostClassifier
@ TurboML - page_link: https://docs.turboml.com/ensembles/heteroadaboostclassifer/
<page_content>
Ensembles

Heterogeneous AdaBoost Classifier

Similar to AdaBoostClassifier, but instead of multiple copies of the same model, it can work with different base models.

## Parameters [Permalink for this section](https://docs.turboml.com/ensembles/heteroadaboostclassifer/\#parameters)

- **base\_models**( `list[Model]`) → The list of classifier models.

- **n\_classes**( `int`) → The number of classes for the classifier.

- **seed**( `int`, Default: `0`) → Random number generator seed for reproducibility.


## Example Usage [Permalink for this section](https://docs.turboml.com/ensembles/heteroadaboostclassifer/\#example-usage)

We can create an instance and deploy AdaBoostClassifier model like this.

```import turboml as tb
model = tb.HeteroAdaBoostClassifier(n_classes=2, base_models = [tb.HoeffdingTreeClassifier(n_classes=2), tb.AMFClassifier(n_classes=2)])
```

Last updated on January 24, 2025

[Heterogeneous Leveraging Bagging Classifier](https://docs.turboml.com/ensembles/heteroleveragingbaggingclassifier/ "Heterogeneous Leveraging Bagging Classifier") [LLAMA Embedding](https://docs.turboml.com/pipeline_components/llamaembedding/ "LLAMA Embedding")
</page_content>

# ONNX
@ TurboML - page_link: https://docs.turboml.com/general_purpose/onnx/
<page_content>
General Purpose

ONNX

Using the Open Neural Network Exchange (ONNX) format to load pre-trained weights and use them for prediction. This allows using models trained via frameworks like PyTorch, TensorFlow, Scikit-Learn etc in TurboML. Note: This model doesn't learn.

## Parameters [Permalink for this section](https://docs.turboml.com/general_purpose/onnx/\#parameters)

- **model\_save\_name**(str) → The name used to save the ONNX model weights.

## Example Usage [Permalink for this section](https://docs.turboml.com/general_purpose/onnx/\#example-usage)

We can create an instance and deploy ONNX model like this.

```import turboml as tb
tb.set_onnx_model("randomforest", onx.SerializeToString())
onnx_model = tb.ONNX(model_save_name = "randomforest")
```

Last updated on January 24, 2025

[Online Neural Networks](https://docs.turboml.com/general_purpose/onlineneuralnetwork/ "Online Neural Networks") [AdaBoost Classifier](https://docs.turboml.com/ensembles/adaboostclassifer/ "AdaBoost Classifier")
</page_content>

# Resnet example using gRPC Client
@ TurboML - page_link: https://docs.turboml.com/byo_models/resnet_example/
<page_content>
Bring Your Own Models

ResNet Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/resnet_example.ipynb)

This example demonstrates using our gRPC client to perform inference with the pretrained ResNet18 model.

```import turboml as tb
```

```!pip install kagglehub
```

```import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.resnet_grpc_server import serve
```

### Start gRPC server for pretrained Resnet18 from jupyter-notebook [Permalink for this section](https://docs.turboml.com/byo_models/resnet_example/\#start-grpc-server-for-pretrained-resnet18-from-jupyter-notebook)

```import threading


def run_server_in_background(url):
    serve(url)  # This will start the gRPC server


# Start the server in a separate thread
url = "0.0.0.0:50021"
server_thread = threading.Thread(
    target=run_server_in_background, args=(url,), daemon=True
)
server_thread.start()

print("gRPC server is running in the background...")
```

### Load image Dataset from Kaggle [Permalink for this section](https://docs.turboml.com/byo_models/resnet_example/\#load-image-dataset-from-kaggle)

```import kagglehub
import shutil

# Download latest version
target_path = "./data/animal-image-classification-dataset"
path = kagglehub.dataset_download("borhanitrash/animal-image-classification-dataset")
shutil.move(path, target_path)

print("Dataset stored in:", target_path)
```

```animal_dataset = datasets.ImageFolder(root=target_path, transform=transforms.ToTensor())
data_loader = DataLoader(animal_dataset, batch_size=32, shuffle=True)
images, labels = next(iter(data_loader))
```

### Convert images into bytes array. [Permalink for this section](https://docs.turboml.com/byo_models/resnet_example/\#convert-images-into-bytes-array)

```import io

images_test = []
labels_test = []

for image_tensor, label in zip(images, labels, strict=False):
    image = transforms.ToPILImage()(image_tensor)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    binary_image = img_byte_arr.getvalue()

    images_test.append(binary_image)
    labels_test.append(label.item())

image_dict_test = {"images": images_test}
label_dict_test = {"labels": labels_test}
image_df_test = pd.DataFrame(image_dict_test)
label_df_test = pd.DataFrame(label_dict_test)
image_df_test.reset_index(inplace=True)
label_df_test.reset_index(inplace=True)

print(f"Processed {len(images_test)} images.")
print(f"Image DataFrame shape: {image_df_test.shape}")
print(f"Label DataFrame shape: {label_df_test.shape}")
```

```image_df_test = image_df_test.reset_index(drop=True)
label_df_test = label_df_test.reset_index(drop=True)
```

```images_test = tb.LocalDataset.from_pd(df=image_df_test, key_field="index")
labels_test = tb.LocalDataset.from_pd(df=label_df_test, key_field="index")
```

```imaginal_fields = ["images"]
features_test = images_test.get_model_inputs(imaginal_fields=imaginal_fields)
targets_test = labels_test.get_model_labels(label_field="labels")
```

### Using TurboML Client to request gRPC server [Permalink for this section](https://docs.turboml.com/byo_models/resnet_example/\#using-turboml-client-to-request-grpc-server)

```grpc_model = tb.GRPCClient(
    server_url="0.0.0.0:50021",
    connection_timeout=10000,
    max_request_time=10000,
    max_retries=1,
)
```

```model_trained = grpc_model.learn(features_test, targets_test)
```

```outputs = model_trained.predict(features_test)
```

```outputs  # {class,probability}
```

Last updated on January 24, 2025

[TF-IDF Example](https://docs.turboml.com/byo_models/tfidf_example/ "TF-IDF Example") [OCR](https://docs.turboml.com/byo_models/ocr_example/ "OCR")
</page_content>

# OVR (OnevsRestClassifier)
@ TurboML - page_link: https://docs.turboml.com/pipeline_components/ovr/
<page_content>
Pipeline Components

One-Vs-Rest

One-vs-the-rest (OvR) multiclass strategy.

This strategy consists in fitting one binary classifier per class. The computational complexity for both learning and predicting grows linearly with the number of classes. Not recommended for very large number of classes.

## Parameters [Permalink for this section](https://docs.turboml.com/pipeline_components/ovr/\#parameters)

- **base\_model**( `Model`) → A binary classifier, although a multi-class classifier will work too.

- **n\_classes**( `int`) → The number of classes for the classifier.


## Example Usage [Permalink for this section](https://docs.turboml.com/pipeline_components/ovr/\#example-usage)

```import turboml as tb
htc_model = tb.HoeffdingTreeClassifier(n_classes=2)
ovr_model = tb.OVR(n_classes = 7, base_model = htc_model)
```

Last updated on January 24, 2025

[LLAMA Embedding](https://docs.turboml.com/pipeline_components/llamaembedding/ "LLAMA Embedding") [PreProcessors](https://docs.turboml.com/pipeline_components/preprocessors/ "PreProcessors")
</page_content>

# Hoeffding Tree Classifier
@ TurboML - page_link: https://docs.turboml.com/classification/hoeffdingtreeclassifier/
<page_content>
Classification

Hoeffding Tree Classifier

**Hoeffding Tree** or Very Fast Decision Tree classifier.
A Hoeffding Tree[1](https://docs.turboml.com/classification/hoeffdingtreeclassifier/#user-content-fn-1) is an incremental, anytime decision tree induction algorithm that is capable of learning from massive data streams, assuming that the distribution generating examples does not change over time. Hoeffding trees exploit the fact that a small sample can often be enough to choose an optimal splitting attribute. This idea is supported mathematically by the Hoeffding bound, which quantifies the number of observations (in our case, examples) needed to estimate some statistics within a prescribed precision (in our case, the goodness of an attribute).

A theoretically appealing feature of Hoeffding Trees not shared by other incremental decision tree learners is that it has sound guarantees of performance. Using the Hoeffding bound one can show that its output is asymptotically nearly identical to that of a non-incremental learner using infinitely many examples. Implementation based on MOA[2](https://docs.turboml.com/classification/hoeffdingtreeclassifier/#user-content-fn-2).

## Parameters [Permalink for this section](https://docs.turboml.com/classification/hoeffdingtreeclassifier/\#parameters)

- **n\_classes**( `int`) → The number of classes for the classifier.

- **grace\_period**( `int`, Default: `200`) → Number of instances a leaf should observe between split attempts.

- **split\_method**( `str`, Default: `gini`) → Split criterion to use.
  - `gini` \- Gini
  - `info_gain` \- Information Gain
  - `hellinger` \- Helinger Distance
- **delta**( `float`, Default: `1e-07`) → Significance level to calculate the Hoeffding bound. The significance level is given by 1 - delta. Values closer to zero imply longer split decision delays.

- **tau**( `float`,Default: `0.05`) → Threshold below which a split will be forced to break ties.

- **leaf\_pred\_method**( `str`,Default: `mc`) → Prediction mechanism used at leafs. For now only Majority Class ( `mc`) is supported.


## Example Usage [Permalink for this section](https://docs.turboml.com/classification/hoeffdingtreeclassifier/\#example-usage)

We can create an instance of the HoeffdingTreeClassifier model like this.

```import turboml as tb
htc_model = tb.HoeffdingTreeClassifier(n_classes=2)
```

## Footnotes [Permalink for this section](https://docs.turboml.com/classification/hoeffdingtreeclassifier/\#footnote-label)

1. G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams. In KDD’01, pages 97@106, San Francisco, CA, 2001. ACM Press. [↩](https://docs.turboml.com/classification/hoeffdingtreeclassifier/#user-content-fnref-1)

2. Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer. MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604, 2010. [↩](https://docs.turboml.com/classification/hoeffdingtreeclassifier/#user-content-fnref-2)


Last updated on January 24, 2025

[Multinomial Naive Bayes](https://docs.turboml.com/classification/multinomialnb/ "Multinomial Naive Bayes") [SGT Classifier](https://docs.turboml.com/classification/sgtclassifier/ "SGT Classifier")
</page_content>

# AdaBoostClassifier
@ TurboML - page_link: https://docs.turboml.com/ensembles/adaboostclassifer/
<page_content>
Ensembles

AdaBoost Classifier

An AdaBoost [1](https://docs.turboml.com/ensembles/adaboostclassifer/#user-content-fn-1) classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

For each incoming observation, each model's learn\_one method is called k times where k is sampled from a Poisson distribution of parameter lambda. The lambda parameter is updated when the weaks learners fit successively the same observation.

## Parameters [Permalink for this section](https://docs.turboml.com/ensembles/adaboostclassifer/\#parameters)

- **base\_model**( `Model`) → The classifier to boost.

- **n\_models**(Default: `10`) → The number of models in the ensemble.

- **n\_classes**( `int`) → The number of classes for the classifier.

- **seed**( `int`, Default: `0`) → Random number generator seed for reproducibility.


## Example Usage [Permalink for this section](https://docs.turboml.com/ensembles/adaboostclassifer/\#example-usage)

We can create an instance and deploy AdaBoostClassifier model like this.

```import turboml as tb
model = tb.AdaBoostClassifier(n_classes=2,base_model = tb.HoeffdingTreeClassifier(n_classes=2))
```

## Footnotes [Permalink for this section](https://docs.turboml.com/ensembles/adaboostclassifer/\#footnote-label)

1. Y. Freund, R. Schapire, “A Decision-Theoretic Generalization of on-Line Learning and an Application to Boosting”, 1995. [↩](https://docs.turboml.com/ensembles/adaboostclassifer/#user-content-fnref-1)


Last updated on January 24, 2025

[ONNX](https://docs.turboml.com/general_purpose/onnx/ "ONNX") [Bandit Model Selection](https://docs.turboml.com/ensembles/banditmodelselection/ "Bandit Model Selection")
</page_content>

# Random Sampler
@ TurboML - page_link: https://docs.turboml.com/pipeline_components/randomsampler/
<page_content>
Pipeline Components

Random Sampler

Random sampling by mixing under-sampling and over-sampling.

This is a wrapper for classifiers. It will train the provided classifier by both under-sampling and over-sampling the stream of given observations so that the class distribution seen by the classifier follows a given desired distribution.

## Parameters [Permalink for this section](https://docs.turboml.com/pipeline_components/randomsampler/\#parameters)

- **classifier**( `Model`) \- Classifier Model.

- **desired\_dist**( `dict`) → The desired class distribution. The keys are the classes whilst the values are the desired class percentages. The values must sum up to 1. If set to None, then the observations will be sampled uniformly at random, which is stricly equivalent to using ensemble.BaggingClassifier.

- **sampling\_rate**( `int`, Default: `1.0`) → The desired ratio of data to sample.

- **seed**( `int` \| `None`, Default: `None`) → Random seed for reproducibility.


## Example Usage [Permalink for this section](https://docs.turboml.com/pipeline_components/randomsampler/\#example-usage)

We can create an instance of the Random Sampler like this.

```import turboml as tb
htc_model = tb.HoeffdingTreeClassifier(n_classes=2)
sampler_model = tb.RandomSampler(base_model = htc_model)
```

Last updated on January 24, 2025

[Random Projection Embedding](https://docs.turboml.com/pipeline_components/randomprojectionembedding/ "Random Projection Embedding")
</page_content>

# FFM Classifier
@ TurboML - page_link: https://docs.turboml.com/classification/ffmclassifier/
<page_content>
Classification

FFM Classifier

**Field-aware Factorization Machine** [1](https://docs.turboml.com/classification/ffmclassifier/#user-content-fn-1) for binary classification.

The model equation is defined by:
Where is the latent vector corresponding to feature for field, and is the latent vector corresponding to feature for field.
`$$ \sum_{f1=1}^{F} \sum_{f2=f1+1}^{F} \mathbf{w_{i1}} \cdot \mathbf{w_{i2}}, \text{where } i1 = \Phi(v_{f1}, f1, f2), \quad i2 = \Phi(v_{f2}, f2, f1), $$`
Our implementation automatically applies MinMax scaling to the inputs, use normal distribution for latent initialization and logarithm loss for optimization.

## Parameters [Permalink for this section](https://docs.turboml.com/classification/ffmclassifier/\#parameters)

- **n\_factors**( `int`, Default: `10`) → Dimensionality of the factorization or number of latent factors.

- **l1\_weight**( `int`, Default: `0.0`) → Amount of L1 regularization used to push weights towards 0.

- **l2\_weight**( `int`, Default: `0.0`) → Amount of L2 regularization used to push weights towards 0.

- **l1\_latent**( `int`, Default: `0.0`) → Amount of L1 regularization used to push latent weights towards 0.

- **l2\_latent**( `int`, Default: `0.0`) → Amount of L2 regularization used to push latent weights towards 0.

- **intercept**( `int`, Default: `0.0`) → Initial intercept value.

- **intercept\_lr**( `float`, Default: `0.01`) → Learning rate scheduler used for updating the intercept. No intercept will be used if this is set to 0.

- **clip\_gradient**(Default: `1000000000000.0`) → Clips the absolute value of each gradient value.


## Example Usage [Permalink for this section](https://docs.turboml.com/classification/ffmclassifier/\#example-usage)

We can create an instance of the FFM model like this.

```import turboml as tb
ffm_model = tb.FFMClassifier()
```

## Footnotes [Permalink for this section](https://docs.turboml.com/classification/ffmclassifier/\#footnote-label)

1. Juan, Y., Zhuang, Y., Chin, W.S. and Lin, C.J., 2016, September. Field-aware factorization machines for CTR prediction. In Proceedings of the 10th ACM Conference on Recommender Systems (pp. 43-50). [↩](https://docs.turboml.com/classification/ffmclassifier/#user-content-fnref-1)


Last updated on January 24, 2025

[AMF Classifier](https://docs.turboml.com/classification/amfclassifier/ "AMF Classifier") [Gaussian Naive Bayes](https://docs.turboml.com/classification/gaussiannb/ "Gaussian Naive Bayes")
</page_content>

# SNARIMAX
@ TurboML - page_link: https://docs.turboml.com/forecasting/snarimax/
<page_content>
Forecasting

SNARIMAX


**SNARIMAX** stands for **S** easonal **N** on-linear **A** uto **R** egressive **I** ntegrated **M** oving- **A** verage with e **X** ogenous inputs model.

This model generalizes many established time series models in a single interface that can be trained online. It assumes that the provided training data is ordered in time and is uniformly spaced. It is made up of the following components:

- S (Seasonal)

- N (Non-linear): Any online regression model can be used, not necessarily a linear regression as is done in textbooks.

- AR (Autoregressive): Lags of the target variable are used as features.

- I (Integrated): The model can be fitted on a differenced version of a time series. In this context, integration is the reverse of differencing.

- MA (Moving average): Lags of the errors are used as features.

- X (Exogenous): Users can provide additional features. Care has to be taken to include features that will be available both at training and prediction time.


Each of these components can be switched on and off by specifying the appropriate parameters. Classical time series models such as `AR`, `MA`, `ARMA`, and `ARIMA` can thus be seen as special parametrizations of the SNARIMAX model.

This model is tailored for time series that are homoskedastic. In other words, it might not work well if the variance of the time series varies widely along time.

## Parameters [Permalink for this section](https://docs.turboml.com/forecasting/snarimax/\#parameters)

- **p**( `int`) → Order of the autoregressive part. This is the number of past target values that will be included as features.

- **d**( `int`) → Differencing order.

- **q**( `int`) → Order of the moving average part. This is the number of past error terms that will be included as features.

- **m**( `int`, Default: `1`) → Season length used for extracting seasonal features. If you believe your data has a seasonal pattern, then set this accordingly. For instance, if the data seems to exhibit a yearly seasonality, and that your data is spaced by month, then you should set this to 12. Note that for this parameter to have any impact you should also set at least one of the `p`, `d`, and `q` parameters.

- **sp**( `int`, Default: `0`) → Seasonal order of the autoregressive part. This is the number of past target values that will be included as features.

- **sd**( `int`, Default: `0`) → Seasonal differencing order.

- **sq**( `int`, Default: `0`) → Seasonal order of the moving average part. This is the number of past error terms that will be included as features.

- **base\_model**( `Model`) → The online regression model to use.


## Example Usage [Permalink for this section](https://docs.turboml.com/forecasting/snarimax/\#example-usage)

We can create an instance and deploy SNARIMAX model like this.

```import turboml as tb
snarimax_model = tb.SNARIMAX(p = 12, q = 12, m = 12, sd = 1, base_model = tb.HoeffdingTreeRegressor())
```

Last updated on January 24, 2025

[SGT Classifier](https://docs.turboml.com/classification/sgtclassifier/ "SGT Classifier") [Adaptive XGBoost](https://docs.turboml.com/general_purpose/adaptivexgboost/ "Adaptive XGBoost")
</page_content>

# LLM Embeddings
@ TurboML - page_link: https://docs.turboml.com/llms/llm_embedding/
<page_content>
LLMs

LLM Embeddings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/llm_embedding.ipynb)

One of the most important ways to model NLP tasks is to use pre-trained language model embeddings. This notebook covers how to download pre-trained models, use them to get text embeddings and build ML models on top of these embeddings using TurboML. We'll demonstrate this on a SMS Spam classification use-case.

```import turboml as tb
```

## The Dataset [Permalink for this section](https://docs.turboml.com/llms/llm_embedding/\#the-dataset)

We choose the standard SMS Spam dataset for this example

```!pip install river
```

```import pandas as pd
from river import datasets

dataset = datasets.SMSSpam()
dataset
```

```dict_list_x = []
dict_list_y = []
for x, y in dataset:
    dict_list_x.append(x)
    dict_list_y.append({"label": float(y)})
```

```df_features = pd.DataFrame.from_dict(dict_list_x).reset_index()
df_labels = pd.DataFrame.from_dict(dict_list_y).reset_index()
```

```df_features
```

```df_labels
```

```features = tb.OnlineDataset.from_pd(
    df=df_features, key_field="index", id="sms_spam_feat", load_if_exists=True
)
labels = tb.OnlineDataset.from_pd(
    df=df_labels, key_field="index", id="sms_spam_labels", load_if_exists=True
)
```

```model_features = features.get_model_inputs(textual_fields=["body"])
model_label = labels.get_model_labels(label_field="label")
```

## Downloading pre-trained models [Permalink for this section](https://docs.turboml.com/llms/llm_embedding/\#downloading-pre-trained-models)

Huggingface Hub ( [https://huggingface.co/models (opens in a new tab)](https://huggingface.co/models)) is one of the largest collection of pre-trained language models. It also has native intergrations with the GGUF format ( [https://huggingface.co/docs/hub/en/gguf (opens in a new tab)](https://huggingface.co/docs/hub/en/gguf)). This format is quickly becoming the standard for saving and loading models, and popular open-source projects like llama.cpp and GPT4All use this format. TurboML also uses the GGUF format to load pre-trained models. Here's how you can specify a model from Huggingface Hub, and TurboML will download and convert this in the right format.

We also support quantization of the model for conversion. The supported options are "f32", "f16", "bf16", "q8\_0", "auto", where "f32" is for float32, "f16" for float16, "bf16" for bfloat16, "q8\_0" for Q8\_0, "auto" for the highest-fidelity 16-bit float type depending on the first loaded tensor type. "auto" is the default option.

For this notebook, we'll use the [https://huggingface.co/BAAI/bge-small-en-v1.5 (opens in a new tab)](https://huggingface.co/BAAI/bge-small-en-v1.5) model, with "f16" quantization.

```gguf_model = tb.llm.acquire_hf_model_as_gguf("BAAI/bge-small-en-v1.5", "f16")
gguf_model
```

Once we have converted the pre-trained model, we can now use this to generate embeddings. Here's how

```embedding_model = tb.LLAMAEmbedding(gguf_model_id=gguf_model)
deployed_model = embedding_model.deploy(
    "bert_embedding", input=model_features, labels=model_label
)
```

```outputs = deployed_model.get_outputs()
embedding = outputs[0].get("record").embeddings
print(
    "Length of the embedding vector is:",
    len(embedding),
    ". The first 5 values are:",
    embedding[:5],
)
```

But embeddings directly don't solve our use-case! We ultimately need a classification model for spam detection. We can build a pre-processor that converts all our text data into numerical embeddings, and then these numerical values can be passed to a classifier model.

```model = tb.LlamaCppPreProcessor(base_model=tb.SGTClassifier(), gguf_model_id=gguf_model)
```

```deployed_model = model.deploy(
    "bert_sgt_classifier", input=model_features, labels=model_label
)
```

```outputs = deployed_model.get_outputs()
outputs[0]
```

Last updated on January 24, 2025

[Custom Evaluation Metric](https://docs.turboml.com/post_deployment_ml/custom_metric/ "Custom Evaluation Metric") [Image Embeddings](https://docs.turboml.com/llms/image_embeddings/ "Image Embeddings")
</page_content>

# LLAMA Embedding
@ TurboML - page_link: https://docs.turboml.com/pipeline_components/llamaembedding/
<page_content>
Pipeline Components

LLAMA Embedding

Use the GGUF format to load pre-trained language models. Invoke them on the textual features in the input to get embeddings for them.

## Parameters [Permalink for this section](https://docs.turboml.com/pipeline_components/llamaembedding/\#parameters)

- **gguf\_model\_id**( `List[int]`) → A model id issued by `tb.acquire_hf_model_as_gguf`.

- **max\_tokens\_per\_input**( `int`) → The maximum number of tokens to consider in the input text. Tokens beyond this limit will be truncated. Default is 512.


## Example Usage [Permalink for this section](https://docs.turboml.com/pipeline_components/llamaembedding/\#example-usage)

We can create an instance and deploy LLAMAEmbedding model like this.

```import turboml as tb
embedding = tb.LLAMAEmbedding(gguf_model_id=tb.acquire_hf_model_as_gguf("BAAI/bge-small-en-v1.5", "f16"), max_tokens_per_input=512)
```

Last updated on January 24, 2025

[Heterogeneous AdaBoost Classifier](https://docs.turboml.com/ensembles/heteroadaboostclassifer/ "Heterogeneous AdaBoost Classifier") [One-Vs-Rest](https://docs.turboml.com/pipeline_components/ovr/ "One-Vs-Rest")
</page_content>

# Model Explanations using iXAI
@ TurboML - page_link: https://docs.turboml.com/post_deployment_ml/model_explanations/
<page_content>
Post-Deployment ML

Model Explanations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/model_explanations.ipynb)

The `iXAI` module can be used in combination with TurboML to provide incremental explanations for the models being trained.

```import turboml as tb
```

We start by importing the `ixai` package and relevant datasets from `river`.

```!pip install river git+https://github.com/mmschlk/iXAI
```

```import pandas as pd
from ixai.explainer import IncrementalPFI
from river.metrics import Accuracy
from river.utils import Rolling
from river.datasets.synth import Agrawal
from river.datasets.synth import ConceptDriftStream
```

The sample size for the model to train on is defined.

Also, we initialize a concept drift data stream using the `Agrawal` synthetic dataset from `river`.

```n_samples = 150_000
stream = Agrawal(classification_function=1, seed=42)
drift_stream = Agrawal(classification_function=2, seed=42)
stream = ConceptDriftStream(
    stream,
    drift_stream,
    position=int(n_samples * 0.5),
    width=int(n_samples * 0.1),
    seed=42,
)
```

```feature_names = list([x_0 for x_0, _ in stream.take(1)][0].keys())
```

A batch DataFrame is constructed from the stream defined above to train our model.

```features_list = []
labels_list = []

for features, label in stream:
    if len(features_list) == n_samples:
        break
    features_list.append(features)
    labels_list.append(label)

features_df = pd.DataFrame(features_list).reset_index()
labels_df = pd.DataFrame(labels_list, columns=["label"]).reset_index()
```

```numerical_fields = feature_names
```

We use the `LocalDataset` class provided by TurboML to convert the DataFrame into a compatible dataset.

As part of defining the dataset, we specify the column to be used for primary keys.

Then, we get the relevant features from our dataset as defined by the `numerical_fields` list.

```dataset_full = tb.LocalDataset.from_pd(df=features_df, key_field="index")
labels_full = tb.LocalDataset.from_pd(df=labels_df, key_field="index")
```

```features = dataset_full.get_model_inputs(numerical_fields=numerical_fields)
label = labels_full.get_model_labels(label_field="label")
```

We will be using and training the `Hoeffding Tree Classifier` for this task.

```model = tb.HoeffdingTreeClassifier(n_classes=2)
```

```model_learned = model.learn(features, label)
```

Once the model has finished training, we get ready to deploy it so that it can be used for prediction.

To begin with, we re-define our dataset to now support streaming data, and get the relevant features as before.

```dataset_full = dataset_full.to_online(
    id="agrawal_model_explaination", load_if_exists=True
)
labels_full = labels_full.to_online(id="labels_model_explaination", load_if_exists=True)
```

```features = dataset_full.get_model_inputs(numerical_fields=numerical_fields)
label = labels_full.get_model_labels(label_field="label")
```

We specify that the model being deployed is to be used only for prediction using the `predict_only` parameter of the `deploy()` method.

```deployed_model = model_learned.deploy(
    name="demo_model_ixai", input=features, labels=label, predict_only=True
)
```

Now, the `get_endpoints()` method is used to fetch an endpoint to which inference requests will be sent.

```model_endpoints = deployed_model.get_endpoints()
```

We define `model_function` as a wrapper for the inference requests being sent to the deployed model such that the outputs are compatible with `iXAI`'s explanations API.

```import requests


def model_function(x):
    resp = requests.post(
        model_endpoints[0], json=x, headers=tb.common.api.headers
    ).json()
    resp["output"] = resp.pop("predicted_class")
    return resp
```

We instantiate the `IncrementalPFI` class from `iXAI` with our prediction function defined above, along with the relevant fields from the dataset and the loss function to calculate the feature importance values.

```incremental_pfi = IncrementalPFI(
    model_function=model_function,
    loss_function=Accuracy(),
    feature_names=numerical_fields,
    smoothing_alpha=0.001,
    n_inner_samples=5,
)
```

Finally, we loop through the stream for the first 10000 samples, updating our metric and `incremental_pfi` after each encountered sample.

At every 1000th step, we print out the metric with the feature importance values.

```training_metric = Rolling(Accuracy(), window_size=1000)
for n, (x_i, y_i) in enumerate(stream, start=1):
    if n == 10000:
        break

    incremental_pfi.explain_one(x_i, y_i)

    if n % 1000 == 0:
        print(
            f"{n}: Accuracy: {training_metric.get()} PFI: {incremental_pfi.importance_values}"
        )
```

Last updated on January 24, 2025

[Drift](https://docs.turboml.com/post_deployment_ml/drift/ "Drift") [Custom Evaluation Metric](https://docs.turboml.com/post_deployment_ml/custom_metric/ "Custom Evaluation Metric")
</page_content>

# ONNX tutorial with PyTorch
@ TurboML - page_link: https://docs.turboml.com/byo_models/onnx_pytorch/
<page_content>
Bring Your Own Models

ONNX - Pytorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/onnx_pytorch.ipynb)

```import turboml as tb
```

```!pip install onnx==1.14.1
```

## PyTorch - Standard Model Training [Permalink for this section](https://docs.turboml.com/byo_models/onnx_pytorch/\#pytorch---standard-model-training)

The following blocks of code define a standard pytorch dataloader, and model training code. This is completely independent of TurboML.

```import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import io
```

```transactions = tb.datasets.FraudDetectionDatasetFeatures()
labels = tb.datasets.FraudDetectionDatasetLabels()
```

```joined_df = pd.merge(transactions.df, labels.df, on="transactionID", how="right")
joined_df
```

```X = joined_df.drop("is_fraud", axis=1)
numerical_fields = [\
    "transactionAmount",\
    "localHour",\
    "isProxyIP",\
    "digitalItemCount",\
    "physicalItemCount",\
]

feats = X[numerical_fields]
targets = joined_df["is_fraud"].astype(int)
```

```class TransactionsDataset(Dataset):
    def __init__(self, feats, targets):
        self.feats = feats
        self.targets = targets

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.feats.iloc[idx], dtype=torch.float),
            "y": torch.tensor(self.targets.iloc[idx], dtype=torch.float),
        }
```

```class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(
            64, 2
        )  # Output size is 2 for binary classification (fraud or not fraud)

    def forward(self, x):
        # x --> (batch_size, input_size)
        x = torch.relu(self.fc1(x))
        # x --> (batch_size, 64)
        x = torch.relu(self.fc2(x))
        # x --> (batch_size, 64)
        x = self.fc3(x)
        # x --> (batch_size, 2)
        return x
```

```model = NeuralNet(input_size=feats.shape[1])
model
```

```ds = TransactionsDataset(feats, targets)
ds[0]
```

```train_size = int(0.8 * len(ds))
test_size = len(ds) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size])
```

```batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

```criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for data in train_loader:
        inputs = data["x"].float().to(device)
        tars = data["y"].long().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, tars)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

```model.eval()  # Set the model to evaluation mode
total_correct = 0
total_samples = 0

with torch.no_grad():
    for data in test_loader:
        inputs = data["x"].float()
        tars = data["y"].long()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += tars.size(0)
        total_correct += (predicted == tars).sum().item()

accuracy = total_correct / total_samples
print("Accuracy:", accuracy)
```

## Export model to ONNX format [Permalink for this section](https://docs.turboml.com/byo_models/onnx_pytorch/\#export-model-to-onnx-format)

Exporting a model to ONNX format depends on the framework. Tutorials for different frameworks can be found at [https://github.com/onnx/tutorials#converting-to-onnx-format (opens in a new tab)](https://github.com/onnx/tutorials#converting-to-onnx-format)

```model.eval()
sample_input = torch.randn(1, len(numerical_fields))
buffer = io.BytesIO()
torch.onnx.export(model, sample_input, buffer, export_params=True, verbose=True)
onnx_model_string = buffer.getvalue()
```

## Create an ONNX model with TurboML [Permalink for this section](https://docs.turboml.com/byo_models/onnx_pytorch/\#create-an-onnx-model-with-turboml)

Now that we've converted the model to ONNX format, we can deploy it with TurboML.

```transactions = transactions.to_online(id="transactions", load_if_exists=True)
labels = labels.to_online(id="transaction_labels", load_if_exists=True)

features = transactions.get_model_inputs(numerical_fields=numerical_fields)
label = labels.get_model_labels(label_field="is_fraud")
```

```tb.set_onnx_model("torchmodel", onnx_model_string)
onnx_model = tb.ONNX(model_save_name="torchmodel")
```

```deployed_model = onnx_model.deploy("onnx_model_torch", input=features, labels=label)
```

```deployed_model.add_metric("WindowedAUC")
```

```model_auc_scores = deployed_model.get_evaluation("WindowedAUC")
plt.plot([model_auc_score.metric for model_auc_score in model_auc_scores])
```

Last updated on January 24, 2025

[Stream Dataset Online](https://docs.turboml.com/general_examples/stream_dataset_online/ "Stream Dataset Online") [ONNX - Scikit-Learn](https://docs.turboml.com/byo_models/onnx_sklearn/ "ONNX - Scikit-Learn")
</page_content>

# Half-Space Trees (HST)
@ TurboML - page_link: https://docs.turboml.com/anomaly_detection/hst/
<page_content>
Anomaly Detection

Half Space Trees

Half-space trees are an online variant of isolation forests. They work well when anomalies are spread out. However, they do not work well if anomalies are packed together in windows.

By default, we apply MinMax scaling to the inputs to ensure each feature has values that are comprised between `0` and `1`.

Note that high scores indicate anomalies, whereas low scores indicate normal observations.

## Parameters [Permalink for this section](https://docs.turboml.com/anomaly_detection/hst/\#parameters)

- **n\_trees**(Default: `20`) → Number of trees to use.

- **height**(Default: `12`) → Height of each tree. Note that a tree of height h is made up of h + 1 levels and therefore contains 2 \*\* (h + 1) - 1 nodes.

- **window\_size**(Default: `50`) → Number of observations to use for calculating the mass at each node in each tree.


## Example Usage [Permalink for this section](https://docs.turboml.com/anomaly_detection/hst/\#example-usage)

```import turboml as tb
hst_model = tb.HST()
```

Last updated on January 24, 2025

[Image Input](https://docs.turboml.com/non_numeric_inputs/image_input/ "Image Input") [MStream](https://docs.turboml.com/anomaly_detection/mstream/ "MStream")
</page_content>

# SGT Regressor
@ TurboML - page_link: https://docs.turboml.com/regression/sgtregressor/
<page_content>
Regression

SGT Regressor

Stochastic Gradient Tree for regression.

Incremental decision tree regressor that minimizes the mean square error to guide its growth.

Stochastic Gradient Trees (SGT) directly minimize a loss function to guide tree growth and update their predictions. Thus, they differ from other incrementally tree learners that do not directly optimize the loss, but a data impurity-related heuristic.

## Parameters [Permalink for this section](https://docs.turboml.com/regression/sgtregressor/\#parameters)

- **delta**( `float`, Default: `1e-07`) → Define the significance level of the F-tests performed to decide upon creating splits or updating predictions.

- **grace\_period**( `int`, Default: `200`) → Interval between split attempts or prediction updates.

- **lambda\_**( `float`, Default: `0.1`) → Positive float value used to impose a penalty over the tree's predictions and force them to become smaller. The greater the lambda value, the more constrained are the predictions.

- **gamma**( `float`, Default: `1.0`) → Positive float value used to impose a penalty over the tree's splits and force them to be avoided when possible. The greater the gamma value, the smaller the chance of a split occurring.


## Example Usage [Permalink for this section](https://docs.turboml.com/regression/sgtregressor/\#example-usage)

We can create an instance of the SGT Regressor model like this.

```import turboml as tb
sgt_model = tb.SGTRegressor()
```

ℹ

This implementation enhances the original proposal[1](https://docs.turboml.com/regression/sgtregressor/#user-content-fn-1) by using an incremental strategy to discretize numerical features dynamically, rather than relying on a calibration set and parameterized number of bins. The strategy used is an adaptation of the Quantization Observer (QO)[2](https://docs.turboml.com/regression/sgtregressor/#user-content-fn-2). Different bin size setting policies are available for selection. They directly related to number of split candidates the tree is going to explore, and thus, how accurate its split decisions are going to be. Besides, the number of stored bins per feature is directly related to the tree's memory usage and runtime.

## Footnotes [Permalink for this section](https://docs.turboml.com/regression/sgtregressor/\#footnote-label)

1. Gouk, H., Pfahringer, B., & Frank, E. (2019, October). Stochastic Gradient Trees. In Asian Conference on Machine Learning (pp. 1094-1109). [↩](https://docs.turboml.com/regression/sgtregressor/#user-content-fnref-1)

2. Mastelini, S.M. and de Leon Ferreira, A.C.P., 2021. Using dynamical quantization to perform split attempts in online tree regressors. Pattern Recognition Letters. [↩](https://docs.turboml.com/regression/sgtregressor/#user-content-fnref-2)


Last updated on January 24, 2025

[Hoeffding Tree Regressor](https://docs.turboml.com/regression/hoeffdingtreeregressor/ "Hoeffding Tree Regressor") [AMF Classifier](https://docs.turboml.com/classification/amfclassifier/ "AMF Classifier")
</page_content>

# Ensembling Custom Python Models in TurboML
@ TurboML - page_link: https://docs.turboml.com/wyo_models/ensemble_python_model/
<page_content>
Write Your Own Models

Ensemble Python Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/ensemble_python_model.ipynb)

TurboML allows you to create custom ensemble models using Python classes, leveraging the flexibility of Python while benefiting from TurboML's performance and scalability. In this notebook, we'll walk through how to create a custom ensemble model using TurboML's PythonEnsembleModel interface.

```import turboml as tb
```

```!pip install river
```

```import pandas as pd
import turboml.common.pytypes as types
import turboml.common.pymodel as model
import logging
from typing import List
import matplotlib.pyplot as plt
```

## Prepare an Evaluation Dataset [Permalink for this section](https://docs.turboml.com/wyo_models/ensemble_python_model/\#prepare-an-evaluation-dataset)

We choose a standard Credit Card Fraud dataset that ships with River to evaluate our models on.

```features = tb.datasets.CreditCardsDatasetFeatures()
labels = tb.datasets.CreditCardsDatasetLabels()

features
```

```features.df.loc[0]
```

## Load Datasets into TurboML [Permalink for this section](https://docs.turboml.com/wyo_models/ensemble_python_model/\#load-datasets-into-turboml)

We'll load the features and labels with our `OnlineDataset` interface.

```features = tb.OnlineDataset.from_local_dataset(
    features, "cc_features", load_if_exists=True
)
labels = tb.OnlineDataset.from_local_dataset(labels, "cc_labels", load_if_exists=True)
```

## Isolate features [Permalink for this section](https://docs.turboml.com/wyo_models/ensemble_python_model/\#isolate-features)

```numerical_cols = features.preview_df.columns.tolist()
numerical_cols.remove("index")
input_features = features.get_model_inputs(numerical_fields=numerical_cols)
label = labels.get_model_labels(label_field="score")
```

## Structure of Ensemble Models [Permalink for this section](https://docs.turboml.com/wyo_models/ensemble_python_model/\#structure-of-ensemble-models)

A custom ensemble model in TurboML must implement three instance methods:

- `init_imports`: Import any external modules used in the class.
- `learn_one`: Receive labeled data for the model to learn from.
- `predict_one`: Receive input features for prediction and output the result.
Here's the general structure:

```class CustomEnsembleModel:
    def __init__(self, base_models: List[types.Model]):
        # Ensure at least one base model is provided
        if not base_models:
            raise ValueError("PythonEnsembleModel requires at least one base model.")
        self.base_models = base_models

    def init_imports(self):
        """
        Import any external symbols/modules used in this class
        """
        pass

    def learn_one(self, input: types.InputData):
        """
        Receives labelled data for the model to learn from
        """
        pass

    def predict_one(self, input: types.InputData, output: types.OutputData):
        """
        Receives input features for a prediction, must pass output to the
        output object
        """
        pass
```

## Example - Creating a Custom Ensemble Model

We'll create a custom ensemble model that averages the predictions of its base models.

```class MyEnsembleModel:
    def __init__(self, base_models: List[model.Model]):
        if not base_models:
            raise ValueError("PythonEnsembleModel requires at least one base model.")
        self.base_models = base_models
        self.logger = logging.getLogger(__name__)

    def init_imports(self):
        pass

    def learn_one(self, input: types.InputData):
        try:
            for model in self.base_models:
                model.learn_one(input)
        except Exception as e:
            self.logger.exception(f"Exception in learn_one: {e}")

    def predict_one(self, input: types.InputData, output: types.OutputData):
        try:
            total_score = 0.0
            for model in self.base_models:
                model_output = model.predict_one(input)
                model_score = model_output.score()
                total_score += model_score
            average_score = total_score / len(self.base_models)
            output.set_score(average_score)
        except Exception as e:
            self.logger.exception(f"Exception in predict_one: {e}")
```

## Set Up the Virtual Environment [Permalink for this section](https://docs.turboml.com/wyo_models/ensemble_python_model/\#set-up-the-virtual-environment)

We'll set up a virtual environment and add our custom ensemble class to it. Since our class requires arguments in the constructor, we'll disable validation when adding it.

```# Set up the virtual environment
venv_name = "my_ensemble_venv"
venv = tb.setup_venv(venv_name, ["river"])

# Add the ensemble class without validation
venv.add_python_class(MyEnsembleModel, do_validate_as_model=False)
```

## Create Base Models

We'll use TurboML's built-in models as base models for our ensemble.

```# Create individual base models
model1 = tb.HoeffdingTreeClassifier(n_classes=2)
model2 = tb.AMFClassifier(n_classes=2)
```

```# Create the PythonEnsembleModel
ensemble_model = tb.PythonEnsembleModel(
    base_models=[model1, model2],
    module_name="",
    class_name="MyEnsembleModel",
    venv_name=venv_name,
)
```

## Deploy the Ensemble Model [Permalink for this section](https://docs.turboml.com/wyo_models/ensemble_python_model/\#deploy-the-ensemble-model)

We'll deploy the ensemble model, providing the input features and labels.

```deployed_ensemble_model = ensemble_model.deploy(
    name="ensemble_model", input=input_features, labels=label
)
```

## Evaluate the Ensemble Model [Permalink for this section](https://docs.turboml.com/wyo_models/ensemble_python_model/\#evaluate-the-ensemble-model)

We'll add a metric to evaluate the model and plot the results.

```# Add a metric to the deployed model
deployed_ensemble_model.add_metric("WindowedRMSE")

# Retrieve the evaluation results
model_rmse_scores = deployed_ensemble_model.get_evaluation("WindowedRMSE")

# Plot the RMSE scores
plt.figure(figsize=(10, 6))
plt.plot([score.metric for score in model_rmse_scores], label="Ensemble Model RMSE")
plt.xlabel("Time Steps")
plt.ylabel("RMSE")
plt.title("Ensemble Model Evaluation")
plt.legend()
plt.show()
```

Last updated on January 24, 2025

[Native Python Model](https://docs.turboml.com/wyo_models/native_python_model/ "Native Python Model") [Batch Python Model](https://docs.turboml.com/wyo_models/batch_python_model/ "Batch Python Model")
</page_content>

# Custom Evaluation Metric
@ TurboML - page_link: https://docs.turboml.com/post_deployment_ml/custom_metric/
<page_content>
Post-Deployment ML

Custom Evaluation Metric

TurboML allows you to define your own aggregate metrics in Python.

```import turboml as tb
```

```from turboml.common import ModelMetricAggregateFunction
import math
import pandas as pd
```

### Model Metric Aggregation function [Permalink for this section](https://docs.turboml.com/post_deployment_ml/custom_metric/\#model-metric-aggregation-function)

Metric aggregate functions are used to add and compute any custom metric over model predictions and labels.

#### Overview of Metric Aggregate Functions [Permalink for this section](https://docs.turboml.com/post_deployment_ml/custom_metric/\#overview-of-metric-aggregate-functions)

A metric aggregate function consists of the following lifecycle methods:

1. `create_state()`: Initializes the aggregation state.
2. `accumulate(state, prediction, label)`: Updates the state based on input values.
3. `retract(state, prediction, label) (optional)`: Reverses the effect of previously accumulated values (useful in sliding windows or similar contexts).
4. `merge_states(state1, state2)`: Merges two states (for distributed computation).
5. `finish(state)`: Computes and returns the final metric value.

### Steps to Define a Metric Aggregate Function [Permalink for this section](https://docs.turboml.com/post_deployment_ml/custom_metric/\#steps-to-define-a-metric-aggregate-function)

**1\. Define a Subclass**

Create a subclass of `ModelMetricAggregateFunction` and override its methods.

**2\. Implement Required Methods**

At a minimum, one needs to implement:

- create\_state
- accumulate
- finish
- merge\_states

### Example: Focal Loss Metric [Permalink for this section](https://docs.turboml.com/post_deployment_ml/custom_metric/\#example-focal-loss-metric)

Here’s an example of a custom focal loss metric function.

```class FocalLoss(ModelMetricAggregateFunction):
    def __init__(self):
        super().__init__()

    def create_state(self):
        """
        Initialize the aggregation state.
        Returns:
            Any: A serializable object representing the initial state of the metric.
            This can be a tuple, dictionary, or any other serializable data structure.
            Note:
                - The serialized size of the state should be less than 8MB to ensure
                  compatibility with distributed systems and to avoid exceeding storage
                  or transmission limits.
                - Ensure the state is lightweight and efficiently encodable for optimal
                  performance.
        """
        return (0.0, 0)

    def _compute_focal_loss(self, prediction, label, gamma=2.0, alpha=0.25):
        if prediction is None or label is None:
            return None
        pt = prediction if label == 1 else 1 - prediction
        pt = max(min(pt, 1 - 1e-6), 1e-6)
        return -alpha * ((1 - pt) ** gamma) * math.log(pt)

    def accumulate(self, state, prediction, label):
        """
        Update the state with a new prediction-target pair.
        Args:
            state (Any): The current aggregation state.
            prediction (float): Predicted value.
            label (float): Ground truth.
        Returns:
            Any: The updated aggregation state, maintaining the same format and requirements as `create_state`.
        """
        loss_sum, weight_sum = state
        focal_loss = self._compute_focal_loss(prediction, label)
        if focal_loss is None:
            return state
        return loss_sum + focal_loss, weight_sum + 1

    def finish(self, state):
        """
        Compute the final metric value.
        Args:
            state (Any): Final state.
        Returns:
            float: The result.
        """
        loss_sum, weight_sum = state
        return 0 if weight_sum == 0 else loss_sum / weight_sum

    def merge_states(self, state1, state2):
        """
        Merge two states (for distributed computations).
        Args:
            state1 (Any): The first aggregation state.
            state2 (Any): The second aggregation state.

        Returns:
            tuple: Merged state, maintaining the same format and requirements as `create_state`.
        """
        loss_sum1, weight_sum1 = state1
        loss_sum2, weight_sum2 = state2
        return loss_sum1 + loss_sum2, weight_sum1 + weight_sum2
```

### Guidelines for Implementation [Permalink for this section](https://docs.turboml.com/post_deployment_ml/custom_metric/\#guidelines-for-implementation)

1. State Management:
   - Ensure the state is serializable and the serialized size of the state should be less than 8MB
2. Edge Cases:
   - Handle cases where inputs might be None.
   - Ensure finish() handles empty states gracefully.

We will create one model to test the metric. Please follow the quickstart doc for details.

```transactions = tb.datasets.FraudDetectionDatasetFeatures().to_online(
    "transactions", load_if_exists=True
)
labels = tb.datasets.FraudDetectionDatasetLabels().to_online(
    "transaction_labels", load_if_exists=True
)
```

```model = tb.HoeffdingTreeClassifier(n_classes=2)
```

```numerical_fields = [\
    "transactionAmount",\
    "localHour",\
]
features = transactions.get_model_inputs(numerical_fields=numerical_fields)
label = labels.get_model_labels(label_field="is_fraud")
```

```deployed_model_hft = model.deploy(name="demo_model_hft", input=features, labels=label)
```

```outputs = deployed_model_hft.get_outputs()
```

We can register a metric and get evaluations

```tb.register_custom_metric("FocalLoss", FocalLoss)
```

```model_scores = deployed_model_hft.get_evaluation("FocalLoss")
model_scores[-1]
```

```import matplotlib.pyplot as plt

plt.plot([model_score.metric for model_score in model_scores])
```

Last updated on January 24, 2025

[Model Explanations](https://docs.turboml.com/post_deployment_ml/model_explanations/ "Model Explanations") [LLM Embeddings](https://docs.turboml.com/llms/llm_embedding/ "LLM Embeddings")
</page_content>

# Adaptive XGBoost
@ TurboML - page_link: https://docs.turboml.com/general_purpose/adaptivexgboost/
<page_content>
General Purpose

Adaptive XGBoost

XGBoost implementation to handle concept drift based on Adaptive XGBoost for Evolving Data Streams[1](https://docs.turboml.com/general_purpose/adaptivexgboost/#user-content-fn-1).

## Parameters [Permalink for this section](https://docs.turboml.com/general_purpose/adaptivexgboost/\#parameters)

- **n\_classes**( `int`) → The `num_class` parameter from XGBoost.

- **learning\_rate**(Default: `0.3`) → The `eta` parameter from XGBoost.

- **max\_depth**(Default: `6`) → The `max_depth` parameter from XGBoost.

- **max\_window\_size**(Default: `1000`) → Max window size for drift detection.

- **min\_window\_size**(Default: `0`) → Min window size for drift detection.

- **max\_buffer**(Default: `5`) → Buffers after which to stop growing and start replacing.

- **pre\_train**(Default: `2`) → Buffers to wait before the first XGBoost training.

- **detect\_drift**(Default: `True`) → If set will use a drift detector (ADWIN).

- **use\_updater**(Default: `True`) → Uses `refresh` updated for XGBoost.

- **trees\_per\_train**(Default: `1`) → The number of trees for each training run.

- **percent\_update\_trees**(Default: `1.0`) → The fraction of boosted rounds to be used for updates.


## Example Usage [Permalink for this section](https://docs.turboml.com/general_purpose/adaptivexgboost/\#example-usage)

We can create an instance and deploy AdaptiveXGBoost model like this.

```import turboml as tb
model = tb.AdaptiveXGBoost(n_classes=2)
```

## Footnotes [Permalink for this section](https://docs.turboml.com/general_purpose/adaptivexgboost/\#footnote-label)

1. J. Montiel, R. Mitchell, E. Frank, B. Pfahringer, T. Abdessalem and A. Bifet [Adaptive XGBoost for Evolving Data Streams (opens in a new tab)](https://arxiv.org/abs/2005.07353) [↩](https://docs.turboml.com/general_purpose/adaptivexgboost/#user-content-fnref-1)


Last updated on January 24, 2025

[SNARIMAX](https://docs.turboml.com/forecasting/snarimax/ "SNARIMAX") [Adaptive LightGBM](https://docs.turboml.com/general_purpose/adaptivelgbm/ "Adaptive LightGBM")
</page_content>

# Random Cut Forest
@ TurboML - page_link: https://docs.turboml.com/anomaly_detection/rcf/
<page_content>
Anomaly Detection

Random Cut Forest

RCF detects anomalous data points within a data set that diverge from otherwise well-structured or patterned data. This algorithm takes a bunch of random data points cuts them into the same number of points and creates trees. If we combine all trees creates a forest of data points to determine that if a particular data point is an anomaly or not.

## Parameters [Permalink for this section](https://docs.turboml.com/anomaly_detection/rcf/\#parameters)

- **time\_decay**(Default: `1/2560`) → Determines how long a sample will remain before being replaced.

- **number\_of\_trees**(Default: `50`) → Number of trees to use.

- **output\_after**(Default: `64`) → The number of points required by stream samplers before results are returned.

- **sample\_size**(Default: `256`) → The sample size used by stream samplers in this forest .


## Example Usage [Permalink for this section](https://docs.turboml.com/anomaly_detection/rcf/\#example-usage)

```import turboml as tb
rcf_model = tb.RCF()
```

Last updated on January 24, 2025

[MStream](https://docs.turboml.com/anomaly_detection/mstream/ "MStream") [AMF Regressor](https://docs.turboml.com/regression/amfregressor/ "AMF Regressor")
</page_content>

# Feature Engineering - Complex Stream Processing
@ TurboML - page_link: https://docs.turboml.com/feature_engineering/advanced/ibis_feature_engineering/
<page_content>
Feature Engineering

Advanced

Ibis Feature Engineering

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/ibis_feature_engineering.ipynb)

With real-time features, there can be situtations where the feature logic cannot be expressed by simple SQL, Aggregates or Scalar Python UDFs. In such scenarios, it may be required to write custom streaming pipelines. This is where TurboML is building on Ibis ( [https://github.com/ibis-project/ibis/ (opens in a new tab)](https://github.com/ibis-project/ibis/)), to expose a DataFrame like API to support complex streaming logic for features. We currently support Apache Flink and RisingWave backends for streaming execution.

```import turboml as tb
```

```import pandas as pd
from turboml.common.sources import (
    FileSource,
    DataSource,
    TimestampFormatConfig,
    Watermark,
    DataDeliveryMode,
    S3Config,
)
from turboml.common.models import BackEnd
import ibis
```

```transactions_df = tb.datasets.FraudDetectionDatasetFeatures().df
labels = tb.datasets.FraudDetectionDatasetLabels().to_online(
    "transaction_labels", load_if_exists=True
)
```

### Add feature definitions [Permalink for this section](https://docs.turboml.com/feature_engineering/advanced/ibis_feature_engineering/\#add-feature-definitions)

To add feature definitions, we have a class from turboml package called **IbisFeatureEngineering**. This allows us to define features.

```fe = tb.IbisFeatureEngineering()
```

Let's upload the data for this demo

```%pip install minio
from minio import Minio

client = Minio(
    "play.min.io",
    access_key="Q3AM3UQ867SPQQA43P2F",
    secret_key="zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG",
    secure=True,
)
bucket_name = "ibis-demo"
found = client.bucket_exists(bucket_name)
```

```if not found:
    client.make_bucket(bucket_name)
    print("Created bucket", bucket_name)
else:
    print("Bucket", bucket_name, "already exists")
```

```import duckdb

con = duckdb.connect()
con.sql("SET s3_region='us-east-1';")
con.sql("SET s3_url_style='path';")
con.sql("SET s3_use_ssl=true;")
con.sql("SET s3_endpoint='play.min.io';")
con.sql("SET s3_access_key_id='Q3AM3UQ867SPQQA43P2F';")
con.sql("SET s3_secret_access_key='zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG';")
```

```con.sql(
    "COPY (SELECT * EXCLUDE(timestamp), TO_TIMESTAMP(CAST(timestamp AS DOUBLE)) AS timestamp FROM transactions_df) TO 's3://ibis-demo/transactions/transactions.parquet' (FORMAT 'parquet');"
)
```

### DataSource [Permalink for this section](https://docs.turboml.com/feature_engineering/advanced/ibis_feature_engineering/\#datasource)

The **DataSource** serves as the foundational entity in the feature engineering workflow. It defines where and how the raw data is accessed for processing. After creating a DataSource, users can register their source configurations to start leveraging them in the pipeline.

#### Type of Delivery Modes [Permalink for this section](https://docs.turboml.com/feature_engineering/advanced/ibis_feature_engineering/\#type-of-delivery-modes)

1. Dynamic:
   - Suitable for real-time or streaming data scenarios.
   - Automatically creates connectors based on the source configuration.
   - The Kafka topic becomes the primary input for feature engineering, ensuring seamless integration with downstream processing pipelines.
2. Static:
   - Designed for batch data sources.
   - RisingWave/Flink reads directly from the source for feature engineering, eliminating the need for an intermediary Kafka topic.

```time_col_config = TimestampFormatConfig(
    format_type=TimestampFormatConfig.FormatType.EpochMillis
)
watermark = Watermark(
    time_col="timestamp", allowed_delay_seconds=60, time_col_config=time_col_config
)
ds1 = DataSource(
    name="transactions_stream",
    key_fields=["transactionID"],
    delivery_mode=DataDeliveryMode.DYNAMIC,
    file_source=FileSource(
        path="transactions",
        format=FileSource.Format.PARQUET,
        s3_config=S3Config(
            bucket="ibis-demo",
            region="us-east-1",
            access_key_id="Q3AM3UQ867SPQQA43P2F",
            secret_access_key="zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG",
            endpoint="https://play.min.io",
        ),
    ),
    watermark=watermark,
)

tb.register_source(ds1)
```

To define features we can fetch the sources and perform operations.

```transactions = fe.get_ibis_table("transactions_stream")
```

```transactions
```

In this example we use one kafka topic (transactions\_stream) to build features using Flink.

We will also use UDF to define custom functions.

```@ibis.udf.scalar.python()
def calculate_frequency_score(digital_count: float, physical_count: float) -> float:
    if digital_count > 10 or physical_count > 10:
        return 0.7  # High item count
    elif digital_count == 0 and physical_count > 0:
        return 0.3  # Physical item-only transaction
    elif digital_count > 0 and physical_count == 0:
        return 0.3  # Digital item-only transaction
    else:
        return 0.1  # Regular transaction
```

We can define features using ibis DSL or SQL

```transactions_with_frequency_score = transactions.select(
    frequency_score=calculate_frequency_score(
        transactions.digitalItemCount, transactions.physicalItemCount
    ),
    transactionID=transactions.transactionID,
    digitalItemCount=transactions.digitalItemCount,
    physicalItemCount=transactions.physicalItemCount,
    transactionAmount=transactions.transactionAmount,
    transactionTime=transactions.transactionTime,
    isProxyIP=transactions.isProxyIP,
)
```

We can preview features locally

```transactions_with_frequency_score.execute().head()
```

After satisfied, we can materialize the features.
It will write the features using flink.

Flink uses a hybrid source to read first from iceberg table and then switches to kafka.

```fe.materialize_features(
    transactions_with_frequency_score,
    "transactions_with_frequency_score",
    "transactionID",
    BackEnd.Flink,
    "transactions_stream",
)
```

We can now train a model using features built using flink

```model = tb.RCF(number_of_trees=50)
```

```numerical_fields = ["frequency_score"]
features = fe.get_model_inputs(
    "transactions_with_frequency_score", numerical_fields=numerical_fields
)
label = labels.get_model_labels(label_field="is_fraud")
```

```deployed_model_rcf = model.deploy(
    name="demo_model_ibis_flink", input=features, labels=label
)
```

```outputs = deployed_model_rcf.get_outputs()
```

```sample_output = outputs[-1]
sample_output
```

```import matplotlib.pyplot as plt

plt.plot([output["record"].score for output in outputs])
```

```model_endpoints = deployed_model_rcf.get_endpoints()
model_endpoints
```

```model_query_datapoint = (
    transactions_df[["transactionID", "digitalItemCount", "physicalItemCount"]]
    .iloc[-1]
    .to_dict()
)
model_query_datapoint
```

```import requests

resp = requests.post(
    model_endpoints[0], json=model_query_datapoint, headers=tb.common.api.headers
)
resp.json()
```

```outputs = deployed_model_rcf.get_inference(transactions_df)
outputs
```

## Risingwave FE [Permalink for this section](https://docs.turboml.com/feature_engineering/advanced/ibis_feature_engineering/\#risingwave-fe)

We can now enrich the earlier built features using flink with features built using RisingWave.

Let's fetch the features from server for the feature group

```transactions_with_frequency_score = fe.get_ibis_table(
    "transactions_with_frequency_score"
)
```

```@ibis.udf.scalar.python()
def detect_fraud(
    transactionAmount: float, transactionTime: int, isProxyIP: float
) -> int:
    # Example logic for flagging fraud:
    # - High transaction amount
    # - Unusual transaction times (e.g., outside of working hours)
    # - Use of proxy IP
    is_high_amount = transactionAmount > 1000  # arbitrary high amount threshold
    is_suspicious_time = (transactionTime < 6) | (
        transactionTime > 22
    )  # non-standard hours
    is_proxy = isProxyIP == 1  # proxy IP flag

    return int(is_high_amount & is_suspicious_time & is_proxy)
```

```fraud_detection_expr = detect_fraud(
    transactions_with_frequency_score.transactionAmount,
    transactions_with_frequency_score.transactionTime,
    transactions_with_frequency_score.isProxyIP,
)
```

```transactions_with_fraud_flag = transactions_with_frequency_score.select(
    transactionAmount=transactions_with_frequency_score.transactionAmount,
    transactionTime=transactions_with_frequency_score.transactionTime,
    isProxyIP=transactions_with_frequency_score.isProxyIP,
    transactionID=transactions_with_frequency_score.tramsactionID,
    digitalItemCount=transactions_with_frequency_score.digitalItemCount,
    physicalItemCount=transactions_with_frequency_score.physicalItemCount,
    frequency_score=transactions_with_frequency_score.frequency_score,
    fraud_flag=fraud_detection_expr,
)
```

```transactions_with_fraud_flag.execute().head()
```

```fe.materialize_features(
    transactions_with_fraud_flag,
    "transactions_with_fraud_flag",
    "transactionID",
    BackEnd.Risingwave,
    "transactions_stream",
)
```

```model = tb.RCF(number_of_trees=50)
```

```numerical_fields = ["frequency_score", "fraud_flag"]
features = fe.get_model_inputs(
    "transactions_with_fraud_flag", numerical_fields=numerical_fields
)
label = labels.get_model_labels(label_field="is_fraud")
```

```deployed_model_rcf = model.deploy(
    name="demo_model_ibis_risingwave", input=features, labels=label
)
```

```outputs = deployed_model_rcf.get_outputs()
```

```sample_output = outputs[-1]
sample_output
```

```import matplotlib.pyplot as plt

plt.plot([output["record"].score for output in outputs])
```

```model_endpoints = deployed_model_rcf.get_endpoints()
model_endpoints
```

```model_query_datapoint = (
    transactions_df[\
        [\
            "transactionID",\
            "digitalItemCount",\
            "physicalItemCount",\
            "transactionAmount",\
            "transactionTime",\
            "isProxyIP",\
        ]\
    ]
    .iloc[-1]
    .to_dict()
)
model_query_datapoint
```

```import requests

resp = requests.post(
    model_endpoints[0], json=model_query_datapoint, headers=tb.common.api.headers
)
resp.json()
```

```outputs = deployed_model_rcf.get_inference(transactions_df)
outputs
```

Last updated on January 24, 2025

[Ibis Quickstart](https://docs.turboml.com/feature_engineering/advanced/ibis_quickstart/ "Ibis Quickstart") [Algorithm Tuning](https://docs.turboml.com/pre_deployment_ml/algorithm_tuning/ "Algorithm Tuning")
</page_content>

# TurboML Ibis Quickstart
@ TurboML - page_link: https://docs.turboml.com/feature_engineering/advanced/ibis_quickstart/
<page_content>
Feature Engineering

Advanced

Ibis Quickstart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/ibis_quickstart.ipynb)

```import turboml as tb
```

```import pandas as pd
import ibis
```

```transactions = tb.datasets.FraudDetectionDatasetFeatures().to_online(
    "ibisqs_transactions", load_if_exists=True
)
labels = tb.datasets.FraudDetectionDatasetLabels().to_online(
    "ibisqs_transaction_labels", load_if_exists=True
)
```

The following cells shows how to define features in ibis. The table parameter in the **create\_ibis\_features** function takes in the ibis expression to be used to prepare the feature.

```table = transactions.to_ibis()
```

```@ibis.udf.scalar.python()
def add_one(x: float) -> float:
    return x + 1
```

```table = table.mutate(updated_transaction_amount=add_one(table.transactionAmount))
```

```agged = table.select(
    total_transaction_amount=table.updated_transaction_amount.sum().over(
        window=ibis.window(preceding=100, following=0, group_by=[table.transactionID]),
        order_by=table.timestamp,
    ),
    transactionID=table.transactionID,
    is_potential_fraud=(
        table.ipCountryCode != table.paymentBillingCountryCode.lower()
    ).ifelse(1, 0),
    ipCountryCode=table.ipCountryCode,
    paymentBillingCountryCode=table.paymentBillingCountryCode,
)
```

```transactions.feature_engineering.create_ibis_features(agged)
```

```transactions.feature_engineering.get_local_features()
```

We need to tell the platform to start computations for all pending features for the given topic. This can be done by calling the **materialize\_ibis\_features** function.

```transactions.feature_engineering.materialize_ibis_features()
```

```model = tb.RCF(number_of_trees=50)
```

```numerical_fields = ["total_transaction_amount", "is_potential_fraud"]
features = transactions.get_model_inputs(numerical_fields=numerical_fields)
label = labels.get_model_labels(label_field="is_fraud")
```

```deployed_model_rcf = model.deploy(name="demo_model_ibis", input=features, labels=label)
```

```outputs = deployed_model_rcf.get_outputs()
```

```len(outputs)
```

```sample_output = outputs[-1]
sample_output
```

```import matplotlib.pyplot as plt

plt.plot([output["record"].score for output in outputs])
```

```model_endpoints = deployed_model_rcf.get_endpoints()
model_endpoints
```

```transactions_df = transactions.preview_df
model_query_datapoint = (
    transactions_df[["transactionID", "ipCountryCode", "paymentBillingCountryCode"]]
    .iloc[-1]
    .to_dict()
)
model_query_datapoint
```

```import requests

resp = requests.post(
    model_endpoints[0], json=model_query_datapoint, headers=tb.common.api.headers
)
resp.json()
```

#### Batch Inference on Models [Permalink for this section](https://docs.turboml.com/feature_engineering/advanced/ibis_quickstart/\#batch-inference-on-models)

While the above method is more suited for individual requests, we can also perform batch inference on the models. We use the **get\_inference** function for this purpose.

```outputs = deployed_model_rcf.get_inference(transactions_df)
outputs
```

Last updated on January 24, 2025

[UDAF](https://docs.turboml.com/feature_engineering/udaf/ "UDAF") [Ibis Feature Engineering](https://docs.turboml.com/feature_engineering/advanced/ibis_feature_engineering/ "Ibis Feature Engineering")
</page_content>

# Performance Improvements
@ TurboML - page_link: https://docs.turboml.com/pre_deployment_ml/performance_improvements/
<page_content>
Pre-Deployment ML

Performance Improvements

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) (opens in a new tab)](https://colab.research.google.com/github/TurboML-Inc/colab-notebooks/blob/main/performance_improvements.ipynb)

In this notebook, we'll cover some examples of how model performance can be improved. The techniques covered are

- Sampling for imbalanced learning
- Bagging
- Boosting
- Continuous Model Selection using Bandits.

```import turboml as tb
```

```import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
```

```transactions = tb.datasets.FraudDetectionDatasetFeatures().to_online(
    id="transactions", load_if_exists=True
)
labels = tb.datasets.FraudDetectionDatasetLabels().to_online(
    id="transaction_labels", load_if_exists=True
)
```

```numerical_fields = [\
    "transactionAmount",\
    "localHour",\
]
categorical_fields = [\
    "digitalItemCount",\
    "physicalItemCount",\
    "isProxyIP",\
]
features = transactions.get_model_inputs(
    numerical_fields=numerical_fields, categorical_fields=categorical_fields
)
label = labels.get_model_labels(label_field="is_fraud")
```

Now that we have our setup ready, let's first see the performance of a base HoeffdingTreeClassfier model.

```htc_model = tb.HoeffdingTreeClassifier(n_classes=2)
```

```deployed_model = htc_model.deploy("htc_classifier", input=features, labels=label)
```

```labels_df = labels.preview_df
```

```outputs = deployed_model.get_outputs()
```

```len(outputs)
```

```output_df = pd.DataFrame(
    {labels.key_field: str(x["record"].key), "class": x["record"].predicted_class}
    for x in outputs
)
joined_df = output_df.merge(labels_df, how="inner", on="transactionID")

true_labels = joined_df["is_fraud"]
real_outputs = joined_df["class"]
joined_df
```

```roc_auc_score(true_labels, real_outputs)
```

Not bad. But can we improve it further? We haven't yet used the fact that the dataset is highly skewed.

## Sampling for Imbalanced Learning [Permalink for this section](https://docs.turboml.com/pre_deployment_ml/performance_improvements/\#sampling-for-imbalanced-learning)

```sampler_model = tb.RandomSampler(
    n_classes=2, desired_dist=[0.5, 0.5], sampling_method="under", base_model=htc_model
)
```

```deployed_model = sampler_model.deploy(
    "undersampler_model", input=features, labels=label
)
```

```outputs = deployed_model.get_outputs()
```

```len(outputs)
```

```output_df = pd.DataFrame(
    {labels.key_field: str(x["record"].key), "class": x["record"].predicted_class}
    for x in outputs
)
joined_df = output_df.merge(labels_df, how="inner", on="transactionID")

true_labels = joined_df["is_fraud"]
real_outputs = joined_df["class"]
joined_df
```

```roc_auc_score(true_labels, real_outputs)
```

## Bagging [Permalink for this section](https://docs.turboml.com/pre_deployment_ml/performance_improvements/\#bagging)

```lbc_model = tb.LeveragingBaggingClassifier(n_classes=2, base_model=htc_model)
```

```deployed_model = lbc_model.deploy("lbc_classifier", input=features, labels=label)
```

```outputs = deployed_model.get_outputs()
```

```len(outputs)
```

```output_df = pd.DataFrame(
    {labels.key_field: str(x["record"].key), "class": x["record"].predicted_class}
    for x in outputs
)
joined_df = output_df.merge(labels_df, how="inner", on="transactionID")

true_labels = joined_df["is_fraud"]
real_outputs = joined_df["class"]
joined_df
```

```roc_auc_score(true_labels, real_outputs)
```

## Boosting [Permalink for this section](https://docs.turboml.com/pre_deployment_ml/performance_improvements/\#boosting)

```abc_model = tb.AdaBoostClassifier(n_classes=2, base_model=htc_model)
```

```deployed_model = abc_model.deploy("abc_classifier", input=features, labels=label)
```

```outputs = deployed_model.get_outputs()
```

```len(outputs)
```

```output_df = pd.DataFrame(
    {labels.key_field: str(x["record"].key), "class": x["record"].predicted_class}
    for x in outputs
)
joined_df = output_df.merge(labels_df, how="inner", on="transactionID")

true_labels = joined_df["is_fraud"]
real_outputs = joined_df["class"]
joined_df
```

```roc_auc_score(true_labels, real_outputs)
```

## Continuous Model Selection with Bandits [Permalink for this section](https://docs.turboml.com/pre_deployment_ml/performance_improvements/\#continuous-model-selection-with-bandits)

```bandit_model = tb.BanditModelSelection(base_models=[htc_model, lbc_model, abc_model])
deployed_model = bandit_model.deploy(
    "demo_classifier_bandit", input=features, labels=label
)
```

```outputs = deployed_model.get_outputs()
```

```len(outputs)
```

```output_df = pd.DataFrame(
    {labels.key_field: str(x["record"].key), "class": x["record"].predicted_class}
    for x in outputs
)
joined_df = output_df.merge(labels_df, how="inner", on="transactionID")

true_labels = joined_df["is_fraud"]
real_outputs = joined_df["class"]
joined_df
```

```roc_auc_score(true_labels, real_outputs)
```

Last updated on January 24, 2025

[Hyperparameter Tuning](https://docs.turboml.com/pre_deployment_ml/hyperparameter_tuning/ "Hyperparameter Tuning") [Drift](https://docs.turboml.com/post_deployment_ml/drift/ "Drift")
</page_content>

# Online Neural Network
@ TurboML - page_link: https://docs.turboml.com/general_purpose/onlineneuralnetwork/
<page_content>
General Purpose

Online Neural Networks

Neural Network implementation using Hedge Backpropagation based on Online Deep Learning: Learning Deep Neural Networks on the Fly[1](https://docs.turboml.com/general_purpose/onlineneuralnetwork/#user-content-fn-1).

## Parameters [Permalink for this section](https://docs.turboml.com/general_purpose/onlineneuralnetwork/\#parameters)

- **max\_num\_hidden\_layers**(Default `10`) → The maximum number of hidden layers
- **qtd\_neuron\_hidden\_layer**(Default: `32`) → Hidden dimension of the intermediate neural network layers.
- **n\_classes**( `int`) → Number of classes.
- **b**(Default: `0.99`) → Discounting parameter in the hedge backprop algorithm.
- **n**(Default: `0.01`) → Learning rate parameter in the hedge backprop algorithm.
- **s**(Default: `0.2`) → Smoothing parameter in the hedge backprop algorithm.

## Example Usage [Permalink for this section](https://docs.turboml.com/general_purpose/onlineneuralnetwork/\#example-usage)

We can create an instance and deploy ONN model like this.

```import turboml as tb
model = tb.ONN(n_classes=2)
```

## Footnotes [Permalink for this section](https://docs.turboml.com/general_purpose/onlineneuralnetwork/\#footnote-label)

1. D. Sahoo, Q. Pham, J. Lu and S. Hoi. [Online Deep Learning: Learning Deep Neural Networks on the Fly (opens in a new tab)](https://arxiv.org/abs/1711.03705) [↩](https://docs.turboml.com/general_purpose/onlineneuralnetwork/#user-content-fnref-1)


Last updated on January 24, 2025

[Neural Networks](https://docs.turboml.com/general_purpose/neuralnetwork/ "Neural Networks") [ONNX](https://docs.turboml.com/general_purpose/onnx/ "ONNX")
</page_content>

# __init__.py
-## Location -> root_directory.common
import base64
import logging
from typing import Optional, Callable, Tuple
import itertools

import cloudpickle
from cloudpickle import DEFAULT_PROTOCOL
from pydantic import BaseModel

from turboml.common.datasets import LocalInputs, LocalLabels

from . import namespaces
from . import llm
from .types import PythonModel
from .pytypes import InputData, OutputData
from .datasets import OnlineDataset, LocalDataset
from . import datasets
from .feature_engineering import (
    IbisFeatureEngineering,
    get_timestamp_formats,
    retrieve_features,
    get_features,
)
from .models import (
    AddPythonClassRequest,
    ServiceEndpoints,
    User,
    InputSpec,
    VenvSpec,
    SupervisedAlgorithms,
    UnsupervisedAlgorithms,
    ExternalUdafFunctionSpec,
    UdafFunctionSpec,
    CustomMetric,
)
from .dataloader import (
    upload_df,
    get_proto_msgs,
    get_protobuf_class,
    create_protobuf_from_row_tuple,
    create_protobuf_from_row_dict,
    PROTO_PREFIX_BYTE_LEN,
)
from .api import api
from .concurrent import use_multiprocessing
from .internal import TbPyArrow
from .ml_algs import (
    evaluation_metrics,
    get_default_parameters,
    ml_modelling,
    _resolve_duplicate_columns,
    get_score_for_model,
    RCF,
    HST,
    MStream,
    ONNX,
    HoeffdingTreeClassifier,
    HoeffdingTreeRegressor,
    AMFClassifier,
    AMFRegressor,
    FFMClassifier,
    FFMRegressor,
    SGTClassifier,
    SGTRegressor,
    RandomSampler,
    NNLayer,
    NeuralNetwork,
    Python,
    ONN,
    OVR,
    MultinomialNB,
    GaussianNB,
    AdaptiveXGBoost,
    AdaptiveLGBM,
    MinMaxPreProcessor,
    NormalPreProcessor,
    RobustPreProcessor,
    LlamaCppPreProcessor,
    LlamaTextPreprocess,
    ClipEmbeddingPreprocessor,
    PreProcessor,
    LabelPreProcessor,
    OneHotPreProcessor,
    TargetPreProcessor,
    FrequencyPreProcessor,
    BinaryPreProcessor,
    ImageToNumericPreProcessor,
    SNARIMAX,
    LeveragingBaggingClassifier,
    HeteroLeveragingBaggingClassifier,
    AdaBoostClassifier,
    HeteroAdaBoostClassifier,
    BanditModelSelection,
    ContextualBanditModelSelection,
    RandomProjectionEmbedding,
    LLAMAEmbedding,
    LlamaText,
    ClipEmbedding,
    RestAPIClient,
    EmbeddingModel,
    Model,
    DeployedModel,
    PythonEnsembleModel,
    GRPCClient,
    LocalModel,
)
from .model_comparison import compare_model_metrics
from .sources import DataSource
from .udf import ModelMetricAggregateFunction
from .env import CONFIG

logger = logging.getLogger("turboml.common")


retrieve_model = Model.retrieve_model

__all__ = [
    "init",
    "use_multiprocessing",
    "IbisFeatureEngineering",
    "get_timestamp_formats",
    "upload_df",
    "register_source",
    "register_custom_metric",
    "get_protobuf_class",
    "create_protobuf_from_row_tuple",
    "create_protobuf_from_row_dict",
    "retrieve_features",
    "get_features",
    "set_onnx_model",
    "ml_modelling",
    "setup_venv",
    "get_proto_msgs",
    "ml_algorithms",
    "evaluation_metrics",
    "get_default_parameters",
    "hyperparameter_tuning",
    "algorithm_tuning",
    "compare_model_metrics",
    "login",
    "get_user_info",
    "InputSpec",
    "RCF",
    "HST",
    "MStream",
    "ONNX",
    "HoeffdingTreeClassifier",
    "HoeffdingTreeRegressor",
    "AMFClassifier",
    "AMFRegressor",
    "FFMClassifier",
    "FFMRegressor",
    "SGTClassifier",
    "SGTRegressor",
    "RandomSampler",
    "NNLayer",
    "NeuralNetwork",
    "Python",
    "PythonEnsembleModel",
    "ONN",
    "OVR",
    "MultinomialNB",
    "GaussianNB",
    "AdaptiveXGBoost",
    "AdaptiveLGBM",
    "MinMaxPreProcessor",
    "NormalPreProcessor",
    "RobustPreProcessor",
    "LlamaCppPreProcessor",
    "LlamaTextPreprocess",
    "ClipEmbeddingPreprocessor",
    "PreProcessor",
    "LabelPreProcessor",
    "OneHotPreProcessor",
    "TargetPreProcessor",
    "FrequencyPreProcessor",
    "BinaryPreProcessor",
    "ImageToNumericPreProcessor",
    "SNARIMAX",
    "LeveragingBaggingClassifier",
    "HeteroLeveragingBaggingClassifier",
    "AdaBoostClassifier",
    "HeteroAdaBoostClassifier",
    "BanditModelSelection",
    "ContextualBanditModelSelection",
    "RandomProjectionEmbedding",
    "LLAMAEmbedding",
    "LlamaText",
    "ClipEmbedding",
    "RestAPIClient",
    "EmbeddingModel",
    "retrieve_model",
    "DeployedModel",
    "GRPCClient",
    "namespaces",
    "llm",
    "LocalModel",
    "PROTO_PREFIX_BYTE_LEN",
    "OnlineDataset",
    "LocalDataset",
    "datasets",
]


def ml_algorithms(have_labels: bool) -> list[str]:
    if have_labels:
        algs = [enum.value for enum in SupervisedAlgorithms]
    else:
        algs = [enum.value for enum in UnsupervisedAlgorithms]

    for alg in algs:
        if alg not in globals():
            raise Exception(f"{alg} class doesn't exist")
        elif alg not in __all__:
            raise Exception(f"{alg} class hasn't been exposed")

    return algs


def login(
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
):
    """
    Authenticate with the TurboML server.
    The user should provide either the api_key or username, password.
    If a username and password are provided, the api_key will be retrieved from the server.
    Note that instead of login, you can set the TURBOML_API_KEY env variable as well with your api_key.

    Args:
        api_key, username, password (str)
    Raises:
        Exception: Raises an exception if authentication fails.
    """
    api.login(api_key, username, password)


def init(backend_url: str, api_key: str):
    """
    Initialize SDK and Authenticate with TurboML backend server.

    Args:
        backend_url, api_key (str)
    Raises:
        Exception: Raises an exception if authentication fails.
    """
    CONFIG.set_backend_server(backend_url)
    login(api_key=api_key)
    response = api.get("service/endpoints").json()
    service_endpoints = ServiceEndpoints(**response)
    CONFIG.set_feature_server(service_endpoints.feature_server)
    CONFIG.set_arrow_server(service_endpoints.arrow_server)


def get_user_info() -> User:
    resp = api.get("user").json()
    user_info = User(**resp)
    return user_info


def register_source(source: DataSource):
    if not isinstance(source, DataSource):
        raise TypeError("Expected a DataSource, found %s" % type(source))
    api.post(
        endpoint="register_datasource",
        json=source.model_dump(exclude_none=True),
    )


def register_custom_metric(name: str, cls: type[ModelMetricAggregateFunction]):
    """
    Adds a custom model metric to the system.

    This function registers a custom metric class that must extend
    ModelMetricAggregateFunction. The metric class should implement the
    methods 'create_state', 'accumulate', 'merge_states' and 'finish' to calculate the desired metric (e.g., accuracy, AUC, etc.).

    Args:
        name (str): The name used to register and identify the metric.
        cls (Callable[..., ModelMetricAggregateFunction]): The custom metric class
            that should inherit from ModelMetricAggregateFunction.

    Raises:
        TypeError: If `cls` does not inherit from `ModelMetricAggregateFunction`.
    """
    if not issubclass(cls, ModelMetricAggregateFunction):
        raise TypeError(
            f"{cls.__name__} must be a subclass of ModelMetricAggregateFunction."
        )
    spec = ExternalUdafFunctionSpec(obj=base64.b64encode(cloudpickle.dumps(cls)))

    payload = UdafFunctionSpec(name=name, spec=spec, libraries=[])
    headers = {"Content-Type": "application/json"}
    api.post(endpoint="register_udaf", data=payload.model_dump_json(), headers=headers)
    api.post(
        endpoint="register_custom_metric",
        data=CustomMetric(metric_name=name, metric_spec={}).model_dump_json(),
        headers=headers,
    )


def set_onnx_model(input_model: str, onnx_bytes: bytes) -> None:
    """
    input_model: str
        The model name(without the .onnx extension)
    onnx_bytes: bytes
        The model bytes
    """
    api.post(
        endpoint="onnx_model",
        data={"input_model": input_model},
        files={
            "onnx_bytes": (
                f"{input_model}_bytes",
                onnx_bytes,
                "application/octet-stream",
            )
        },
    )


class Venv(BaseModel):
    name: str

    def add_python_file(self, filepath: str):
        """Add a python source file to the system.

        This function registers input source file in the system.

        Args:
            filepath (str): Path of the Python source file.

        Raises:
            Exception: Raises an exception if registering the source with the system fails.
        """
        with open(filepath, "rb") as file:
            files = {"python_file": (filepath, file, "text/x-python")}
            api.post(f"venv/{self.name}/python_source", files=files)

    def add_python_class(
        self, cls: Callable[..., PythonModel], do_validate_as_model: bool = True
    ):
        """
        Add a Python class to the system.
        By default, validates the class as a model by instantiating and calling the init_imports,
        learn_one, and predict_one methods. However this can be disabled with
        do_validate_as_model=False, for instance when the required libraries are not
        available or cannot be installed in the current environment.
        """
        if not isinstance(cls, type):  # Should be a class
            raise ValueError("Input should be a class")
        if do_validate_as_model:
            try:
                Venv._validate_python_model_class(cls)
            except Exception as e:
                raise ValueError(
                    f"{e!r}. HINT: Set do_validate_as_model=False to skip validation if you believe the class is valid."
                ) from e
        serialized_cls = base64.b64encode(
            cloudpickle.dumps(cls, protocol=DEFAULT_PROTOCOL)
        )
        req = AddPythonClassRequest(obj=serialized_cls, name=cls.__name__)
        headers = {"Content-Type": "application/json"}
        api.post(f"venv/{self.name}/class", data=req.model_dump_json(), headers=headers)

    @staticmethod
    def _validate_python_model_class(model_cls: Callable[..., PythonModel]):
        try:
            model = model_cls()
            logger.debug("Model class instantiated successfully")
            init_imports = getattr(model, "init_imports", None)
            if init_imports is None or not callable(init_imports):
                raise ValueError(
                    "Model class must have an init_imports method to import libraries"
                )
            init_imports()
            logger.debug("Model class imports initialized successfully")
            learn_one = getattr(model, "learn_one", None)
            predict_one = getattr(model, "predict_one", None)
            if learn_one is None or not callable(learn_one):
                raise ValueError("Model class must have a learn_one method")
            if predict_one is None or not callable(predict_one):
                raise ValueError("Model class must have a predict_one method")
            ## TODO: Once we have the Model.get_dimensions interface in place, use it to determine
            ## appropriate input and output shape for the model before passing them to make this check
            ## less brittle.
            model.learn_one(InputData.random())
            logger.debug("Model class learn_one method validated successfully")
            model.predict_one(InputData.random(), OutputData.random())
            logger.debug("Model class predict_one method validated successfully")
        except Exception as e:
            ## NOTE: We have the
            raise ValueError(f"Model class validation failed: {e!r}") from e


def setup_venv(venv_name: str, lib_list: list[str]) -> Venv:
    """Executes `pip install " ".join(lib_list)` in venv_name virtual environment.
    If venv_name doesn't exist, it'll create one.

    Args:
        venv_name (str): Name of virtual environment
        lib_list (list[str]): List of libraries to install. Will be executed as `pip install " ".join(lib_list)`

    Raises:
        Exception: Raises an exception if setting up the venv fails.
    """
    payload = VenvSpec(venv_name=venv_name, lib_list=lib_list)
    api.post("venv", json=payload.model_dump())
    return Venv(name=venv_name)


def _check_hyperparameter_space(
    hyperparameter_space: list[dict[str, list[str]]], model: Model
):
    model_config = model.get_model_config()

    if len(hyperparameter_space) != len(model_config):
        raise Exception(
            "The number of hyperparameter spaces should be equal to the number of entities in the model."
        )

    for idx in range(len(hyperparameter_space)):
        for key, value in hyperparameter_space[idx].items():
            if key not in model_config[idx]:
                raise Exception(
                    f"Hyperparameter {key} is not a part of the model configuration."
                )
            if not value:
                raise Exception(f"No values provided for hyperparameter {key}.")

        for key, value in model_config[idx].items():
            if key not in hyperparameter_space[idx].keys():
                hyperparameter_space[idx][key] = [value]


SCORE_METRICS = [
    "average_precision",
    "neg_brier_score",
    "neg_log_loss",
    "roc_auc",
    "roc_auc_ovo",
    "roc_auc_ovo_weighted",
    "roc_auc_ovr",
    "roc_auc_ovr_weighted",
]


def hyperparameter_tuning(
    metric_to_optimize: str,
    model: Model,
    hyperparameter_space: list[dict[str, list[str]]],
    input: LocalInputs,
    labels: LocalLabels,
) -> list[Tuple[Model, float]]:
    """
    Perform Hyperparameter Tuning on a model using Grid Search.

    Args:
        metric_to_optimize: str
            The performance metric to be used to find the best model.
        model: turboml.Model
            The model object to be tuned.
        hyperparameter_space: list[dict[str, list[str]]]
            A list of dictionaries specifying the hyperparameters and the corresponding values to be tested for each entity which is a part of `model`.
        input: Inputs
            The input configuration for the models
        labels: Labels
            The label configuration for the models

    Returns:
        list[Tuple[Model, float]]: The list of all models with their corresponding scores sorted in descending order.

    """
    _check_hyperparameter_space(hyperparameter_space, model)

    product_spaces = [
        list(itertools.product(*space.values())) for space in hyperparameter_space
    ]
    combined_product = list(itertools.product(*product_spaces))

    keys = [list(space.keys()) for space in hyperparameter_space]

    hyperparameter_combinations = []
    for product_combination in combined_product:
        combined_dicts = []
        for key_set, value_set in zip(keys, product_combination, strict=False):
            combined_dicts.append(dict(zip(key_set, value_set, strict=False)))
        hyperparameter_combinations.append(combined_dicts)

    return algorithm_tuning(
        [
            Model._construct_model(config, index=0, is_flat=True)[0]
            for config in hyperparameter_combinations
        ],
        metric_to_optimize,
        input,
        labels,
    )


def algorithm_tuning(
    models_to_test: list[Model],
    metric_to_optimize: str,
    input: LocalInputs,
    labels: LocalLabels,
) -> list[Tuple[Model, float]]:
    """
    Test a list of models to find the best model for the given metric.

    Args:
        models_to_test: List[turboml.Model]
            List of models to be tested.
        metric_to_optimize: str
            The performance metric to be used to find the best model.
        input: Inputs
            The input configuration for the models
        labels: Labels
            The label configuration for the models

    Returns:
        list[Tuple[Model, float]]: The list of all models with their corresponding scores sorted in descending order.
    """
    from sklearn import metrics
    import pandas as pd

    if metric_to_optimize not in metrics.get_scorer_names():
        raise Exception(f"{metric_to_optimize} is not yet supported.")
    if not models_to_test:
        raise Exception("No models specified for testing.")

    prediction_column = (
        "score" if metric_to_optimize in SCORE_METRICS else "predicted_class"
    )

    perf_metric = metrics.get_scorer(metric_to_optimize)
    assert isinstance(
        perf_metric, metrics._scorer._Scorer
    ), f"Invalid metric {metric_to_optimize}"

    input_df, label_df = _resolve_duplicate_columns(
        input.dataframe, labels.dataframe, input.key_field
    )
    merged_df = pd.merge(input_df, label_df, on=input.key_field)

    input_spec = InputSpec(
        key_field=input.key_field,
        time_field=input.time_field or "",
        numerical_fields=input.numerical_fields or [],
        categorical_fields=input.categorical_fields or [],
        textual_fields=input.textual_fields or [],
        imaginal_fields=input.imaginal_fields or [],
        label_field=labels.label_field,
    )

    input_table = TbPyArrow.df_to_table(merged_df, input_spec)
    results = []
    for model in models_to_test:
        trained_model, score = get_score_for_model(
            model, input_table, input_spec, labels, perf_metric, prediction_column
        )
        show_model_results(trained_model, score, metric_to_optimize)
        results.append((trained_model, score))

    return sorted(results, key=lambda x: x[1], reverse=True)


def show_model_results(trained_model, score, metric_name):
    """Displays formatted information for a trained model and its performance score."""
    model_name = trained_model.__class__.__name__
    model_params = {
        k: v
        for k, v in trained_model.__dict__.items()
        if not k.startswith("_") and not callable(v)
    }

    params_display = "\n".join(f"  - {k}: {v}" for k, v in model_params.items())

    print(f"\nModel: {model_name}")
    print("Parameters:")
    print(params_display)
    print(f"{metric_name.capitalize()} Score: {score:.5f}\n")


# api.py
-## Location -> root_directory.common
```python
from .env import CONFIG

import os
import logging
from typing import Optional
import time
import requests

import jwt
import tenacity
import uuid


# Refresh access token if it is about to expire in 1 hour
TOKEN_EXPIRY_THRESHOLD = 3600

logger = logging.getLogger(__name__)


class ApiException(Exception):
    def __init__(self, message: str, status_code: int):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

    def __repr__(self):
        return f"{self.__class__.__name__}(message={self.message}, status_code={self.status_code})"

    def __str__(self):
        return self.__repr__()


class NotFoundException(ApiException):
    def __init__(self, message: str):
        super().__init__(message, 404)


class Api:
    def __init__(self):
        self.session = requests.Session()
        self._api_key: Optional[str] = None
        self._access_token: Optional[str] = None
        self._namespace: Optional[str] = None
        if api_key := os.getenv("TURBOML_API_KEY"):
            self._api_key = api_key
        if namespace := os.getenv("TURBOML_ACTIVE_NAMESPACE"):
            self._namespace = namespace
            logger.debug(
                f"Namespace set to '{namespace}' from environment variable 'TURBOML_ACTIVE_NAMESPACE'"
            )
        else:
            logger.debug(
                "No namespace set; 'TURBOML_ACTIVE_NAMESPACE' environment variable not found."
            )

    def clear_session(self):
        self._api_key = None
        self._access_token = None

    def login(
        self,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        if api_key:
            self._api_key = api_key
            resp = self.session.get(
                url=f"{self.API_BASE_ADDRESS}/user",
                headers=self.headers,
            )
            if resp.status_code != 200:
                self._api_key = None
                raise ApiException("Invalid API key", status_code=resp.status_code)
            return
        if username:
            assert password, "Provide a password along with username"
            resp = self.session.post(
                url=f"{self.API_BASE_ADDRESS}/login",
                data={"username": username, "password": password},
            )
            if resp.status_code != 200:
                raise ApiException(
                    "Invalid username/password", status_code=resp.status_code
                )
            self._access_token = resp.json()["access_token"]
            return
        raise ValueError("Provide either an API key or username/password")

    def _refresh_access_token_if_about_to_expire(self) -> None:
        assert self._access_token, "No access token found"
        decoded_jwt = jwt.decode(
            self._access_token,
            algorithms=["HS256"],
            options={"verify_signature": False},
        )
        token_expiry = decoded_jwt.get("exp")
        if token_expiry - time.time() < TOKEN_EXPIRY_THRESHOLD:
            resp = self.session.post(
                url=f"{self.API_BASE_ADDRESS}/renew_token",
                headers={"Authorization": f"Bearer {self._access_token}"},
            )
            if resp.status_code != 200:
                raise ApiException(
                    "Failed to refresh access token try to login in again using login()",
                    status_code=resp.status_code,
                )
            self._access_token = resp.json()["access_token"]

    @property
    def API_BASE_ADDRESS(self) -> str:
        return CONFIG.TURBOML_BACKEND_SERVER_ADDRESS + "/api"

    @property
    def headers(self) -> dict[str, str]:
        headers = {}
        if self._namespace:
            headers["X-Turboml-Namespace"] = self._namespace
        if self._api_key:
            headers["Authorization"] = f"apiKey {self._api_key}"
            return headers
        if self._access_token:
            self._refresh_access_token_if_about_to_expire()
            headers["Authorization"] = f"Bearer {self._access_token}"
            return headers
        raise ValueError("No API key or access token found. Please login first")

    def set_active_namespace(self, namespace: str):
        original_namespace = self._namespace
        self._namespace = namespace
        resp = self.get("user/namespace")
        if resp.status_code not in range(200, 300):
            self._namespace = original_namespace
            raise Exception(f"Failed to set namespace: {resp.json()['detail']}")

    @property
    def arrow_headers(self) -> list[tuple[bytes, bytes]]:
        return [(k.lower().encode(), v.encode()) for k, v in self.headers.items()]

    @property
    def namespace(self) -> str:
        return self.get("user/namespace").json()

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=5),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    def _request(self, method, url, headers, params, data, json, files):
        resp = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            data=data,
            json=json,
            files=files,
        )
        if resp.status_code == 502:  # Catch and retry on Bad Gateway
            raise Exception("Bad Gateway")
        return resp

    def request(
        self,
        method,
        endpoint,
        host=None,
        data=None,
        params=None,
        json=None,
        files=None,
        headers=None,
        exclude_namespace=False,
    ):
        if not host:
            host = self.API_BASE_ADDRESS
        combined_headers = self.headers.copy()
        if headers:
            combined_headers.update(headers)
        # Exclude the namespace header if requested
        if exclude_namespace:
            combined_headers.pop("X-Turboml-Namespace", None)

        idempotency_key = uuid.uuid4().hex
        combined_headers["Idempotency-Key"] = idempotency_key

        resp = self._request(
            method=method.upper(),
            url=f"{host}/{endpoint}",
            headers=combined_headers,
            params=params,
            data=data,
            json=json,
            files=files,
        )
        if not (200 <= resp.status_code < 300):
            try:
                json_resp = resp.json()
                error_details = json_resp.get("detail", json_resp)
            except ValueError:
                error_details = resp.text
            if resp.status_code == 404:
                raise NotFoundException(error_details)
            raise ApiException(
                error_details,
                status_code=resp.status_code,
            ) from None
        return resp

    def get(self, endpoint, **kwargs):
        return self.request("GET", endpoint, **kwargs)

    def options(self, endpoint, **kwargs):
        return self.request("OPTIONS", endpoint, **kwargs)

    def head(self, endpoint, **kwargs):
        return self.request("HEAD", endpoint, **kwargs)

    def post(self, endpoint, **kwargs):
        return self.request("POST", endpoint, **kwargs)

    def put(self, endpoint, **kwargs):
        return self.request("PUT", endpoint, **kwargs)

    def patch(self, endpoint, **kwargs):
        return self.request("PATCH", endpoint, **kwargs)

    def delete(self, endpoint, **kwargs):
        return self.request("DELETE", endpoint, **kwargs)


api = Api()

```

# concurrent.py
-## Location -> root_directory.common
```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

_tb_use_multiprocessing = True


def use_multiprocessing(enable: bool):
    global _tb_use_multiprocessing
    _tb_use_multiprocessing = enable


def multiprocessing_enabled() -> bool:
    global _tb_use_multiprocessing
    return _tb_use_multiprocessing


def get_executor_pool_class() -> type[ProcessPoolExecutor | ThreadPoolExecutor]:
    return ProcessPoolExecutor if multiprocessing_enabled() else ThreadPoolExecutor

```

# dataloader.py
-## Location -> root_directory.common
```python
from __future__ import annotations
from datetime import datetime
from enum import StrEnum
import hashlib
from pickle import PickleError
import sys
import struct
from functools import partial
from typing import (
    Callable,
    List,
    Optional,
    TypedDict,
    TYPE_CHECKING,
)
import tempfile
import os
import importlib.util
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from subprocess import run
import logging

from tqdm import tqdm
import pandas as pd
from google.protobuf.message import Message
import pyarrow
import pyarrow.flight
from pyarrow.flight import FlightDescriptor
from turboml.common.concurrent import multiprocessing_enabled, get_executor_pool_class
from .api import api
from .env import CONFIG
from .internal import TbItertools, TbPyArrow

if TYPE_CHECKING:
    from types import ModuleType
    from .models import (
        RegisteredSchema,
    )
    from google.protobuf import message


logger = logging.getLogger(__name__)


class StreamType(StrEnum):
    INPUT_TOPIC = "input_topic"
    OUTPUT = "output"
    TARGET_DRIFT = "target_drift"
    UNIVARIATE_DRIFT = "univariate_drift"
    MULTIVARIATE_DRIFT = "multivariate_drift"


class Record(TypedDict):
    offset: int
    record: bytes


def _get_raw_msgs(dataset_type: StreamType, name: str, **kwargs):
    """
    Returns a dataframe of type [offset: int, record: bytes] for the dataset
    """
    if dataset_type == StreamType.UNIVARIATE_DRIFT:
        numeric_feature = kwargs.get("numeric_feature")
        if numeric_feature is None:
            raise ValueError("numeric_feature is required for univariate drift")
        name = f"{name}:{numeric_feature}"
    if dataset_type == StreamType.MULTIVARIATE_DRIFT:
        label = kwargs.get("label")
        if label is None:
            raise ValueError("label is required for multivariate drift")
        name = f"{name}:{label}"
    arrow_descriptor = pyarrow.flight.Ticket(f"{dataset_type.value}:{name}")
    client = pyarrow.flight.connect(f"{CONFIG.ARROW_SERVER_ADDRESS}")
    TbPyArrow.wait_for_available(client)
    reader = client.do_get(
        arrow_descriptor,
        options=pyarrow.flight.FlightCallOptions(headers=api.arrow_headers),
    )
    LOG_FREQUENCY_SEC = 3
    last_log_time = 0
    yielded_total = 0
    yielded_batches = 0
    start_time = datetime.now().timestamp()
    while True:
        table = reader.read_chunk().data
        df = TbPyArrow.arrow_table_to_pandas(table)
        if df.empty:
            logger.info(
                f"Yielded {yielded_total} records ({yielded_batches} batches) in {datetime.now().timestamp() - start_time:.0f} seconds"
            )
            break
        yielded_total += len(df)
        yielded_batches += 1
        if (now := datetime.now().timestamp()) - last_log_time > LOG_FREQUENCY_SEC:
            logger.info(
                f"Yielded {yielded_total} records ({yielded_batches} batches) in {now - start_time:.0f} seconds"
            )
            last_log_time = now
        assert isinstance(df, pd.DataFrame)
        yield df


PROTO_PREFIX_BYTE_LEN = 6


def _records_to_proto_messages(
    df: pd.DataFrame,
    proto_msg: Callable[[], message.Message],
) -> tuple[list[int], list[message.Message]]:
    offsets = []
    proto_records = []
    for _, offset_message in df.iterrows():
        offset, message = offset_message["offset"], offset_message["record"]
        assert isinstance(message, bytes)
        proto = proto_msg()
        proto.ParseFromString(message[PROTO_PREFIX_BYTE_LEN:])
        offsets.append(offset)
        proto_records.append(proto)
    return offsets, proto_records


class RecordList(TypedDict):
    offsets: list[int]
    records: list[message.Message]


# HACK: Since it is observed that the ProcessPoolExecutor fails to pickle proto messages under
# certain (not yet understood) conditions, we switch to the ThreadPoolExecutor upon encountering
# such an error.
# Ref: https://turboml.slack.com/archives/C07FM09V0MA/p1729082597265189


def get_proto_msgs(
    dataset_type: StreamType,
    name: str,
    proto_msg: Callable[[], message.Message],
    **kwargs,
    # limit: int = -1
) -> list[Record]:
    executor_pool_class = get_executor_pool_class()
    try:
        return _get_proto_msgs(
            dataset_type, name, proto_msg, executor_pool_class, **kwargs
        )
    except PickleError as e:
        if not multiprocessing_enabled():
            raise e
        logger.warning(
            f"Failed to pickle proto message class {proto_msg}: {e!r}. Retrying with ThreadPoolExecutor"
        )

        return _get_proto_msgs(
            dataset_type, name, proto_msg, ThreadPoolExecutor, **kwargs
        )


def _get_proto_msgs(
    dataset_type: StreamType,
    name: str,
    proto_msg: Callable[[], message.Message],
    executor_cls: type[ProcessPoolExecutor | ThreadPoolExecutor],
    **kwargs,
) -> list[Record]:
    messages_generator = _get_raw_msgs(dataset_type, name, **kwargs)
    offsets = []
    records = []
    with executor_cls(max_workers=os.cpu_count()) as executor:
        futures: list[Future[tuple[list[int], list[message.Message]]]] = []
        for df in messages_generator:
            future = executor.submit(
                _records_to_proto_messages,
                df,
                proto_msg,
            )
            futures.append(future)
        for future in futures:
            offsets_chunk, records_chunk = future.result()
            offsets.extend(offsets_chunk)
            records.extend(records_chunk)

    ret = []
    for i, record in zip(offsets, records, strict=True):
        ret.append({"offset": i, "record": record})
    return ret


def create_protobuf_from_row_tuple(
    row: tuple,
    fields: List[str],
    proto_cls: Callable[[], message.Message],
    prefix: bytes,
):
    """Create a Protocol Buffers (protobuf) message by populating its fields with values from a tuple of row data.

    Args:
        row (Iterable): An iterable representing a row of data. Each element corresponds to a field in the protobuf message.
        fields (List[str]): A list of field names corresponding to the fields in the protobuf message class.
        proto_cls (type): The protobuf message class to instantiate.
        prefix (str): A string prefix to be concatenated with the serialized message.

    Returns:
        str: A string representing the serialized protobuf message with the specified prefix.
    """
    import pandas as pd

    my_msg = proto_cls()
    for i, field in enumerate(fields):
        value = row[i]

        if pd.isna(value):
            # Leave the field unset if the value is NaN
            continue

        try:
            setattr(my_msg, field, value)
        except TypeError as e:
            logger.error(
                f"Error setting field '{field}' with value '{value}' in '{row}': {e!r}"
            )
            raise

    return prefix + my_msg.SerializeToString()


def create_protobuf_from_row_dict(
    row: dict,
    proto_cls: type[message.Message],
    prefix: bytes,
):
    """Create a Protocol Buffers (protobuf) message by populating its fields with values a from dictionary row of data.

    Args:
        row (Iterable): An iterable representing a row of data. Each element corresponds to a field in the protobuf message.
        proto_cls (type): The protobuf message class to instantiate.
        prefix (str): A string prefix to be concatenated with the serialized message.

    Returns:
        str: A string representing the serialized protobuf message with the specified prefix.
    """
    my_msg = proto_cls()
    for field, value in row.items():
        if value is None:
            continue  # skip None values -- protobuf isn't happy with them
        try:
            setattr(my_msg, field, value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Error setting field '{field}'='{value}': {e!r}") from e

    return prefix + my_msg.SerializeToString()


def _get_message_cls_from_pb_module(
    module: ModuleType, message_name: str | None
) -> type[Message] | None:
    messageClasses = [
        v
        for v in vars(module).values()
        if isinstance(v, type) and issubclass(v, Message)
    ]
    if len(messageClasses) == 0:
        logger.error(
            f"No message classes found in protobuf module composed of classes: {list(vars(module).keys())}"
        )
        return None

    if message_name is None:
        return messageClasses[0] if len(messageClasses) > 0 else None

    matching_class = [v for v in messageClasses if v.DESCRIPTOR.name == message_name]
    if len(matching_class) == 0:
        all_message_names = [v.DESCRIPTOR.name for v in messageClasses]
        logger.error(
            f"Could not find message class '{message_name}' in protobuf module composed of classes: {all_message_names}"
        )
        return None
    return matching_class[0]


def _canonicalize_schema_body(schema_body: str) -> str:
    "Schema registry formats does itd own canonicalization, but we need to do it for comparison"
    return "\n".join(
        line.strip()  # Remove leading/trailing whitespace
        for line in schema_body.split("\n")
        if (
            not line.strip().startswith("//")  # Remove comments
            and not line.strip() == ""
        )  # Remove empty lines
    )


def get_protobuf_class(
    schema: str, message_name: str | None, retry: bool = True
) -> type[Message] | None:
    """
    Generate a python class from a Protocol Buffers (protobuf) schema and message name.
    If class_name is None, the first class in the schema is returned.
    If a matching class is not found, None is returned.
    """
    schema = _canonicalize_schema_body(schema)
    basename = f"p_{hashlib.md5(schema.encode()).hexdigest()[:8]}"
    module_name = f"{basename}_pb2"

    if module_name in sys.modules:
        module = sys.modules[module_name]
        return _get_message_cls_from_pb_module(module, message_name)

    with tempfile.TemporaryDirectory(prefix="turboml_") as tempdir:
        filename = os.path.join(tempdir, f"{basename}.proto")
        with open(filename, "w") as f:
            _ = f.write(schema)
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)
        run(
            [
                "protoc",
                f"--python_out={dirname}",
                f"--proto_path={dirname}",
                basename,
            ],
            check=True,
        )
        module_path = os.path.join(dirname, module_name + ".py")
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert module_spec is not None
        assert module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        if _get_message_cls_from_pb_module(module, None) is None:
            # Retry once if the module is empty
            # This is a rather bizarre behavior that seems to occur
            # in our CI, so we retry once to see if it resolves itself
            if not retry:
                return None
            logger.error(
                f"A seemingly empty protobuf module was generated from module_path='{module_path}', schema={schema}. Retrying once..."
            )
            return get_protobuf_class(schema, message_name, retry=False)

        sys.modules[module_name] = module
        return _get_message_cls_from_pb_module(module, message_name)


def upload_df(
    dataset_id: str,
    df: pd.DataFrame,
    schema: RegisteredSchema,
    protoMessageClass: Optional[type[message.Message]] = None,
) -> None:
    """Upload data from a DataFrame to a dataset after preparing and serializing it as Protocol Buffers (protobuf) messages.

    Args:
        dataset_id (str): The Kafka dataset_id to which the data will be sent.
        df (pd.DataFrame): The DataFrame containing the data to be uploaded.
        schema (Schema): Dataset schema.
        protoMessageClass (Optional(Message)): Protobuf Message Class to use. Generated if not provided.
    """
    # dataset = api.get(f"dataset?dataset_id={dataset_id}").json()
    # dataset = Dataset(**dataset)
    if protoMessageClass is None:
        protoMessageClass = get_protobuf_class(
            schema=schema.schema_body, message_name=schema.message_name
        )
        if protoMessageClass is None:
            raise ValueError(
                f"Could not find protobuf message class message={schema.message_name} schema={schema.schema_body}"
            )

    fields = df.columns.tolist()
    prefix = struct.pack("!xIx", schema.id)
    descriptor = FlightDescriptor.for_command(f"produce:{dataset_id}")
    pa_schema = pyarrow.schema([("value", pyarrow.binary())])

    partial_converter_func = partial(
        create_protobuf_from_row_tuple,
        fields=fields,
        proto_cls=protoMessageClass,
        prefix=prefix,
    )

    logger.info(f"Uploading {df.shape[0]} rows to dataset {dataset_id}")
    executor_pool_class = get_executor_pool_class()

    client = pyarrow.flight.connect(f"{CONFIG.ARROW_SERVER_ADDRESS}")
    TbPyArrow.wait_for_available(client)
    writer, _ = client.do_put(
        descriptor,
        pa_schema,
        options=pyarrow.flight.FlightCallOptions(headers=api.arrow_headers),
    )
    try:
        _upload_df_batch(df, executor_pool_class, partial_converter_func, writer)
    except (PickleError, ModuleNotFoundError) as e:
        if not multiprocessing_enabled():
            raise e
        logger.warning(
            f"Dataframe batch update failed due to exception {e!r}. Retrying with ThreadPoolExecutor"
        )
        _upload_df_batch(df, ThreadPoolExecutor, partial_converter_func, writer)

    logger.info("Upload complete. Waiting for server to process messages.")
    writer.close()


def _upload_df_batch(
    df: pd.DataFrame,
    executor_pool_class: type[ProcessPoolExecutor | ThreadPoolExecutor],
    partial_func,
    writer,
):
    with executor_pool_class(max_workers=os.cpu_count()) as executor:
        data_iterator = executor.map(
            partial_func,
            df.itertuples(index=False, name=None),
            chunksize=1024,
        )

        CHUNK_SIZE = 1024
        row_length = df.shape[0]
        with tqdm(
            total=row_length, desc="Progress", unit="rows", unit_scale=True
        ) as pbar:
            for messages in TbItertools.chunked(data_iterator, CHUNK_SIZE):
                batch = pyarrow.RecordBatch.from_arrays([messages], ["value"])
                writer.write(batch)
                pbar.update(len(messages))

```

# datasets.py
-## Location -> root_directory.common
```python
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import inspect
import logging
import re
import time
from typing import TYPE_CHECKING, Callable, Final, Generic, TypeVar

import ibis
from pydantic import (
    BaseModel,
    ConfigDict,
    model_validator,
)

from turboml.common import dataloader
from turboml.common.api import ApiException, api, NotFoundException
from turboml.common.feature_engineering import (
    FeatureEngineering,
    LocalFeatureEngineering,
    get_features,
)
from turboml.common.internal import TbPandas
from turboml.common.protos import output_pb2

from .models import (
    DataDrift,
    Dataset,
    DatasetRegistrationRequest,
    DatasetRegistrationResponse,
    Datatype,
    DatasetField,
    RegisteredSchema,
    TurboMLResourceIdentifier,
    DatasetSchema,
)  # noqa TCH0001
import pandas as pd

if TYPE_CHECKING:
    from google.protobuf import message

DATATYPE_NUMERICAL = Datatype.FLOAT
DATATYPE_CATEGORICAL = Datatype.INT64
DATATYPE_LABEL = Datatype.FLOAT
DATATYPE_KEY = Datatype.STRING
DATATYPE_IMAGE = Datatype.BYTES
DATATYPE_TEXT = Datatype.STRING
DATATYPE_TIMETICK = Datatype.INT64

logger = logging.getLogger("turboml.datasets")


class LocalInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: pd.DataFrame
    key_field: DatasetField
    time_field: DatasetField | None
    numerical_fields: list[DatasetField]
    categorical_fields: list[DatasetField]
    textual_fields: list[DatasetField]
    imaginal_fields: list[DatasetField]

    @dataclass
    class _FieldMeta:
        name: str
        _type: str
        wanted_dtype: Datatype

    def all_fields_meta(self):
        return (
            [LocalInputs._FieldMeta(self.key_field, "key", DATATYPE_KEY)]
            + [
                LocalInputs._FieldMeta(field, "numerical", DATATYPE_NUMERICAL)
                for field in self.numerical_fields
            ]
            + [
                LocalInputs._FieldMeta(field, "categorical", DATATYPE_CATEGORICAL)
                for field in self.categorical_fields
            ]
            + [
                LocalInputs._FieldMeta(field, "textual", DATATYPE_TEXT)
                for field in self.textual_fields
            ]
            + [
                LocalInputs._FieldMeta(field, "imaginal", DATATYPE_IMAGE)
                for field in self.imaginal_fields
            ]
            + (
                [LocalInputs._FieldMeta(self.time_field, "time", DATATYPE_TIMETICK)]
                if self.time_field
                else []
            )
        )

    @model_validator(mode="after")
    def select_fields(self):
        all_fields_meta = self.all_fields_meta()

        all_field_names = [field.name for field in all_fields_meta]

        # if a field is used in more than one place, we'll raise an error
        if len(all_field_names) != len(set(all_field_names)):
            # figure out duplicates
            duplicates = [
                field for field, count in Counter(all_field_names).items() if count > 1
            ]
            raise ValueError(f"Fields {duplicates} are specified more than once.")

        absent_fields = set(all_field_names) - set(self.dataframe.columns)
        if absent_fields:
            raise ValueError(
                f"Fields {absent_fields} are not present in the dataframe."
            )

        df = pd.DataFrame()
        for field_meta in all_fields_meta:
            name, type_, wanted_dtype = (
                field_meta.name,
                field_meta._type,
                field_meta.wanted_dtype,
            )
            try:
                column = self.dataframe[name]
                assert isinstance(column, pd.Series)
                column = TbPandas.fill_nans_with_default(column)
                column = column.astype(wanted_dtype.to_pandas_dtype())
                df[name] = column
            except Exception as e:
                raise ValueError(
                    f"Failed to convert {type_} field '{name}' to {wanted_dtype}. "
                    f"Error from pandas.astype(): {e!r}"
                ) from e

        self.dataframe = df
        return self

    @model_validator(mode="after")
    def _validate_time_field(self):
        if not self.time_field:
            return self
        time_field_is_datetime64 = pd.api.types.is_datetime64_any_dtype(
            self.dataframe[self.time_field]
        )
        if not time_field_is_datetime64:
            raise ValueError(f"Field '{self.time_field}' is not of a datetime type.")
        return self

    def validate_fields(self, dataframe: pd.DataFrame):
        ## TODO: key field?

        for field in self.numerical_fields:
            if not pd.api.types.is_numeric_dtype(dataframe[field]):
                raise ValueError(f"Field '{field}' is not of a numeric type.")

        # QUESTION: why is this commented out?
        # for field in self.categorical_fields:
        #    if not pd.api.types.is_categorical_dtype(dataframe[field]):
        #        raise ValueError(f"Field '{field}' is not of categorical type.")

        for field in self.textual_fields:
            if not pd.api.types.is_string_dtype(dataframe[field]):
                raise ValueError(f"Field '{field}' is not of a textual type.")

        # QUESTION: why is this commented out?
        # for field in self.imaginal_fields:
        #     if not pd.api.types.is_string_dtype(dataframe[field]):
        #         raise ValueError(f"Field '{field}' is not of a imaginal type.")


# NOTE: At most places where we were accepting `Inputs` previously, we should accept `LocalInputs | OnlineInputs`.
# However for the moment I've kept it as `LocalInputs`, which includes `OnlineInputs` as well since we're
# subclassing (for now) and basically load the entire dataset into memory by default.
# At a later point we should change this so that its possible to pass streaming generators
# from online datasets without loading everything into memory.
class OnlineInputs(LocalInputs):
    dataset_id: TurboMLResourceIdentifier


class LocalLabels(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: pd.DataFrame
    key_field: DatasetField
    label_field: DatasetField

    @model_validator(mode="after")
    def validate_and_select_label_field(self):
        if self.label_field not in self.dataframe:
            raise ValueError(
                f"Field '{self.label_field}' is not present in the dataframe."
            )
        label_field_is_numeric = pd.api.types.is_numeric_dtype(
            self.dataframe[self.label_field]
        )
        if not label_field_is_numeric:
            raise ValueError(f"Field '{self.label_field}' is not of a numeric type.")
        df = pd.DataFrame()
        df[self.label_field] = self.dataframe[self.label_field].astype(DATATYPE_LABEL)
        df[self.key_field] = self.dataframe[self.key_field].astype(DATATYPE_KEY)
        self.dataframe = df
        return self


class OnlineLabels(LocalLabels):
    dataset_id: TurboMLResourceIdentifier


FE = TypeVar("FE", LocalFeatureEngineering, FeatureEngineering)


class _BaseInMemoryDataset(Generic[FE]):
    _init_key: Final[object] = object()

    def __init__(
        self,
        init_key: object,
        schema: DatasetSchema,
        df: pd.DataFrame,
        key_field: str,
        feature_engineering: Callable[[pd.DataFrame], FE] = LocalFeatureEngineering,
    ):
        if init_key not in [_BaseInMemoryDataset._init_key, OnlineDataset._init_key]:
            raise AssertionError(
                f"Use from_* methods to instantiate {self.__class__.__name__}"
            )
        self.schema = schema
        self.df = df  # The dataset, as it is
        self.key_field = key_field
        self.feature_engineering = feature_engineering(self.df.copy())

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{key}={value}' for key, value in vars(self).items())})"

    @staticmethod
    def from_schema(
        schema: DatasetSchema,
        key_field: str,
        feature_engineering: Callable[[pd.DataFrame], FE] = LocalFeatureEngineering,
    ) -> _BaseInMemoryDataset[FE]:
        return _BaseInMemoryDataset(
            _BaseInMemoryDataset._init_key,
            schema,
            pd.DataFrame(),
            key_field,
            feature_engineering,
        )

    def __getitem__(self, item):
        """
        Returns a new dataset that is a view of the original dataset.
        """
        if not isinstance(item, slice):
            raise NotImplementedError("Only slicing is supported for now")

        df_view = self.df[item].copy()
        fe = self.feature_engineering
        assert isinstance(df_view, pd.DataFrame)
        assert isinstance(fe, LocalFeatureEngineering)

        return _BaseInMemoryDataset(
            _BaseInMemoryDataset._init_key,
            self.schema,
            df_view,
            self.key_field,
            feature_engineering=lambda df: fe.clone_with_df(df),
        )

    def _is_pd_schema_compatible(self, df: pd.DataFrame) -> bool:
        if len(df) == 0:
            raise ValueError("Empty dataframe not allowed")
        return DatasetSchema.from_pd(df) == self.schema

    def add_pd(self, df: pd.DataFrame):
        if not self._is_pd_schema_compatible(df):
            raise ValueError(
                "Schema mismatch: the dataframe does not match the dataset's input schema."
                f" Expected: {self.schema}, got: {DatasetSchema.from_pd(df)}"
            )
        self.df = pd.concat([self.df, df], ignore_index=True)
        self.feature_engineering._update_input_df(self.df.copy())

    def get_model_inputs(
        self,
        numerical_fields: list | None = None,
        categorical_fields: list | None = None,
        textual_fields: list | None = None,
        imaginal_fields: list | None = None,
        time_field: str | None = None,
    ):
        # Normalize
        if numerical_fields is None:
            numerical_fields = []
        if categorical_fields is None:
            categorical_fields = []
        if textual_fields is None:
            textual_fields = []
        if imaginal_fields is None:
            imaginal_fields = []

        return LocalInputs(
            dataframe=self.feature_engineering.local_features_df,
            key_field=self.key_field,
            numerical_fields=numerical_fields,
            categorical_fields=categorical_fields,
            textual_fields=textual_fields,
            imaginal_fields=imaginal_fields,
            time_field=time_field,
        )

    def get_model_labels(self, label_field: str):
        return LocalLabels(
            dataframe=self.feature_engineering.local_features_df,
            key_field=self.key_field,
            label_field=label_field,
        )


class LocalDataset(_BaseInMemoryDataset[LocalFeatureEngineering]):
    """
    LocalDataset represents an in-memory dataset. In-memory datasets can
    be used for local feature engineering experiments, and training local models.
    A LocalDataset can also be upgraded to an OnlineDataset for online feature
    engineering and serving models based on the same data.
    """

    def __getitem__(self, item):
        s = super().__getitem__(item)
        assert isinstance(s, _BaseInMemoryDataset)
        return LocalDataset(
            LocalDataset._init_key,
            s.schema,
            s.df,
            s.key_field,
            feature_engineering=lambda _: s.feature_engineering,
        )

    def __len__(self):
        return len(self.df)

    @staticmethod
    def from_pd(
        df: pd.DataFrame,
        key_field: str,
    ) -> LocalDataset:
        if len(df) == 0:
            raise ValueError("Empty dataframe")
        schema = DatasetSchema.from_pd(df)
        return LocalDataset(LocalDataset._init_key, schema, df, key_field)

    def to_online(self, id: str, load_if_exists: bool = False) -> OnlineDataset:
        return OnlineDataset.from_local_dataset(self, id, load_if_exists)


class _InMemoryDatasetOnlineFE(_BaseInMemoryDataset[FeatureEngineering]):
    pass


class OnlineDataset:
    """
    OnlineDataset represents a dataset managed and stored by the TurboML platform.
    In addition to operations available on LocalDataset, an online dataset can be
    used to "materialize" engineered features, register and monitor drift, and
    serve models based on the data.
    """

    _init_key = object()

    def __init__(
        self,
        dataset_id: str,
        init_key: object,
        key_field: str,
        protobuf_cls: type[message.Message],
        registered_schema: RegisteredSchema,
        fe: LocalFeatureEngineering | None = None,
    ):
        if init_key is not OnlineDataset._init_key:
            raise AssertionError(
                f"Use load() or from_*() methods to instantiate {self.__class__.__name__}"
            )

        def feature_engineering(df: pd.DataFrame):
            if fe:
                return FeatureEngineering.inherit_from_local(fe, dataset_id)
            return FeatureEngineering(dataset_id, df)

        self.__local_dataset = _InMemoryDatasetOnlineFE.from_schema(
            registered_schema.native_schema,
            key_field=key_field,
            feature_engineering=feature_engineering,
        )

        self.dataset_id = dataset_id
        self.protobuf_cls = protobuf_cls
        self.registered_schema = registered_schema

    @property
    def schema(self):
        return self.__local_dataset.schema

    @property
    def key_field(self):
        return self.__local_dataset.key_field

    @property
    def feature_engineering(self):
        return self.__local_dataset.feature_engineering

    @property
    def preview_df(self):
        return self.__local_dataset.df

    def __repr__(self):
        return f"OnlineDataset(id={self.dataset_id}, key_field={self.key_field}, schema={self.schema})"

    @staticmethod
    def load(dataset_id: str) -> OnlineDataset | None:
        try:
            dataset = api.get(f"dataset?dataset_id={dataset_id}").json()
        except NotFoundException:
            return None
        dataset = Dataset(**dataset)
        schema = api.get(f"dataset/{dataset_id}/schema").json()
        schema = RegisteredSchema(**schema)
        protobuf_cls = dataloader.get_protobuf_class(
            schema=schema.schema_body,
            message_name=dataset.meta.input_pb_message_name,
        )
        if protobuf_cls is None:
            raise ValueError(
                f"Failed to load protobuf message class for message_name={dataset.message_name}, schema={schema.schema_body}"
            )
        online_dataset = OnlineDataset(
            dataset_id=dataset_id,
            key_field=dataset.key,
            init_key=OnlineDataset._init_key,
            protobuf_cls=protobuf_cls,
            registered_schema=schema,
        )
        online_dataset.sync_features()
        return online_dataset

    @staticmethod
    def _register_dataset(
        dataset_id: str, columns: dict[str, Datatype], key_field: str
    ):
        registration_request = DatasetRegistrationRequest(
            dataset_id=dataset_id,
            data_schema=DatasetRegistrationRequest.ExplicitSchema(fields=columns),
            key_field=key_field,
        )
        try:
            response = api.post("dataset", json=registration_request.model_dump())
        except ApiException as e:
            if "already exists" in str(e):
                raise ValueError(
                    f"Dataset with ID '{dataset_id}' already exists. Use OnlineDataset.load() to load it or specify a different ID."
                ) from e
            raise

        return DatasetRegistrationResponse(**response.json())

    @staticmethod
    def from_local_dataset(
        dataset: LocalDataset, dataset_id: str, load_if_exists: bool = False
    ) -> OnlineDataset:
        if load_if_exists and (online_dataset := OnlineDataset.load(dataset_id)):
            if online_dataset.schema != dataset.schema:
                raise ValueError(
                    f"Dataset already exists with different schema: {online_dataset.schema} != {dataset.schema}"
                )
            return online_dataset
        try:
            response = OnlineDataset._register_dataset(
                dataset_id, dataset.schema.fields, dataset.key_field
            )
        except ApiException as e:
            raise Exception(f"Failed to register dataset: {e!r}") from e

        protobuf_cls = dataloader.get_protobuf_class(
            schema=response.registered_schema.schema_body,
            message_name=response.registered_schema.message_name,
        )
        if protobuf_cls is None:
            raise ValueError(
                f"Failed to load protobuf message class for message_name={response.registered_schema.message_name},"
                f" schema={response.registered_schema.schema_body}"
            )
        online_dataset = OnlineDataset(
            dataset_id=dataset_id,
            key_field=dataset.key_field,
            init_key=OnlineDataset._init_key,
            registered_schema=response.registered_schema,
            protobuf_cls=protobuf_cls,
        )
        try:
            online_dataset.add_pd(dataset.df)
        except Exception as e:
            raise ValueError(f"Failed to push dataset: {e!r}") from e
            ## TODO: cleanup ops
        logger.info(
            f"Pushed dataset {online_dataset.dataset_id}. Note that any feature definitions will have to be materialized before they can be used with online models."
        )
        return online_dataset

    def add_pd(self, df: pd.DataFrame):
        if not self.__local_dataset._is_pd_schema_compatible(df):
            raise ValueError(
                "Schema mismatch: the dataframe does not match the dataset's input schema."
                f" Expected: {self.schema}, got: {DatasetSchema.from_pd(df)}"
            )
        try:
            dataloader.upload_df(
                self.dataset_id, df, self.registered_schema, self.protobuf_cls
            )
        except Exception as e:
            raise ValueError(f"Failed to upload data: {e!r}") from e

        ## TODO:
        # We really shouldn't maintain a local copy of the dataset
        # or its features. Instead we should support a way to iterate through the dataset
        # or derived featuresets in a streaming fashion, for example by using a generator
        # Still, we should make it so the preview_df is populated by the latest few thousand rows
        old_len = len(self.preview_df)
        while True:
            self.sync_features()
            if len(self.preview_df) > old_len:
                break
            time.sleep(0.5)

    def add_row_dict(self, row: dict):
        raise NotImplementedError  ## TODO: complete

    @staticmethod
    def from_pd(
        df: pd.DataFrame, id: str, key_field: str, load_if_exists: bool = False
    ) -> OnlineDataset:
        if load_if_exists and (dataset := OnlineDataset.load(id)):
            return dataset
        df_schema = DatasetSchema.from_pd(df)
        OnlineDataset._register_dataset(id, df_schema.fields, key_field)
        dataset = OnlineDataset.load(id)
        assert dataset is not None
        dataset.add_pd(df)
        return dataset

    # -- fn

    def sync_features(self):
        features_df = get_features(self.dataset_id)
        input_df = features_df[list(self.schema.fields.keys())].copy()
        assert isinstance(input_df, pd.DataFrame)
        self.__local_dataset.df = input_df
        self.__local_dataset.feature_engineering._update_input_df(features_df)

    def to_ibis(self):
        """
        Converts the dataset into an Ibis table.

        Returns:
            ibis.expr.types.Table: An Ibis in-memory table representing the features
            associated with the given dataset_id.

        Raises:
            Exception: If any error occurs during the retrieval of the table name,
            features, or conversion to Ibis table.
        """
        try:
            df = get_features(self.dataset_id)
            return ibis.memtable(df, name=self.dataset_id)
        except Exception as e:
            raise e

    def register_univariate_drift(self, numerical_field: str, label: str | None = None):
        if not numerical_field:
            raise Exception("Numerical field not specified")

        payload = DataDrift(label=label, numerical_fields=[numerical_field])
        api.put(endpoint=f"dataset/{self.dataset_id}/drift", json=payload.model_dump())

    def register_multivariate_drift(self, numerical_fields: list[str], label: str):
        payload = DataDrift(label=label, numerical_fields=numerical_fields)
        api.put(endpoint=f"dataset/{self.dataset_id}/drift", json=payload.model_dump())

    def get_univariate_drift(
        self,
        label: str | None = None,
        numerical_field: str | None = None,
        limit: int = -1,
    ):
        if numerical_field is None and label is None:
            raise Exception("Numerical field and label both cannot be None")

        if numerical_field is not None and label is None:
            label = self._get_default_mv_drift_label([numerical_field])

        return dataloader.get_proto_msgs(
            dataloader.StreamType.UNIVARIATE_DRIFT,
            self.dataset_id,
            output_pb2.Output,
            numeric_feature=label,
        )

    def get_multivariate_drift(
        self,
        label: str | None = None,
        numerical_fields: list[str] | None = None,
        limit: int = -1,
    ):
        if numerical_fields is None and label is None:
            raise Exception("Numerical fields and label both cannot be None")

        if numerical_fields is not None and label is None:
            label = self._get_default_mv_drift_label(numerical_fields)

        return dataloader.get_proto_msgs(
            dataloader.StreamType.MULTIVARIATE_DRIFT,
            self.dataset_id,
            output_pb2.Output,
            label=label,
        )

    def _get_default_mv_drift_label(self, numerical_fields: list[str]):
        payload = DataDrift(numerical_fields=numerical_fields, label=None)

        drift_label = api.get(
            f"dataset/{self.dataset_id}/drift_label", json=payload.model_dump()
        ).json()["label"]

        return drift_label

    def get_model_labels(self, label_field: str):
        local_labels = self.__local_dataset.get_model_labels(label_field)
        return OnlineLabels(
            dataset_id=self.dataset_id,
            **local_labels.model_dump(),
        )

    def get_model_inputs(
        self,
        numerical_fields: list[str] | None = None,
        categorical_fields: list[str] | None = None,
        textual_fields: list[str] | None = None,
        imaginal_fields: list[str] | None = None,
        time_field: str | None = None,
    ):
        local_inputs = self.__local_dataset.get_model_inputs(
            numerical_fields=numerical_fields,
            categorical_fields=categorical_fields,
            textual_fields=textual_fields,
            imaginal_fields=imaginal_fields,
            time_field=time_field,
        )
        return OnlineInputs(
            dataset_id=self.dataset_id,
            **local_inputs.model_dump(),
        )


class PandasHelpers:
    @staticmethod
    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a dataframe by removing NaNs and replacing them with type-default values"""
        norm_df = pd.DataFrame()
        for cname in df.columns:
            col = df[cname]
            assert isinstance(col, pd.Series)
            norm_df[cname] = TbPandas.fill_nans_with_default(col)
        return norm_df


DATA_BASE_URL = (
    "https://raw.githubusercontent.com/TurboML-Inc/colab-notebooks/refs/heads/+data/"
)


class StandardDataset(LocalDataset):
    """
    Base class for standard datasets used in our docs.
    """

    @property
    def description(self):
        assert self.__doc__ is not None, "No docstring"
        desc = re.split(pattern=r"\w+\n\s{4}\-{3,}", string=self.__doc__, maxsplit=0)[0]
        return inspect.cleandoc(desc)

    def __repr__(self):
        NEWLINE = "\n"
        schema_k_v = (f"{i[0]}: {i[1]}" for i in self.schema.fields.items())
        return f"""{self.description}


  Samples  {len(self)}
  Schema   {(NEWLINE + 11 * " ").join(schema_k_v)}
"""


## TODO: cache these datasets on disk (/tmp/turboml/datasets) to avoid downloading them
# every time in CI etc


class FraudDetectionDatasetFeatures(StandardDataset):
    """Fraud Detection - Features

    The dataset contains a total of 200,000 fraudulent and non-fraudulent transactions
    described by 22 features. The corresponding labels are available in the FraudDetectionDatasetLabels dataset.
    """

    def __init__(
        self,
    ):
        tx_df = pd.read_csv(f"{DATA_BASE_URL}/transactions.csv")
        schema = DatasetSchema.from_pd(tx_df)
        super().__init__(self._init_key, schema, tx_df, "transactionID")


class FraudDetectionDatasetLabels(StandardDataset):
    """Fraud Detection - Labels

    The dataset contains a total of 200,000 fraudulent and non-fraudulent transactions.
    The corresponding features are available in the FraudDetectionDatasetFeatures dataset.
    """

    def __init__(self):
        labels_df = pd.read_csv(f"{DATA_BASE_URL}/labels.csv")
        schema = DatasetSchema.from_pd(labels_df)
        super().__init__(self._init_key, schema, labels_df, "transactionID")


_credit_cards_df = None


def _load_credit_cards_dataset():
    global _credit_cards_df
    if _credit_cards_df is not None:
        return _credit_cards_df

    try:
        from river import datasets
    except ImportError:
        raise ImportError(
            "The river library is required to load the CreditCards dataset. "
            "Please install it using `pip install river`."
        ) from None
    cc_feats = []
    cc_labels = []
    for sample, score in datasets.CreditCard():
        cc_feats.append(sample)
        cc_labels.append({"score": score})

    feats_df = pd.DataFrame(cc_feats).reset_index()
    labels_df = pd.DataFrame(cc_labels).reset_index()
    _credit_cards_df = pd.merge(feats_df, labels_df, on="index")
    return _credit_cards_df


class CreditCardsDatasetFeatures(StandardDataset):
    """Credit card frauds - Features

    The dataset contains labels for transactions made by credit cards in September 2013 by european
    cardholders. The dataset presents transactions that occurred in two days, where 492
    out of the 284,807 transactions are fraudulent. The dataset is highly unbalanced, the positive class
    (frauds) account for 0.172% of all transactions.

    The corresponding labels are available in CreditCardsDatasetLabels.

    Dataset source: River (https://riverml.xyz)
    """

    def __init__(self):
        try:
            df = _load_credit_cards_dataset()
        except ImportError as e:
            raise e
        df = df.drop(columns=["score"])
        schema = DatasetSchema.from_pd(df)
        super().__init__(self._init_key, schema, df, "index")


class CreditCardsDatasetLabels(StandardDataset):
    """Credit card frauds - Labels

    The dataset contains labels for transactions made by credit cards in September 2013 by european
    cardholders. The dataset presents transactions that occurred in two days, where 492
    out of the 284,807 transactions are fraudulent. The dataset is highly unbalanced, the positive class
    (frauds) account for 0.172% of all transactions.

    The corresponding features are available in CreditCardsDatasetLabels.

    Dataset source: River (https://riverml.xyz)
    """

    def __init__(self):
        try:
            df = _load_credit_cards_dataset()
        except ImportError as e:
            raise e
        df = df[["index", "score"]]
        assert isinstance(df, pd.DataFrame)
        schema = DatasetSchema.from_pd(df)
        super().__init__(self._init_key, schema, df, "index")

```

# default_model_configs.py
-## Location -> root_directory.common
```python
from google.protobuf import json_format
from frozendict import frozendict
import copy

from turboml.common.protos import config_pb2


class _DefaultModelConfigs:
    def __init__(self):
        self.default_configs = frozendict(
            {
                "MStream": config_pb2.MStreamConfig(
                    num_rows=2, num_buckets=1024, factor=0.8
                ),
                "RCF": config_pb2.RCFConfig(
                    time_decay=0.000390625,
                    number_of_trees=50,
                    output_after=64,
                    sample_size=256,
                ),
                "HST": config_pb2.HSTConfig(n_trees=20, height=12, window_size=50),
                "HoeffdingTreeClassifier": config_pb2.HoeffdingClassifierConfig(
                    delta=1e-7,
                    tau=0.05,
                    grace_period=200,
                    n_classes=2,
                    leaf_pred_method="mc",
                    split_method="gini",
                ),
                "HoeffdingTreeRegressor": config_pb2.HoeffdingRegressorConfig(
                    delta=1e-7, tau=0.05, grace_period=200, leaf_pred_method="mean"
                ),
                "AMFClassifier": config_pb2.AMFClassifierConfig(
                    n_classes=2,
                    n_estimators=10,
                    step=1,
                    use_aggregation=True,
                    dirichlet=0.5,
                    split_pure=False,
                ),
                "AMFRegressor": config_pb2.AMFRegressorConfig(
                    n_estimators=10,
                    step=1,
                    use_aggregation=True,
                    dirichlet=0.5,
                ),
                "FFMClassifier": config_pb2.FFMClassifierConfig(
                    n_factors=10,
                    l1_weight=0,
                    l2_weight=0,
                    l1_latent=0,
                    l2_latent=0,
                    intercept=0,
                    intercept_lr=0.01,
                    clip_gradient=1e12,
                ),
                "FFMRegressor": config_pb2.FFMRegressorConfig(
                    n_factors=10,
                    l1_weight=0,
                    l2_weight=0,
                    l1_latent=0,
                    l2_latent=0,
                    intercept=0,
                    intercept_lr=0.01,
                    clip_gradient=1e12,
                ),
                "SGTClassifier": config_pb2.SGTClassifierConfig(
                    delta=1e-7,
                    gamma=0.1,
                    grace_period=200,
                    **{"lambda": 0.1},  # HACK: lambda is a reserved keyword in Python
                ),
                "SGTRegressor": config_pb2.SGTRegressorConfig(
                    delta=1e-7,
                    gamma=0.1,
                    grace_period=200,
                    **{"lambda": 0.1},
                ),
                "SNARIMAX": config_pb2.SNARIMAXConfig(
                    horizon=1, p=1, d=1, q=1, m=1, sp=0, sd=0, sq=0
                ),
                "ONNX": config_pb2.ONNXConfig(model_save_name=""),
                "LeveragingBaggingClassifier": config_pb2.LeveragingBaggingClassifierConfig(
                    n_models=10,
                    n_classes=2,
                    w=6,
                    bagging_method="bag",
                    seed=0,
                ),
                "HeteroLeveragingBaggingClassifier": config_pb2.HeteroLeveragingBaggingClassifierConfig(
                    n_classes=2,
                    w=6,
                    bagging_method="bag",
                    seed=0,
                ),
                "AdaBoostClassifier": config_pb2.AdaBoostClassifierConfig(
                    n_models=10,
                    n_classes=2,
                    seed=0,
                ),
                "HeteroAdaBoostClassifier": config_pb2.HeteroAdaBoostClassifierConfig(
                    n_classes=2,
                    seed=0,
                ),
                "RandomSampler": config_pb2.RandomSamplerConfig(
                    n_classes=2,
                    desired_dist=[0.5, 0.5],
                    sampling_method="mixed",
                    sampling_rate=1.0,
                    seed=0,
                ),
                "Python": config_pb2.PythonConfig(
                    module_name="", class_name="", venv_name=""
                ),
                "PythonEnsembleModel": config_pb2.PythonEnsembleConfig(
                    module_name="",
                    class_name="",
                    venv_name="",
                ),
                "PreProcessor": config_pb2.PreProcessorConfig(
                    preprocessor_name="MinMax",
                ),
                "NeuralNetwork": config_pb2.NeuralNetworkConfig(
                    dropout=0,
                    layers=[
                        config_pb2.NeuralNetworkConfig.NeuralNetworkLayer(
                            output_size=64,
                            activation="relu",
                            dropout=0.3,
                            residual_connections=[],
                            use_bias=True,
                        ),
                        config_pb2.NeuralNetworkConfig.NeuralNetworkLayer(
                            output_size=64,
                            activation="relu",
                            dropout=0.3,
                            residual_connections=[],
                            use_bias=True,
                        ),
                        config_pb2.NeuralNetworkConfig.NeuralNetworkLayer(
                            output_size=1,
                            activation="sigmoid",
                            dropout=0.3,
                            residual_connections=[],
                            use_bias=True,
                        ),
                    ],
                    loss_function="mse",
                    learning_rate=1e-2,
                    optimizer="sgd",
                    batch_size=64,
                ),
                "ONN": config_pb2.ONNConfig(
                    max_num_hidden_layers=10,
                    qtd_neuron_hidden_layer=32,
                    n_classes=2,
                    b=0.99,
                    n=0.01,
                    s=0.2,
                ),
                "OVR": config_pb2.OVRConfig(
                    n_classes=2,
                ),
                "BanditModelSelection": config_pb2.BanditModelSelectionConfig(
                    bandit="EpsGreedy",
                    metric_name="WindowedMAE",
                ),
                "ContextualBanditModelSelection": config_pb2.ContextualBanditModelSelectionConfig(
                    contextualbandit="LinTS",
                    metric_name="WindowedMAE",
                ),
                "RandomProjectionEmbedding": config_pb2.RandomProjectionEmbeddingConfig(
                    n_embeddings=2,
                    type_embedding="Gaussian",
                ),
                "EmbeddingModel": config_pb2.EmbeddingModelConfig(),
                "MultinomialNB": config_pb2.MultinomialConfig(n_classes=2, alpha=1.0),
                "GaussianNB": config_pb2.GaussianConfig(
                    n_classes=2,
                ),
                "AdaptiveXGBoost": config_pb2.AdaptiveXGBoostConfig(
                    n_classes=2,
                    learning_rate=0.3,
                    max_depth=6,
                    max_window_size=1000,
                    min_window_size=0,
                    max_buffer=5,
                    pre_train=2,
                    detect_drift=True,
                    use_updater=True,
                    trees_per_train=1,
                    percent_update_trees=1.0,
                ),
                "AdaptiveLGBM": config_pb2.AdaptiveLGBMConfig(
                    n_classes=2,
                    learning_rate=0.3,
                    max_depth=6,
                    max_window_size=1000,
                    min_window_size=0,
                    max_buffer=5,
                    pre_train=2,
                    detect_drift=True,
                    use_updater=True,
                    trees_per_train=1,
                ),
                "LLAMAEmbedding": config_pb2.LLAMAEmbeddingModelConfig(),
                "LlamaText": config_pb2.LlamaTextConfig(),
                "ClipEmbedding": config_pb2.ClipEmbeddingConfig(),
                "RestAPIClient": config_pb2.RestAPIClientConfig(
                    max_retries=3,
                    connection_timeout=10,
                    max_request_time=30,
                ),
                "GRPCClient": config_pb2.GRPCClientConfig(
                    max_retries=3,
                    connection_timeout=10000,
                    max_request_time=30000,
                ),
            }
        )
        self.algo_config_mapping = frozendict(
            {
                "MStream": "mstream_config",
                "RCF": "rcf_config",
                "HST": "hst_config",
                "HoeffdingTreeClassifier": "hoeffding_classifier_config",
                "HoeffdingTreeRegressor": "hoeffding_regressor_config",
                "AMFClassifier": "amf_classifier_config",
                "AMFRegressor": "amf_regressor_config",
                "FFMClassifier": "ffm_classifier_config",
                "SGTClassifier": "sgt_classifier_config",
                "SGTRegressor": "sgt_regressor_config",
                "FFMRegressor": "ffm_regressor_config",
                "SNARIMAX": "snarimax_config",
                "ONNX": "onnx_config",
                "LeveragingBaggingClassifier": "leveraging_bagging_classifier_config",
                "HeteroLeveragingBaggingClassifier": "hetero_leveraging_bagging_classifier_config",
                "AdaBoostClassifier": "adaboost_classifier_config",
                "HeteroAdaBoostClassifier": "hetero_adaboost_classifier_config",
                "RandomSampler": "random_sampler_config",
                "Python": "python_config",
                "PythonEnsembleModel": "python_ensemble_config",
                "PreProcessor": "preprocessor_config",
                "NeuralNetwork": "nn_config",
                "ONN": "onn_config",
                "OVR": "ovr_model_selection_config",
                "BanditModelSelection": "bandit_model_selection_config",
                "ContextualBanditModelSelection": "contextual_bandit_model_selection_config",
                "RandomProjectionEmbedding": "random_projection_config",
                "EmbeddingModel": "embedding_model_config",
                "MultinomialNB": "multinomial_config",
                "GaussianNB": "gaussian_config",
                "AdaptiveXGBoost": "adaptive_xgboost_config",
                "AdaptiveLGBM": "adaptive_lgbm_config",
                "LLAMAEmbedding": "llama_embedding_config",
                "LlamaText": "llama_text_config",
                "ClipEmbedding": "clip_embedding_config",
                "RestAPIClient": "rest_api_client_config",
                "GRPCClient": "grpc_client_config",
            }
        )

    def get_default_parameters(self):
        parameters = {}
        for alg, config in self.default_configs.items():
            parameters[alg] = json_format.MessageToDict(config)
        return parameters

    def fill_config(self, conf: config_pb2.ModelConfig, parameters):
        new_config = json_format.ParseDict(
            parameters, copy.deepcopy(self.default_configs[conf.algorithm])
        )
        try:
            getattr(conf, self.algo_config_mapping[conf.algorithm]).CopyFrom(new_config)
        except Exception as e:
            raise Exception(f"Failed to match config: {conf.algorithm}") from e
        return conf


DefaultModelConfigs = _DefaultModelConfigs()

```

# env.py
-## Location -> root_directory.common
```python
from __future__ import annotations
from pydantic import Field  # noqa: TCH002
from pydantic_settings import BaseSettings, SettingsConfigDict


class _TurboMLConfig(BaseSettings):
    model_config = SettingsConfigDict(frozen=True)

    TURBOML_BACKEND_SERVER_ADDRESS: str = Field(default="http://localhost:8500")
    FEATURE_SERVER_ADDRESS: str = Field(default="grpc+tcp://localhost:8552")
    ARROW_SERVER_ADDRESS: str = Field(default="grpc+tcp://localhost:8502")

    def set_backend_server(self, value: str):
        object.__setattr__(self, "TURBOML_BACKEND_SERVER_ADDRESS", value)

    def set_feature_server(self, value: str):
        object.__setattr__(self, "FEATURE_SERVER_ADDRESS", value)

    def set_arrow_server(self, value: str):
        object.__setattr__(self, "ARROW_SERVER_ADDRESS", value)


# Global config object
CONFIG = _TurboMLConfig()  # type: ignore

```

# feature_engineering.py
-## Location -> root_directory.common
```python
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
                ## TODO: A method to drop features could be nice to have
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
                    ## TODO: should we pass as kwargs instead of relying on order?
                    *[row[col] for col in argument_names]
                ),
                axis=1,
            )
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
                "VARCHAR"
                if table_columns[col] == "character varying"
                else table_columns[col]
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
                "VARCHAR"
                if table_columns[col] == "character varying"
                else table_columns[col]
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

```

# internal.py
-## Location -> root_directory.common
```python
import itertools
from typing import Generator, Iterable
import typing
from pandas.core.base import DtypeObj
import pyarrow as pa
import pyarrow.flight
import pandas as pd
import logging
import threading

from .api import api
from tqdm import tqdm

from .models import InputSpec

logger = logging.getLogger(__name__)


class TurboMLResourceException(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


class TbPyArrow:
    """
    Utility class containing some shared methods and data for our
    PyArrow based data exchange.
    """

    @staticmethod
    def _input_schema(has_labels: bool) -> pa.Schema:
        label_schema = [("label", pa.float32())] if has_labels else []
        return pa.schema(
            [
                ("numeric", pa.list_(pa.float32())),
                ("categ", pa.list_(pa.int64())),
                ("text", pa.list_(pa.string())),
                ("image", pa.list_(pa.binary())),
                ("time_tick", pa.int32()),
                ("key", pa.string()),
            ]
            + label_schema
        )

    @staticmethod
    def arrow_table_to_pandas(
        table: pa.Table, to_pandas_opts: dict | None = None
    ) -> pd.DataFrame:
        default_opts = {"split_blocks": False, "date_as_object": False}
        to_pandas_opts = {**default_opts, **(to_pandas_opts or {})}
        return table.to_pandas(**to_pandas_opts)

    @staticmethod
    def df_to_table(df: pd.DataFrame, input_spec: InputSpec) -> pa.Table:
        # transform df to input form, where each column
        # is a list of values of the corresponding type
        input_df = pd.DataFrame()
        input_df["key"] = df[input_spec.key_field].astype("str").values
        input_df["time_tick"] = (
            0
            if input_spec.time_field in ["", None]
            else df[input_spec.time_field].astype("int32").values
        )
        input_df["numeric"] = (
            df[input_spec.numerical_fields].astype("float32").values.tolist()
        )
        input_df["categ"] = (
            df[input_spec.categorical_fields].astype("int64").values.tolist()
        )
        input_df["text"] = df[input_spec.textual_fields].astype("str").values.tolist()
        input_df["image"] = (
            df[input_spec.imaginal_fields].astype("bytes").values.tolist()
        )

        has_labels = input_spec.label_field is not None and input_spec.label_field in df
        if has_labels:
            input_df["label"] = df[input_spec.label_field].astype("float32").values

        return pa.Table.from_pandas(input_df, TbPyArrow._input_schema(has_labels))

    @staticmethod
    def wait_for_available(client: pyarrow.flight.FlightClient, timeout=10):
        try:
            client.wait_for_available(timeout=timeout)
        except pyarrow.flight.FlightUnauthenticatedError:
            # Server is up - wait_for_available() does not ignore auth errors
            pass

    @staticmethod
    def handle_flight_error(
        e: Exception, client: pyarrow.flight.FlightClient, allow_recovery=True
    ):
        if isinstance(e, pyarrow.flight.FlightUnavailableError):
            if not allow_recovery:
                raise TurboMLResourceException(
                    "Failed to initialize TurboMLArrowServer: Check logs for more details"
                ) from e

            # If the server is not available, we can try to start it
            api.post(endpoint="start_arrow_server")
            TbPyArrow.wait_for_available(client)
            return
        if isinstance(e, pyarrow.flight.FlightTimedOutError):
            if not allow_recovery:
                raise TurboMLResourceException(
                    "Flight server timed out: Check logs for more details"
                ) from e
            TbPyArrow.wait_for_available(client)
            return
        if isinstance(e, pyarrow.flight.FlightInternalError):
            raise TurboMLResourceException(
                f"Internal flight error: {e!r}. Check logs for more details"
            ) from e
        if isinstance(e, pyarrow.flight.FlightError):
            raise TurboMLResourceException(
                f"Flight server error: {e!r}. Check logs for more details"
            ) from e
        raise Exception(f"Unknown error: {e!r}") from e

    @staticmethod
    def _put_and_retry(
        client: pyarrow.flight.FlightClient,
        upload_descriptor: pyarrow.flight.FlightDescriptor,
        options: pyarrow.flight.FlightCallOptions,
        input_table: pa.Table,
        can_retry: bool = True,
        max_chunksize: int = 1024,
        epochs: int = 1,
    ) -> None:
        try:
            writer, _ = client.do_put(
                upload_descriptor, input_table.schema, options=options
            )
            TbPyArrow._write_in_chunks(
                writer, input_table, max_chunksize=max_chunksize, epochs=epochs
            )
            writer.close()
        except Exception as e:
            TbPyArrow.handle_flight_error(e, client, can_retry)
            return TbPyArrow._put_and_retry(
                client,
                upload_descriptor,
                options,
                input_table,
                can_retry=False,
                max_chunksize=max_chunksize,
                epochs=epochs,
            )

    @staticmethod
    def _exchange_and_retry(
        client: pyarrow.flight.FlightClient,
        upload_descriptor: pyarrow.flight.FlightDescriptor,
        options: pyarrow.flight.FlightCallOptions,
        input_table: pa.Table,
        can_retry: bool = True,
        max_chunksize: int = 1024,
    ) -> pa.Table:
        try:
            writer, reader = client.do_exchange(upload_descriptor, options=options)
            writer.begin(input_table.schema)
            write_event = threading.Event()
            writer_thread = threading.Thread(
                target=TbPyArrow._write_in_chunks,
                args=(writer, input_table),
                kwargs={
                    "max_chunksize": max_chunksize,
                    "write_event": write_event,
                },
            )
            writer_thread.start()
            write_event.wait()
            read_result = TbPyArrow._read_in_chunks(reader)
            writer_thread.join()
            writer.close()
            return read_result
        except Exception as e:
            TbPyArrow.handle_flight_error(e, client, can_retry)
            return TbPyArrow._exchange_and_retry(
                client,
                upload_descriptor,
                options,
                input_table,
                can_retry=False,
                max_chunksize=max_chunksize,
            )

    @staticmethod
    def _write_in_chunks(
        writer: pyarrow.flight.FlightStreamWriter,
        input_table: pa.Table,
        write_event: threading.Event | None = None,
        max_chunksize: int = 1024,
        epochs: int = 1,
    ) -> None:
        total_rows = input_table.num_rows
        n_chunks = total_rows // max_chunksize + 1
        logger.info(f"Starting to upload data... Total rows: {total_rows}")

        for epoch in range(epochs):
            with tqdm(
                total=n_chunks, desc="Progress", unit="chunk", unit_scale=True
            ) as pbar:
                if epochs > 1:
                    pbar.set_postfix(epoch=epoch + 1)

                for start in range(0, total_rows, max_chunksize):
                    chunk_table = input_table.slice(
                        start, min(max_chunksize, total_rows - start)
                    )
                    writer.write_table(chunk_table)
                    if start == 0 and write_event:
                        write_event.set()
                    pbar.update(1)

        writer.done_writing()
        logger.info("Completed data upload.")

    @staticmethod
    def _read_in_chunks(reader: pyarrow.flight.FlightStreamReader) -> pa.Table:
        batches = []
        while True:
            try:
                chunk = reader.read_chunk()
                batches.append(chunk.data)
            except StopIteration:
                break
        return pa.Table.from_batches(batches) if batches else None


class TbPandas:
    @staticmethod
    def fill_nans_with_default(series: pd.Series):
        if series.isna().any():
            default = TbPandas.default_for_type(series.dtype)
            return series.fillna(value=default)
        return series

    @staticmethod
    def default_for_type(dtype: DtypeObj):
        if pd.api.types.is_numeric_dtype(dtype):
            return 0
        if pd.api.types.is_string_dtype(dtype):
            return ""
        if pd.api.types.is_bool_dtype(dtype):
            return False
        if pd.api.types.is_datetime64_dtype(dtype):
            return pd.Timestamp("1970-01-01")
        raise ValueError(f"Unsupported dtype: {dtype}")


T = typing.TypeVar("T")


class TbItertools:
    @staticmethod
    def chunked(iterable: Iterable[T], n: int) -> Generator[list[T], None, None]:
        """Yield successive n-sized chunks from iterable."""
        iterator = iter(iterable)
        while True:
            chunk = list(itertools.islice(iterator, n))
            if not chunk:
                break
            yield list(chunk)

```

# llm.py
-## Location -> root_directory.common
```python
import logging
import time

from turboml.common.types import GGUFModelId
from .api import api
from .models import (
    LlamaServerRequest,
    LlamaServerResponse,
    HFToGGUFRequest,
    ModelAcquisitionJob,
)

logger = logging.getLogger("turboml.llm")


def acquire_hf_model_as_gguf(
    hf_repo_id: str,
    model_type: HFToGGUFRequest.GGUFType = HFToGGUFRequest.GGUFType.AUTO,
    select_gguf_file: str | None = None,
) -> GGUFModelId:
    """
    Attempts to acquires a model from the Hugging Face repository and convert
    it to the GGUF format. The model is then stored in the TurboML system, and
    the model key is returned.
    """
    req = HFToGGUFRequest(
        hf_repo_id=hf_repo_id,
        model_type=model_type,
        select_gguf_file=select_gguf_file,
    )
    acq_resp = api.post("acquire_hf_model_as_gguf", json=req.model_dump()).json()
    status_endpoint = acq_resp["status_endpoint"]
    last_status = None
    last_progress = None

    while True:
        job_info = api.get(status_endpoint.lstrip("/")).json()
        job_info = ModelAcquisitionJob(**job_info)

        status = job_info.status
        progress = job_info.progress_message or "No progress info"

        if status != last_status or progress != last_progress:
            logger.info(f"[hf-acquisition] Status: {status}, Progress: {progress}")
            last_status = status
            last_progress = progress

        if status == "completed":
            gguf_id = job_info.gguf_id
            if not gguf_id:
                raise AssertionError("GGUF ID not found in job_info")
            logger.info(f"[hf-acquisition] Acquisition Done, gguf_id = {gguf_id}")
            return GGUFModelId(gguf_id)
        elif status == "failed":
            error_msg = job_info.error_message or "Unknown error"
            raise RuntimeError(f"HF->GGUF acquisition failed: {error_msg}")

        time.sleep(5)


def spawn_llm_server(req: LlamaServerRequest) -> LlamaServerResponse:
    """
    If source_type=HUGGINGFACE, we do the async acquisition under the hood,
    but we poll until it’s done. Then we do the normal /model/openai call.
    """
    if req.source_type == LlamaServerRequest.SourceType.HUGGINGFACE:
        if not req.hf_spec:
            raise ValueError("hf_spec is required for source_type=HUGGINGFACE")
        gguf_id = acquire_hf_model_as_gguf(
            hf_repo_id=req.hf_spec.hf_repo_id,
            model_type=req.hf_spec.model_type,
            select_gguf_file=req.hf_spec.select_gguf_file,
        )
        req.source_type = LlamaServerRequest.SourceType.GGUF_ID
        req.gguf_id = gguf_id

    resp = api.post("model/openai", json=req.model_dump())
    return LlamaServerResponse(**resp.json())


def stop_llm_server(server_id: str):
    """
    To DE_acquire(iLETE /model/openai/{server_id}
    """
    api.delete(f"model/openai/{server_id}")

```

# ml_algs.py
-## Location -> root_directory.common
```python
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
        ## TODO: Shouldn't we be retrieving the latest model snapshot from the server?
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
        ## TODO: Add this assertion for type narrowing. Currently it fails because preprocessors don't inherit from PreProcessor.
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
    def validate_layers(cls, layers):
        if len(layers) == 0:
            raise Exception("layers must be non empty")

        ## TODO other layer checks
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

```

# models.py
-## Location -> root_directory.common
```python
from __future__ import annotations
from copy import deepcopy
import re
from typing import (
    Any,
    Optional,
    Tuple,
    Literal,
    Type,
    List,
    Union,
    Annotated,
    TYPE_CHECKING,
)
from enum import StrEnum, Enum, auto
from datetime import datetime, timezone

from google.protobuf.descriptor import FieldDescriptor
from pydantic import (
    Base64Bytes,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    StringConstraints,
    create_model,
    field_serializer,
    field_validator,
    validator,
    model_validator,
    StrictBool,
)
from pandas._libs.tslibs.timestamps import Timestamp
import pandas as pd
import numpy as np

from turboml.common import dataloader


from .sources import PostgresSource, FileSource  # noqa: TCH001

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo


TurboMLResourceIdentifier = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-zA-Z0-9_-]+$",
        ## TODO: We need to ensure we're using this type everywhere identifiers are accepted (url/query params!)
        # Otherwise this would break APIs.
        # to_lower=True,
    ),
]
DatasetId = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")]

# Since our dataset fields are used as protobuf message fields, we need to ensure they're valid
# protobuf field names. This means they must start with an underscore ('_') or a letter (a-z, A-Z),
# followed by alphanumeric characters or underscores.
DatasetField = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")]

SQLIden = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")]


class SchemaType(StrEnum):
    PROTOBUF = "PROTOBUF"


class KafkaConnectDatasetRegistrationRequest(BaseModel):
    dataset_id: DatasetId
    source: Union[FileSource, PostgresSource]
    key_field: str

    @field_validator("dataset_id")
    def check_dataset_id(cls, v: str):
        # We use `.` to partition dataset_ids into namespaces, so we don't allow it in dataset names
        # `-` is used as another internal delimiter, so we don't allow it either.
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("Invalid dataset name")
        return v


class Datatype(StrEnum):
    """
    Data types supported by the TurboML platform, corresponding to protobuf types.
    """

    INT32 = auto()
    INT64 = auto()
    FLOAT = auto()
    DOUBLE = auto()
    STRING = auto()
    BOOL = auto()
    BYTES = auto()

    ## TODO: we support some more types for floats and datetimes...

    def to_protobuf_type(self) -> str:
        return str(self).lower()

    @staticmethod
    def from_proto_field_descriptor_type(type_: int) -> Datatype:
        match type_:
            case FieldDescriptor.TYPE_INT32:
                return Datatype.INT32
            case FieldDescriptor.TYPE_INT64:
                return Datatype.INT64
            case FieldDescriptor.TYPE_FLOAT:
                return Datatype.FLOAT
            case FieldDescriptor.TYPE_DOUBLE:
                return Datatype.DOUBLE
            case FieldDescriptor.TYPE_STRING:
                return Datatype.STRING
            case FieldDescriptor.TYPE_BOOL:
                return Datatype.BOOL
            case FieldDescriptor.TYPE_BYTES:
                return Datatype.BYTES
            case _:
                raise ValueError(f"Unsupported protobuf type: {type_}")

    def to_pandas_dtype(self) -> str:
        """Convert TurboML datatype to pandas dtype that works with astype()"""
        match self:
            case Datatype.INT32:
                return "int32"
            case Datatype.INT64:
                return "int64"
            case Datatype.FLOAT:
                return "float32"
            case Datatype.DOUBLE:
                return "float64"
            case Datatype.STRING:
                return "string"
            case Datatype.BOOL:
                return "bool"
            case Datatype.BYTES:
                return "bytes"
            case _:
                raise ValueError(f"Unsupported datatype for pandas conversion: {self}")

    @staticmethod
    def from_pandas_column(column: pd.Series) -> Datatype:
        match column.dtype:
            case np.int32:
                return Datatype.INT32
            case np.int64:
                return Datatype.INT64
            case np.float32:
                return Datatype.FLOAT
            case np.float64:
                return Datatype.DOUBLE
            case np.bool_:
                return Datatype.BOOL
            case np.bytes_:
                return Datatype.BYTES
            case "string":
                return Datatype.STRING
            case np.object_:
                # At this point we're not sure of the type: pandas by default
                # interprets both `bytes` and `str` into `object_` columns
                proto_dtype = Datatype._infer_pd_object_col_type(column)
                if proto_dtype is None:
                    raise ValueError(f"Unsupported dtype: {column.dtype}")
                return proto_dtype
            case _:
                raise ValueError(f"Unsupported dtype: {column.dtype}")

    @staticmethod
    def _infer_pd_object_col_type(column: pd.Series) -> Optional[Datatype]:
        first_non_na_idx = column.first_valid_index()
        if first_non_na_idx is None:
            return None
        try:
            if (
                isinstance(column.loc[first_non_na_idx], str)
                and column.astype(str) is not None
            ):
                return Datatype.STRING
        except UnicodeDecodeError:
            pass

        try:
            if (
                isinstance(column.loc[first_non_na_idx], bytes)
                and column.astype(bytes) is not None
            ):
                return Datatype.BYTES
        except TypeError:
            pass

        return None


class DatasetSchema(BaseModel):
    fields: dict[TurboMLResourceIdentifier, Datatype]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{k}: {v}' for k, v in self.fields.items())})"

    @staticmethod
    def from_pd(df: pd.DataFrame) -> DatasetSchema:
        fields = {}
        for column_name in df.columns:
            column = df[column_name]
            assert isinstance(column, pd.Series)
            fields[column_name] = Datatype.from_pandas_column(column)
        return DatasetSchema(fields=fields)

    @staticmethod
    def from_protobuf_schema(schema: str, message_name: str | None) -> DatasetSchema:
        cls = dataloader.get_protobuf_class(
            schema=schema,
            message_name=message_name,
        )
        if cls is None:
            raise ValueError(
                f"No matching protobuf message found for message={message_name}, schema={schema}"
            )
        columns = {}
        for field in cls.DESCRIPTOR.fields:
            name = field.name
            proto_type = Datatype.from_proto_field_descriptor_type(field.type)
            columns[name] = proto_type
        return DatasetSchema(fields=columns)

    def to_protobuf_schema(self, message_name: str) -> str:
        NEWLINE = "\n"

        def column_to_field_decl(cname: str, ctype: Datatype, idx: int) -> str:
            return f"optional {ctype.to_protobuf_type()} {cname} = {idx};"

        field_decls = map(
            column_to_field_decl,
            self.fields.keys(),
            self.fields.values(),
            range(1, len(self.fields) + 1),
        )
        return f"""
syntax = "proto2";
message {message_name} {{
{NEWLINE.join(field_decls)}
}}
"""


class DatasetRegistrationRequest(BaseModel):
    class SchemaFromRegistry(BaseModel):
        type_: Literal["registry"] = "registry"
        kind: Literal["protobuf"] = "protobuf"
        message_name: str

    class ExplicitSchema(DatasetSchema):
        type_: Literal["explicit"] = "explicit"

        @staticmethod
        def from_pd(df: pd.DataFrame) -> DatasetRegistrationRequest.ExplicitSchema:
            ds = DatasetSchema.from_pd(df)
            return DatasetRegistrationRequest.ExplicitSchema(**ds.model_dump())

    dataset_id: DatasetId
    data_schema: Union[SchemaFromRegistry, ExplicitSchema] = Field(
        discriminator="type_"
    )
    key_field: str


class RegisteredSchema(BaseModel):
    id: int
    schema_type: SchemaType
    schema_body: str
    message_name: str
    native_schema: DatasetSchema


class DatasetRegistrationResponse(BaseModel):
    registered_schema: RegisteredSchema


class DatasetSpec(BaseModel):
    dataset_id: DatasetId
    key: str


class DbColumn(BaseModel):
    name: str
    dtype: str


class Dataset(BaseModel):
    class JoinInformation(BaseModel):
        sources: tuple[DatasetId, DatasetId]
        joined_on_column_pairs: list[tuple[str, str]]
        prefixes: tuple[SQLIden, SQLIden]

    class Metadata(BaseModel):
        kafka_topic: str
        input_pb_message_name: str
        risingwave_source: str
        risingwave_view: str
        join_information: Optional[Dataset.JoinInformation] = None

    feature_version: int = 0
    sink_version: int = 0
    table_columns: list[DbColumn]
    key: str
    source_type: str
    message_name: str
    file_proto: str
    sql_feats: dict[str, dict] = Field(default_factory=dict)  ## TODO: type
    agg_feats: dict[str, dict] = Field(default_factory=dict)  ## TODO: type
    udf_feats: dict[str, dict] = Field(default_factory=dict)  ## TODO: type
    udaf_feats: dict[str, dict] = Field(default_factory=dict)  ## TODO: type
    agg_cols_indexes: list[str] = Field(default_factory=list)
    meta: Metadata = Field()
    timestamp_fields: dict[str, str] = Field(default_factory=dict)
    drifts: list[DataDrift] = Field(default_factory=list)
    ibis_feats: list[dict] = Field(default_factory=list)


class LabelAssociation(BaseModel):
    dataset_id: DatasetId
    field: str


class LearnerConfig(BaseModel):
    predict_workers: int
    update_batch_size: int
    synchronization_method: str


class ModelParams(BaseModel):
    # Silence pydantic warning about protected namespace
    model_config = ConfigDict(protected_namespaces=())

    model_configs: list[dict]

    ## TODO: We should replace these with InputSpec
    label: LabelAssociation
    numerical_fields: list[str]
    categorical_fields: list[str]
    textual_fields: list[str]
    imaginal_fields: list[str]
    time_field: Optional[str]


class MLModellingRequest(ModelParams):
    id: TurboMLResourceIdentifier
    dataset_id: DatasetId

    # Use a pretrained model as the initial state
    initial_model_id: Optional[str] = None

    learner_config: Optional[LearnerConfig] = None
    predict_only: bool = False


class ModelConfigStorageRequest(BaseModel):
    id: TurboMLResourceIdentifier
    version: TurboMLResourceIdentifier
    initial_model_key: str | None
    params: ModelParams | None

    @model_validator(mode="after")
    def validate_model_params(self):
        if not self.initial_model_key and not self.params:
            raise ValueError(
                "Either initial_model_key or model_params must be provided"
            )
        return self


class DataDriftType(StrEnum):
    UNIVARIATE = "univariate"
    MULTIVARIATE = "multivariate"


class DataDrift(BaseModel):
    label: Optional[TurboMLResourceIdentifier]
    numerical_fields: list[str]

    ## TODO: we could do some validations on fields for improved UX
    # (verify that they're present and are numeric)


class DriftQuery(BaseModel):
    limit: int = 100
    start_timestamp: datetime
    end_timestamp: datetime


class DataDriftQuery(DataDrift, DriftQuery):
    pass


class TargetDriftQuery(DriftQuery):
    pass


class DriftScores(BaseModel):
    scores: List[float]
    timestamps: List[int]

    @validator("timestamps", pre=True, each_item=True)
    def convert_timestamp_to_epoch_microseconds(
        cls, value: Union[Timestamp, Any]
    ) -> int:
        if isinstance(value, Timestamp):
            return int(value.timestamp() * 1_000_000)
        return int(float(value) * 1_000_000)


class VenvSpec(BaseModel):
    venv_name: str
    lib_list: list[str]

    @field_validator("venv_name")
    def ensure_venv_name_is_not_funny(cls, v):
        # Restrict venv names to alphanumeric
        safe_name_regex = r"^[a-zA-Z0-9_]+$"
        if not re.match(safe_name_regex, v):
            raise ValueError("Venv name must be alphanumeric")
        return v


class AddPythonClassRequest(BaseModel):
    class PythonClassValidationType(StrEnum):
        NONE = auto()
        MODEL_CLASS = auto()
        MODULE = auto()

    obj: Base64Bytes
    name: str
    # NOTE: No validations on the backend for now
    validation_type: Optional[PythonClassValidationType] = (
        PythonClassValidationType.NONE
    )


class HFToGGUFRequest(BaseModel):
    class GGUFType(StrEnum):
        F32 = "f32"
        F16 = "f16"
        BF16 = "bf16"
        QUANTIZED_8_0 = "q8_0"
        AUTO = "auto"

    hf_repo_id: str
    model_type: GGUFType = Field(default=GGUFType.AUTO)
    select_gguf_file: Optional[str] = None

    @field_validator("hf_repo_id")
    def validate_hf_repo_id(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_\.-]+$", v):
            raise ValueError("Invalid HF repo id")
        return v

    class Config:
        protected_namespaces = ()


class ModelAcquisitionJob(BaseModel):
    class AcquisitionStatus(StrEnum):
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"
        FAILED = "failed"

    job_id: str
    status: AcquisitionStatus = AcquisitionStatus.PENDING
    hf_repo_id: str
    model_type: str
    select_gguf_file: Optional[str] = None
    gguf_id: Optional[str] = None
    error_message: Optional[str] = None
    progress_message: Optional[str] = None


class LlamaServerRequest(BaseModel):
    class SourceType(StrEnum):
        HUGGINGFACE = "huggingface"
        GGUF_ID = "gguf_id"

    class HuggingFaceSpec(HFToGGUFRequest):
        pass

    class ServerParams(BaseModel):
        threads: int = -1
        seed: int = -1
        context_size: int = 0
        flash_attention: bool = False

    source_type: SourceType
    gguf_id: Optional[str] = None
    hf_spec: Optional[HuggingFaceSpec] = None
    server_params: ServerParams = Field(default_factory=ServerParams)

    @field_validator("source_type", mode="before")
    def accept_string_for_enum(cls, v):
        if isinstance(v, str):
            return cls.SourceType(v)
        return v

    @model_validator(mode="after")
    def validate_model_source(self):
        if self.source_type == self.SourceType.HUGGINGFACE and not self.hf_spec:
            raise ValueError("Huggingface model source requires hf_spec")
        if self.source_type == self.SourceType.GGUF_ID and not self.gguf_id:
            raise ValueError("GGUF model source requires gguf_id")
        return self


class LlamaServerResponse(BaseModel):
    server_id: str
    server_relative_url: str


class HFToGGUFResponse(BaseModel):
    gguf_id: str


class MetricRegistrationRequest(BaseModel):
    ## TODO: metric types should be enum (incl custom metrics)
    metric: str


class FeatureMetadata(BaseModel):
    author: int
    introduced_in_version: int
    created_at: str  # As datetime is not json serializable
    datatype: str  # Pandas type as SDK is python


class SqlFeatureSpec(BaseModel):
    feature_name: str
    sql_spec: str


class AggregateFeatureSpec(BaseModel):
    feature_name: str
    column: str
    aggregation_function: str
    group_by_columns: list[str]
    interval: str
    timestamp_column: str


class UdafFeatureSpec(BaseModel):
    feature_name: str
    arguments: list[str]
    function_name: str
    group_by_columns: list[str]
    timestamp_column: str
    interval: str


class CustomMetric(BaseModel):
    metric_name: str
    metric_spec: dict


class RwEmbeddedUdafFunctionSpec(BaseModel):
    input_types: list[str]
    output_type: str
    function_file_contents: str


class ExternalUdafFunctionSpec(BaseModel):
    obj: Base64Bytes


class UdafFunctionSpec(BaseModel):
    name: Annotated[
        str, StringConstraints(min_length=1, pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    ]
    libraries: list[str]
    spec: RwEmbeddedUdafFunctionSpec | ExternalUdafFunctionSpec


class UdfFeatureSpec(BaseModel):
    feature_name: str
    arguments: list[str]
    function_name: str
    libraries: list[str]
    is_rich_function: bool = False
    io_threads: Optional[int] = None
    class_name: Optional[str] = None
    initializer_arguments: list[str]


class UdfFunctionSpec(BaseModel):
    name: Annotated[
        str, StringConstraints(min_length=1, pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    ]
    input_types: list[str]
    output_type: str
    libraries: list[str]
    function_file_contents: str
    is_rich_function: bool = False
    io_threads: Optional[int] = None
    class_name: Optional[str] = None
    initializer_arguments: list[str]

    @model_validator(mode="after")
    def validate_rich_function(self):
        if self.is_rich_function and not self.class_name:
            raise ValueError("class_name is required for rich functions")
        return self


class IbisFeatureSpec(BaseModel):
    dataset_id: DatasetId
    udfs_spec: list[UdfFunctionSpec]
    encoded_table: str


class FeatureGroup(BaseModel):
    feature_version: int = 0
    key_field: str
    meta: dict = Field(default_factory=dict)
    udfs_spec: list[UdfFunctionSpec]
    primary_source_name: str


class BackEnd(Enum):
    Risingwave = auto()
    Flink = auto()


class ApiKey(BaseModel):
    id: int
    "Unique identifier for the key"
    suffix: str
    "Last 8 characters of the key"
    expire_at: Optional[datetime]
    label: Optional[str]
    created_at: datetime
    revoked_at: Optional[datetime]


class FetchFeatureRequest(BaseModel):
    dataset_id: DatasetId
    limit: int


class FeatureMaterializationRequest(BaseModel):
    dataset_id: DatasetId
    sql_feats: list[SqlFeatureSpec] = Field(default_factory=list)
    agg_feats: list[AggregateFeatureSpec] = Field(default_factory=list)
    udf_feats: list[UdfFeatureSpec] = Field(default_factory=list)
    udaf_feats: list[UdafFeatureSpec] = Field(default_factory=list)
    ibis_feats: Optional[IbisFeatureSpec] = None


class FeaturePreviewRequest(FeatureMaterializationRequest):
    limit: int = 10


class IbisFeatureMaterializationRequest(BaseModel):
    feature_group_name: Annotated[
        str,
        StringConstraints(min_length=1, pattern=r"^[a-z]([a-z0-9_]{0,48}[a-z0-9])?$"),
    ]
    key_field: str
    udfs_spec: list[UdfFunctionSpec]
    backend: BackEnd
    encoded_table: str
    primary_source_name: str

    @field_serializer("backend")
    def serialize_backend(self, backend: BackEnd, _info):
        return backend.value


class TimestampQuery(BaseModel):
    column_name: str
    timestamp_format: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str


class Oauth2StartResponse(BaseModel):
    auth_uri: str


class NewApiKeyRequest(BaseModel):
    expire_at: Optional[datetime]
    label: str

    @field_validator("expire_at")
    def validate_expire_at(cls, v):
        if v is not None and v < datetime.now():
            raise ValueError("expire_at must be in the future")
        return v


class NewApiKeyResponse(BaseModel):
    key: str
    expire_at: Optional[datetime]


def _partial_model(model: Type[BaseModel]):
    """
    Decorator to create a partial model, where all fields are optional.
    Useful for PATCH requests, where we want to allow partial updates
    and the models may be derived from the original model.
    """

    def make_field_optional(
        field: FieldInfo, default: Any = None
    ) -> Tuple[Any, FieldInfo]:
        new = deepcopy(field)
        new.default = default
        new.annotation = Optional[field.annotation]  # type: ignore
        return new.annotation, new

    fields = {
        field_name: make_field_optional(field_info)
        for field_name, field_info in model.model_fields.items()
    }
    return create_model(  # type: ignore
        f"Partial{model.__name__}",
        __doc__=model.__doc__,
        __base__=model,
        __module__=model.__module__,
        **fields,  # type: ignore
    )


@_partial_model
class ApiKeyPatchRequest(BaseModel):
    label: Optional[str]
    expire_at: Optional[datetime]

    @field_validator("expire_at", mode="before")
    def validate_expire_at(cls, v):
        if v is None:
            return None
        v = datetime.fromtimestamp(v, tz=timezone.utc)
        if v < datetime.now(timezone.utc):
            raise ValueError("expire_at must be in the future")
        return v


class User(BaseModel):
    id: int
    username: str
    email: str | None = None


class NamespaceAcquisitionRequest(BaseModel):
    namespace: Annotated[
        str,
        StringConstraints(
            min_length=1, max_length=32, pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$"
        ),
    ]


class UserManager:
    def __init__(self):
        # Cache username -> user
        ## TODO: we should change this to user_id -> user, along
        # with changing the API/auth to use user_id instead of username.
        # This is part of our agreement to use username for display purposes only,
        # as well as most of our other resources using IDs.
        self.user_cache = {}


class TosResponse(BaseModel):
    version: int
    format: str  # text/plain or text/html
    content: str


class TosAcceptanceRequest(BaseModel):
    version: int


class InputSpec(BaseModel):
    key_field: str
    time_field: Optional[str]
    numerical_fields: list[str]
    categorical_fields: list[str]
    textual_fields: list[str]
    imaginal_fields: list[str]

    label_field: str


class ModelMetadata(BaseModel):
    # NOTE: We could use a proto-derived pydantic model here
    # (with `so1n/protobuf_to_pydantic`) but on our last attempt
    # the generated models were problematic for `oneof` proto fields and the hacks
    # weren't worth it. We can still revisit this in the future.
    # Ref: https://github.com/so1n/protobuf_to_pydantic/issues/31
    process_config: dict
    offset: str
    input_db_source: str
    input_db_columns: list[DbColumn]
    metrics: list[str]
    label_association: LabelAssociation
    drift: str

    def get_input_spec(self) -> InputSpec:
        key_field = self.process_config["inputConfig"]["keyField"]
        time_field = self.process_config["inputConfig"].get("time_tick", None)
        numerical_fields = list(self.process_config["inputConfig"].get("numerical", []))
        categorical_fields = list(
            self.process_config["inputConfig"].get("categorical", [])
        )
        textual_fields = list(self.process_config["inputConfig"].get("textual", []))
        imaginal_fields = list(self.process_config["inputConfig"].get("imaginal", []))
        # label_field = self.label_association.field if self.label_association else None
        return InputSpec(
            key_field=key_field,
            time_field=time_field,
            numerical_fields=numerical_fields,
            categorical_fields=categorical_fields,
            textual_fields=textual_fields,
            imaginal_fields=imaginal_fields,
            label_field=self.label_association.field,
        )


class ModelInfo(BaseModel):
    metadata: ModelMetadata
    endpoint_paths: list[str]


class StoredModel(BaseModel):
    name: str
    version: str
    stored_size: int
    created_at: datetime


class ProcessMeta(BaseModel):
    caller: str
    namespace: str
    job_id: str


class ProcessInfo(BaseModel):
    pid: int
    cmd: list[str]
    stdout_path: str
    stderr_path: str
    # For turboml jobs
    meta: Optional[ProcessMeta]
    restart: bool
    stopped: bool = False

    @field_validator("cmd", mode="before")
    def tolerate_string_cmd(cls, v):  # For backward compatibility
        if isinstance(v, str):
            return v.split()
        return v


class ProcessOutput(BaseModel):
    stdout: str
    stderr: str


class ProcessPatchRequest(BaseModel):
    action: Literal["kill", "restart"]


class ModelPatchRequest(BaseModel):
    action: Literal["pause", "resume"]


class ModelDeleteRequest(BaseModel):
    delete_output_topic: StrictBool


# Moved from ml_algs
class SupervisedAlgorithms(StrEnum):
    HoeffdingTreeClassifier = "HoeffdingTreeClassifier"
    HoeffdingTreeRegressor = "HoeffdingTreeRegressor"
    AMFClassifier = "AMFClassifier"
    AMFRegressor = "AMFRegressor"
    FFMClassifier = "FFMClassifier"
    FFMRegressor = "FFMRegressor"
    SGTClassifier = "SGTClassifier"
    SGTRegressor = "SGTRegressor"
    SNARIMAX = "SNARIMAX"
    LeveragingBaggingClassifier = "LeveragingBaggingClassifier"
    HeteroLeveragingBaggingClassifier = "HeteroLeveragingBaggingClassifier"
    AdaBoostClassifier = "AdaBoostClassifier"
    HeteroAdaBoostClassifier = "HeteroAdaBoostClassifier"
    RandomSampler = "RandomSampler"
    NeuralNetwork = "NeuralNetwork"
    ONN = "ONN"
    Python = "Python"
    OVR = "OVR"
    BanditModelSelection = "BanditModelSelection"
    ContextualBanditModelSelection = "ContextualBanditModelSelection"
    RandomProjectionEmbedding = "RandomProjectionEmbedding"
    EmbeddingModel = "EmbeddingModel"
    MultinomialNB = "MultinomialNB"
    GaussianNB = "GaussianNB"
    AdaptiveXGBoost = "AdaptiveXGBoost"
    AdaptiveLGBM = "AdaptiveLGBM"
    LLAMAEmbedding = "LLAMAEmbedding"
    LlamaText = "LlamaText"
    RestAPIClient = "RestAPIClient"
    ClipEmbedding = "ClipEmbedding"
    PythonEnsembleModel = "PythonEnsembleModel"
    GRPCClient = "GRPCClient"


class UnsupervisedAlgorithms(StrEnum):
    MStream = "MStream"
    RCF = "RCF"
    HST = "HST"
    ONNX = "ONNX"


class EvaluationMetrics(StrEnum):
    WindowedAUC = "WindowedAUC"
    WindowedMAE = "WindowedMAE"
    WindowedMSE = "WindowedMSE"
    WindowedRMSE = "WindowedRMSE"
    WindowedAccuracy = "WindowedAccuracy"


# Timestamp conversion and support for duckdb and risingwave
class TimestampRealType(StrEnum):
    epoch_seconds = "epoch_seconds"
    epoch_milliseconds = "epoch_milliseconds"


class RisingWaveVarcharType(StrEnum):
    YYYY_MM_DD = "YYYY MM DD"
    YYYY_MM_DD_HH24_MI_SS_US = "YYYY-MM-DD HH24:MI:SS.US"
    YYYY_MM_DD_HH12_MI_SS_US = "YYYY-MM-DD HH12:MI:SS.US"
    YYYY_MM_DD_HH12_MI_SS_MS = "YYYY-MM-DD HH12:MI:SS.MS"
    YYYY_MM_DD_HH24_MI_SS_MS = "YYYY-MM-DD HH24:MI:SS.MS"
    YYYY_MM_DD_HH24_MI_SSTZH_TZM = "YYYY-MM-DD HH24:MI:SSTZH:TZM"
    YYYY_MM_DD_HH12_MI_SSTZH_TZM = "YYYY-MM-DD HH12:MI:SSTZH:TZM"


class DuckDbVarcharType(StrEnum):
    YYYY_MM_DD = "%x"
    YYYY_MM_DD_HH24_MI_SS_US = "%x %H.%f"
    YYYY_MM_DD_HH12_MI_SS_US = "%x %I.%f %p"
    YYYY_MM_DD_HH12_MI_SS_MS = "%x %I.%g %p"
    YYYY_MM_DD_HH24_MI_SS_MS = "%x %H.%g"
    YYYY_MM_DD_HH24_MI_SSTZH_TZM = "%x %H.%g %z"
    YYYY_MM_DD_HH12_MI_SSTZH_TZM = "%x %I.%g %p %z"


class Evaluations(BaseModel):
    class ModelOutputType(StrEnum):
        PREDICTED_CLASS = "predicted_class"
        SCORE = "score"

    model_names: list
    metric: str
    filter_expression: str = ""
    window_size: PositiveInt = 1000
    limit: PositiveInt = 100
    is_web: bool = False
    output_type: Optional[ModelOutputType] = ModelOutputType.SCORE

    class Config:
        protected_namespaces = ()


class ModelScores(BaseModel):
    scores: List[float]
    timestamps: List[int]
    page: int
    next_page: Optional[List[int]] = None

    @validator("timestamps", pre=True, each_item=True)
    def convert_timestamp_to_epoch_microseconds(
        cls, value: Union[Timestamp, Any]
    ) -> int:
        if isinstance(value, Timestamp):
            return int(value.timestamp() * 1_000_000)
        return int(float(value) * 1_000_000)


class KafkaTopicInfo(BaseModel):
    name: str
    partitions: int
    replication_factor: int
    num_messages: int


class KafkaTopicSettings(BaseModel):
    ## TODO(maniktherana): prune as much Optional as we can to get stronger types
    compression_type: Optional[str] = None
    leader_replication_throttled_replicas: Optional[str] = None
    remote_storage_enable: Optional[bool] = None
    message_downconversion_enable: Optional[bool] = None
    min_insync_replicas: Optional[int] = None
    segment_jitter_ms: Optional[int] = None
    local_retention_ms: Optional[int] = None
    cleanup_policy: Optional[str] = None
    flush_ms: Optional[int] = None
    follower_replication_throttled_replicas: Optional[str] = None
    segment_bytes: Optional[int] = None
    retention_ms: Optional[int] = None
    flush_messages: Optional[int] = None
    message_format_version: Optional[str] = None
    max_compaction_lag_ms: Optional[int] = None
    file_delete_delay_ms: Optional[int] = None
    max_message_bytes: Optional[int] = None
    min_compaction_lag_ms: Optional[int] = None
    message_timestamp_type: Optional[str] = None
    local_retention_bytes: Optional[int] = None
    preallocate: Optional[bool] = None
    index_interval_bytes: Optional[int] = None
    min_cleanable_dirty_ratio: Optional[float] = None
    unclean_leader_election_enable: Optional[bool] = None
    retention_bytes: Optional[int] = None
    delete_retention_ms: Optional[int] = None
    message_timestamp_after_max_ms: Optional[int] = None
    message_timestamp_before_max_ms: Optional[int] = None
    segment_ms: Optional[int] = None
    message_timestamp_difference_max_ms: Optional[int] = None
    segment_index_bytes: Optional[int] = None


class DetailedKafkaTopicInfo(BaseModel):
    name: str
    partitions: int
    replication_factor: int
    urp: int
    in_sync_replicas: int
    total_replicas: int
    cleanup_policy: str
    segment_size: int
    segment_count: int


class KafkaTopicConsumer(BaseModel):
    group_id: str
    active_consumers: int
    state: str


class SchemaInfo(BaseModel):
    subject: str
    id: int
    type: str
    version: int


class DetailedSchemaInfo(BaseModel):
    subject: str
    latest_version: int
    latest_id: int
    latest_type: str
    all_versions: list[SchemaInfo]


class ServiceEndpoints(BaseModel):
    arrow_server: str
    feature_server: str

```

# model_comparison.py
-## Location -> root_directory.common
```python
import turboml as tb
from typing import List


def compare_model_metrics(models: List, metric: str, x_axis_title: str = "Samples"):
    """Generates a plotly plot for Windowed metrics comparison for a list of models

    Args:
        model_names (List): List of models to compare
        metric (str): Metric for evaluation of models, should be chosen from tb.evaluation_metrics()
        x_axis_title (str, optional): X axis title for the plot. Defaults to "Samples".

    Raises:
        Exception: If other metrics are chosen then Execption is raised
    """
    import plotly.graph_objs as go

    model_traces = []
    windowed_metrics = tb.evaluation_metrics()
    if metric in windowed_metrics:
        for model in models:
            # It is assumed that the user registers the metric before comparing the models
            evals = model.get_evaluation(metric_name=metric)
            model_evaluations = [eval.metric for eval in evals]
            index = [eval.index for eval in evals]
            trace = go.Scatter(
                x=index,
                y=model_evaluations,
                mode="lines",
                name=model.model_name,
            )
            model_traces.append(trace)
        layout = go.Layout(
            title=metric,
            xaxis=dict(title=x_axis_title),  # noqa
            yaxis=dict(title="Metric"),  # noqa
        )
        fig = go.Figure(data=model_traces, layout=layout)
        fig.show()
    else:
        raise Exception(
            f"The other Windowed metrics arent supported yet, please choose from {tb.evaluation_metrics()}, if you want to use batch metrics please use tb.compare_batch_metrics"
        )

```

# namespaces.py
-## Location -> root_directory.common
```python
from .api import api
from .models import NamespaceAcquisitionRequest


def set_active_namespace(namespace: str) -> None:
    api.set_active_namespace(namespace)


def get_default_namespace() -> str:
    return api.get(endpoint="user/namespace/default").json()


def set_default_namespace(namespace: str) -> None:
    resp = api.put(endpoint=f"user/namespace/default?namespace={namespace}")
    if resp.status_code not in range(200, 300):
        raise Exception(f"Failed to set default namespace: {resp.json()['detail']}")


def acquire_namespace(namespace: str) -> None:
    payload = NamespaceAcquisitionRequest(namespace=namespace)
    resp = api.post(
        endpoint="user/namespace/acquire",
        json=payload.model_dump(),
        exclude_namespace=True,
    )
    if resp.status_code not in range(200, 300):
        raise Exception(f"Failed to acquire namespace: {resp.json()['detail']}")


def list_namespaces(include_shared: bool = False) -> list[str]:
    resp = api.get(endpoint=f"user/namespaces?include_shared={include_shared}")
    if resp.status_code not in range(200, 300):
        raise Exception(f"Failed to list namespaces: {resp.json()['detail']}")
    return resp.json()

```

# pymodel.py
-## Location -> root_directory.common
```python
from turboml_bindings.pymodel import *  # noqa

```

# pytypes.py
-## Location -> root_directory.common
```python
from turboml_bindings.pytypes import *  # noqa

```

# types.py
-## Location -> root_directory.common
```python
from __future__ import annotations
from abc import abstractmethod
from typing import NewType, TYPE_CHECKING

if TYPE_CHECKING:
    import turboml.common.pytypes as pytypes

GGUFModelId = NewType("GGUFModelId", str)


class PythonModel:
    @abstractmethod
    def init_imports(self):
        """
        Must import all libraries/modules needed in learn_one and predict_one
        """
        pass

    @abstractmethod
    def learn_one(self, input: pytypes.InputData) -> None:
        pass

    @abstractmethod
    def predict_one(self, input: pytypes.InputData, output: pytypes.OutputData) -> None:
        pass

```

# udf.py
-## Location -> root_directory.common
```python
from typing import Any


class ModelMetricAggregateFunction:
    """
    Base class for defining a Model Metric Aggregate Function.
    """

    def __init__(self):
        pass

    def create_state(self) -> Any:
        """
        Create the initial state for the UDAF.

        Returns:
            Any: The initial state.
        """
        raise NotImplementedError(
            "The 'create_state' method must be implemented by subclasses."
        )

    def accumulate(self, state: Any, prediction: float, label: float) -> Any:
        """
        Accumulate input data (prediction and label) into the state.

        Args:
            state (Any): The current state.
            prediction (float): The predicted value.
            label (float): The ground truth label.

        Returns:
            Any: The updated state.
        """
        raise NotImplementedError(
            "The 'accumulate' method must be implemented by subclasses."
        )

    def retract(self, state: Any, prediction: float, label: float) -> Any:
        """
        Retract input data from the state (optional).

        Args:
            state (Any): The current state.
            prediction (float): The predicted value.
            label (float): The ground truth label.

        Returns:
            Any: The updated state.
        """
        raise NotImplementedError(
            "The 'retract' method must be implemented by subclasses."
        )

    def merge_states(self, state1: Any, state2: Any) -> Any:
        """
        Merge two states into one.

        Args:
            state1 (Any): The first state.
            state2 (Any): The second state.

        Returns:
            Any: The merged state.
        """
        raise NotImplementedError(
            "The 'merge_states' method must be implemented by subclasses."
        )

    def finish(self, state: Any) -> float:
        """
        Finalize the aggregation and compute the result.

        Args:
            state (Any): The final state.

        Returns:
            float: The result of the aggregation.
        """
        raise NotImplementedError(
            "The 'finish' method must be implemented by subclasses."
        )

```

# util.py
-## Location -> root_directory.common
```python
import ast
import inspect
from collections.abc import Sequence
from typing import TypeVar, cast, List

import pyarrow as pa

V = TypeVar("V")


def promote_list(val: V | Sequence[V]) -> list[V]:
    """Ensure that the value is a list.

    Parameters
    ----------
    val
        Value to promote

    Returns
    -------
    list

    """
    if isinstance(val, list):
        return val
    elif isinstance(val, dict):
        return [val]
    elif val is None:
        return []
    else:
        return [val]


def risingwave_type_to_pyarrow(type: str):
    """
    Convert a SQL data type string to `pyarrow.DataType`.
    """
    t = type.upper()
    if t.endswith("[]"):
        return pa.list_(risingwave_type_to_pyarrow(type[:-2]))
    elif t.startswith("STRUCT"):
        return _parse_struct(t)
    return _simple_type(t)


def _parse_struct(type: str):
    # extract 'STRUCT<a:INT, b:VARCHAR, c:STRUCT<d:INT>, ...>'
    type_list = type[7:-1]  # strip "STRUCT<>"
    fields = []
    start = 0
    depth = 0
    for i, c in enumerate(type_list):
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif c == "," and depth == 0:
            name, t = type_list[start:i].split(":", maxsplit=1)
            name = name.strip()
            t = t.strip()
            fields.append(pa.field(name, risingwave_type_to_pyarrow(t)))
            start = i + 1
    if ":" in type_list[start:].strip():
        name, t = type_list[start:].split(":", maxsplit=1)
        name = name.strip()
        t = t.strip()
        fields.append(pa.field(name, risingwave_type_to_pyarrow(t)))
    return pa.struct(fields)


def _simple_type(t: str):
    type_map = {
        "NULL": pa.null,
        "BOOLEAN": pa.bool_,
        "BOOL": pa.bool_,
        "TINYINT": pa.int8,
        "INT8": pa.int8,
        "SMALLINT": pa.int16,
        "INT16": pa.int16,
        "INT": pa.int32,
        "INTEGER": pa.int32,
        "INT32": pa.int32,
        "BIGINT": pa.int64,
        "INT64": pa.int64,
        "UINT8": pa.uint8,
        "UINT16": pa.uint16,
        "UINT32": pa.uint32,
        "UINT64": pa.uint64,
        "FLOAT32": pa.float32,
        "REAL": pa.float32,
        "FLOAT64": pa.float64,
        "DOUBLE PRECISION": pa.float64,
        "DOUBLE": pa.float64,
        "DATE32": pa.date32,
        "DATE": pa.date32,
        "TIME64": lambda: pa.time64("us"),
        "TIME": lambda: pa.time64("us"),
        "TIME WITHOUT TIME ZONE": lambda: pa.time64("us"),
        "TIMESTAMP": lambda: pa.timestamp("us"),
        "TIMESTAMP WITHOUT TIME ZONE": lambda: pa.timestamp("us"),
        "INTERVAL": pa.month_day_nano_interval,
        "STRING": pa.string,
        "VARCHAR": pa.string,
        "LARGE_STRING": pa.large_string,
        "BINARY": pa.binary,
        "BYTEA": pa.binary,
        "LARGE_BINARY": pa.large_binary,
    }

    if t in type_map:
        return type_map[t]()

    raise ValueError(f"Unsupported type: {t}")


def _is_running_ipython() -> bool:
    """Checks if we are currently running in IPython"""
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


def _get_ipython_cell_sources() -> list[str]:
    """Returns the source code of all cells in the running IPython session.
    See https://github.com/wandb/weave/pull/1864
    """
    shell = get_ipython()  # type: ignore[name-defined]  # noqa: F821
    if not hasattr(shell, "user_ns"):
        raise AttributeError("Cannot access user namespace")
    cells = cast(list[str], shell.user_ns["In"])
    # First cell is always empty
    return cells[1:]


def _extract_relevant_imports(
    import_statements: List[ast.AST], used_names: set
) -> List[str]:
    """Filter and format relevant import statements based on used names."""
    relevant_imports = []
    for node in import_statements:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname in used_names or alias.name in used_names:
                    relevant_imports.append(
                        f"import {alias.name}"
                        + (f" as {alias.asname}" if alias.asname else "")
                    )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname in used_names or alias.name in used_names:
                    relevant_imports.append(
                        f"from {node.module} import {alias.name}"
                        + (f" as {alias.asname}" if alias.asname else "")
                    )
    return relevant_imports


def _find_imports_and_used_names(source_code: str, func_source: str) -> List[str]:
    """Find imports and their relevance to the function source code."""
    module_ast = ast.parse(source_code)
    func_ast = ast.parse(func_source).body[0]

    import_statements = [
        node
        for node in ast.walk(module_ast)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    used_names = {node.id for node in ast.walk(func_ast) if isinstance(node, ast.Name)}

    return _extract_relevant_imports(import_statements, used_names)


def get_imports_used_in_function(func) -> str:
    """Get all relevant imports used in a function."""
    if _is_running_ipython():
        cell_sources = _get_ipython_cell_sources()
        imports = []
        for cell_source in cell_sources:
            try:
                imports.extend(
                    _find_imports_and_used_names(cell_source, inspect.getsource(func))
                )
            except Exception:
                continue
        return "\n".join(set(imports))

    else:
        module_source_code = inspect.getsource(inspect.getmodule(func))
        func_source = inspect.getsource(func)
        return "\n".join(_find_imports_and_used_names(module_source_code, func_source))

```


# config_pb2.py
-## Location -> root_directory.common.protos
```python
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: config.proto
# Protobuf Python Version: 4.25.5
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0c\x63onfig.proto\x12\x06\x63onfig\"F\n\rMStreamConfig\x12\x10\n\x08num_rows\x18\x01 \x01(\x05\x12\x13\n\x0bnum_buckets\x18\x02 \x01(\x05\x12\x0e\n\x06\x66\x61\x63tor\x18\x03 \x01(\x02\"c\n\tRCFConfig\x12\x12\n\ntime_decay\x18\x01 \x01(\x02\x12\x17\n\x0fnumber_of_trees\x18\x02 \x01(\x05\x12\x14\n\x0coutput_after\x18\x03 \x01(\x05\x12\x13\n\x0bsample_size\x18\x04 \x01(\x05\"A\n\tHSTConfig\x12\x0f\n\x07n_trees\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\x13\n\x0bwindow_size\x18\x03 \x01(\x05\"9\n\nONNXConfig\x12\x17\n\x0fmodel_save_name\x18\x01 \x01(\t\x12\x12\n\nmodel_data\x18\x02 \x01(\x0c\"\x90\x01\n\x19HoeffdingClassifierConfig\x12\r\n\x05\x64\x65lta\x18\x01 \x01(\x02\x12\x0b\n\x03tau\x18\x02 \x01(\x02\x12\x14\n\x0cgrace_period\x18\x03 \x01(\x05\x12\x11\n\tn_classes\x18\x04 \x01(\x05\x12\x18\n\x10leaf_pred_method\x18\x05 \x01(\t\x12\x14\n\x0csplit_method\x18\x06 \x01(\t\"f\n\x18HoeffdingRegressorConfig\x12\r\n\x05\x64\x65lta\x18\x01 \x01(\x02\x12\x0b\n\x03tau\x18\x02 \x01(\x02\x12\x14\n\x0cgrace_period\x18\x03 \x01(\x05\x12\x18\n\x10leaf_pred_method\x18\x04 \x01(\t\"\x8c\x01\n\x13\x41MFClassifierConfig\x12\x11\n\tn_classes\x18\x01 \x01(\x05\x12\x14\n\x0cn_estimators\x18\x02 \x01(\x05\x12\x0c\n\x04step\x18\x03 \x01(\x02\x12\x17\n\x0fuse_aggregation\x18\x04 \x01(\x08\x12\x11\n\tdirichlet\x18\x05 \x01(\x02\x12\x12\n\nsplit_pure\x18\x06 \x01(\x08\"d\n\x12\x41MFRegressorConfig\x12\x14\n\x0cn_estimators\x18\x01 \x01(\x05\x12\x0c\n\x04step\x18\x02 \x01(\x02\x12\x17\n\x0fuse_aggregation\x18\x03 \x01(\x08\x12\x11\n\tdirichlet\x18\x04 \x01(\x02\"\xb4\x01\n\x13\x46\x46MClassifierConfig\x12\x11\n\tn_factors\x18\x01 \x01(\x05\x12\x11\n\tl1_weight\x18\x02 \x01(\x02\x12\x11\n\tl2_weight\x18\x03 \x01(\x02\x12\x11\n\tl1_latent\x18\x04 \x01(\x02\x12\x11\n\tl2_latent\x18\x05 \x01(\x02\x12\x11\n\tintercept\x18\x06 \x01(\x02\x12\x14\n\x0cintercept_lr\x18\x07 \x01(\x02\x12\x15\n\rclip_gradient\x18\x08 \x01(\x02\"\xb3\x01\n\x12\x46\x46MRegressorConfig\x12\x11\n\tn_factors\x18\x01 \x01(\x05\x12\x11\n\tl1_weight\x18\x02 \x01(\x02\x12\x11\n\tl2_weight\x18\x03 \x01(\x02\x12\x11\n\tl1_latent\x18\x04 \x01(\x02\x12\x11\n\tl2_latent\x18\x05 \x01(\x02\x12\x11\n\tintercept\x18\x06 \x01(\x02\x12\x14\n\x0cintercept_lr\x18\x07 \x01(\x02\x12\x15\n\rclip_gradient\x18\x08 \x01(\x02\"\x87\x01\n\x0eSNARIMAXConfig\x12\x0f\n\x07horizon\x18\x01 \x01(\x05\x12\t\n\x01p\x18\x02 \x01(\x05\x12\t\n\x01\x64\x18\x03 \x01(\x05\x12\t\n\x01q\x18\x04 \x01(\x05\x12\t\n\x01m\x18\x05 \x01(\x05\x12\n\n\x02sp\x18\x06 \x01(\x05\x12\n\n\x02sd\x18\x07 \x01(\x05\x12\n\n\x02sq\x18\x08 \x01(\x05\x12\x14\n\x0cnum_children\x18\t \x01(\x05\"\xd0\x02\n\x13NeuralNetworkConfig\x12>\n\x06layers\x18\x01 \x03(\x0b\x32..config.NeuralNetworkConfig.NeuralNetworkLayer\x12\x0f\n\x07\x64ropout\x18\x02 \x01(\x02\x12\x15\n\rloss_function\x18\x03 \x01(\t\x12\x11\n\toptimizer\x18\x04 \x01(\t\x12\x15\n\rlearning_rate\x18\x05 \x01(\x02\x12\x12\n\nbatch_size\x18\x06 \x01(\x05\x1a\x92\x01\n\x12NeuralNetworkLayer\x12\x12\n\ninput_size\x18\x01 \x01(\x05\x12\x13\n\x0boutput_size\x18\x02 \x01(\x05\x12\x12\n\nactivation\x18\x03 \x01(\t\x12\x0f\n\x07\x64ropout\x18\x04 \x01(\x02\x12\x1c\n\x14residual_connections\x18\x05 \x03(\x05\x12\x10\n\x08use_bias\x18\x06 \x01(\x08\"\x7f\n\tONNConfig\x12\x1d\n\x15max_num_hidden_layers\x18\x01 \x01(\x05\x12\x1f\n\x17qtd_neuron_hidden_layer\x18\x02 \x01(\x05\x12\x11\n\tn_classes\x18\x03 \x01(\x05\x12\t\n\x01\x62\x18\x04 \x01(\x02\x12\t\n\x01n\x18\x05 \x01(\x02\x12\t\n\x01s\x18\x06 \x01(\x02\"4\n\tOVRConfig\x12\x11\n\tn_classes\x18\x01 \x01(\x05\x12\x14\n\x0cnum_children\x18\x02 \x01(\x05\"O\n\x1fRandomProjectionEmbeddingConfig\x12\x14\n\x0cn_embeddings\x18\x01 \x01(\x05\x12\x16\n\x0etype_embedding\x18\x02 \x01(\t\",\n\x14\x45mbeddingModelConfig\x12\x14\n\x0cnum_children\x18\x01 \x01(\x05\"A\n\x1a\x42\x61nditModelSelectionConfig\x12\x0e\n\x06\x62\x61ndit\x18\x01 \x01(\t\x12\x13\n\x0bmetric_name\x18\x02 \x01(\t\"U\n$ContextualBanditModelSelectionConfig\x12\x18\n\x10\x63ontextualbandit\x18\x01 \x01(\t\x12\x13\n\x0bmetric_name\x18\x02 \x01(\t\"\x8f\x01\n!LeveragingBaggingClassifierConfig\x12\x10\n\x08n_models\x18\x01 \x01(\x05\x12\x11\n\tn_classes\x18\x02 \x01(\x05\x12\t\n\x01w\x18\x03 \x01(\x02\x12\x16\n\x0e\x62\x61gging_method\x18\x04 \x01(\t\x12\x0c\n\x04seed\x18\x05 \x01(\x05\x12\x14\n\x0cnum_children\x18\x06 \x01(\x05\"\x83\x01\n\'HeteroLeveragingBaggingClassifierConfig\x12\x11\n\tn_classes\x18\x01 \x01(\x05\x12\t\n\x01w\x18\x02 \x01(\x02\x12\x16\n\x0e\x62\x61gging_method\x18\x03 \x01(\t\x12\x0c\n\x04seed\x18\x04 \x01(\x05\x12\x14\n\x0cnum_children\x18\x05 \x01(\x05\"c\n\x18\x41\x64\x61\x42oostClassifierConfig\x12\x10\n\x08n_models\x18\x01 \x01(\x05\x12\x11\n\tn_classes\x18\x02 \x01(\x05\x12\x0c\n\x04seed\x18\x03 \x01(\x05\x12\x14\n\x0cnum_children\x18\x04 \x01(\x05\"W\n\x1eHeteroAdaBoostClassifierConfig\x12\x11\n\tn_classes\x18\x01 \x01(\x05\x12\x0c\n\x04seed\x18\x02 \x01(\x05\x12\x14\n\x0cnum_children\x18\x03 \x01(\x05\"Y\n\x13SGTClassifierConfig\x12\r\n\x05\x64\x65lta\x18\x01 \x01(\x02\x12\x14\n\x0cgrace_period\x18\x02 \x01(\x05\x12\x0e\n\x06lambda\x18\x03 \x01(\x02\x12\r\n\x05gamma\x18\x04 \x01(\x02\"X\n\x12SGTRegressorConfig\x12\r\n\x05\x64\x65lta\x18\x01 \x01(\x02\x12\x14\n\x0cgrace_period\x18\x02 \x01(\x05\x12\x0e\n\x06lambda\x18\x03 \x01(\x02\x12\r\n\x05gamma\x18\x04 \x01(\x02\"\x92\x01\n\x13RandomSamplerConfig\x12\x11\n\tn_classes\x18\x01 \x01(\x05\x12\x14\n\x0c\x64\x65sired_dist\x18\x02 \x03(\x02\x12\x17\n\x0fsampling_method\x18\x03 \x01(\t\x12\x15\n\rsampling_rate\x18\x04 \x01(\x02\x12\x0c\n\x04seed\x18\x05 \x01(\x05\x12\x14\n\x0cnum_children\x18\x06 \x01(\x05\"5\n\x11MultinomialConfig\x12\x11\n\tn_classes\x18\x01 \x01(\x05\x12\r\n\x05\x61lpha\x18\x02 \x01(\x02\"#\n\x0eGaussianConfig\x12\x11\n\tn_classes\x18\x01 \x01(\x05\"X\n\x0cPythonConfig\x12\x13\n\x0bmodule_name\x18\x01 \x01(\t\x12\x12\n\nclass_name\x18\x02 \x01(\t\x12\x11\n\tvenv_name\x18\x03 \x01(\t\x12\x0c\n\x04\x63ode\x18\x04 \x01(\t\"`\n\x14PythonEnsembleConfig\x12\x13\n\x0bmodule_name\x18\x01 \x01(\t\x12\x12\n\nclass_name\x18\x02 \x01(\t\x12\x11\n\tvenv_name\x18\x03 \x01(\t\x12\x0c\n\x04\x63ode\x18\x04 \x01(\t\"\xbf\x01\n\x12PreProcessorConfig\x12\x19\n\x11preprocessor_name\x18\x01 \x01(\t\x12\x17\n\x0ftext_categories\x18\x02 \x03(\x05\x12\x14\n\x0cnum_children\x18\x03 \x01(\x05\x12\x15\n\rgguf_model_id\x18\x04 \x01(\t\x12\x1c\n\x14max_tokens_per_input\x18\x05 \x01(\x05\x12\x13\n\x0bimage_sizes\x18\x06 \x03(\x05\x12\x15\n\rchannel_first\x18\x07 \x01(\x08\"\x8f\x02\n\x15\x41\x64\x61ptiveXGBoostConfig\x12\x11\n\tn_classes\x18\x01 \x01(\x05\x12\x15\n\rlearning_rate\x18\x02 \x01(\x02\x12\x11\n\tmax_depth\x18\x03 \x01(\x05\x12\x17\n\x0fmax_window_size\x18\x04 \x01(\x05\x12\x17\n\x0fmin_window_size\x18\x05 \x01(\x05\x12\x12\n\nmax_buffer\x18\x06 \x01(\x05\x12\x11\n\tpre_train\x18\x07 \x01(\x05\x12\x14\n\x0c\x64\x65tect_drift\x18\x08 \x01(\x08\x12\x13\n\x0buse_updater\x18\t \x01(\x08\x12\x17\n\x0ftrees_per_train\x18\n \x01(\x05\x12\x1c\n\x14percent_update_trees\x18\x0b \x01(\x02\"\xee\x01\n\x12\x41\x64\x61ptiveLGBMConfig\x12\x11\n\tn_classes\x18\x01 \x01(\x05\x12\x15\n\rlearning_rate\x18\x02 \x01(\x02\x12\x11\n\tmax_depth\x18\x03 \x01(\x05\x12\x17\n\x0fmax_window_size\x18\x04 \x01(\x05\x12\x17\n\x0fmin_window_size\x18\x05 \x01(\x05\x12\x12\n\nmax_buffer\x18\x06 \x01(\x05\x12\x11\n\tpre_train\x18\x07 \x01(\x05\x12\x14\n\x0c\x64\x65tect_drift\x18\x08 \x01(\x08\x12\x13\n\x0buse_updater\x18\t \x01(\x08\x12\x17\n\x0ftrees_per_train\x18\n \x01(\x05\"t\n\x13RestAPIClientConfig\x12\x12\n\nserver_url\x18\x01 \x01(\t\x12\x13\n\x0bmax_retries\x18\x02 \x01(\x05\x12\x1a\n\x12\x63onnection_timeout\x18\x03 \x01(\x05\x12\x18\n\x10max_request_time\x18\x04 \x01(\x05\"q\n\x10GRPCClientConfig\x12\x12\n\nserver_url\x18\x01 \x01(\t\x12\x13\n\x0bmax_retries\x18\x02 \x01(\x05\x12\x1a\n\x12\x63onnection_timeout\x18\x03 \x01(\x05\x12\x18\n\x10max_request_time\x18\x04 \x01(\x05\"P\n\x19LLAMAEmbeddingModelConfig\x12\x15\n\rgguf_model_id\x18\x01 \x01(\t\x12\x1c\n\x14max_tokens_per_input\x18\x02 \x01(\x05\">\n\x0fLlamaTextConfig\x12\x15\n\rgguf_model_id\x18\x01 \x01(\t\x12\x14\n\x0cnum_children\x18\x02 \x01(\x05\"B\n\x13\x43lipEmbeddingConfig\x12\x15\n\rgguf_model_id\x18\x01 \x01(\t\x12\x14\n\x0cnum_children\x18\x02 \x01(\x05\"\xdf\x12\n\x0bModelConfig\x12\x11\n\talgorithm\x18\x01 \x01(\t\x12\x14\n\x0cnum_children\x18& \x01(\x05\x12/\n\x0emstream_config\x18\x02 \x01(\x0b\x32\x15.config.MStreamConfigH\x00\x12\'\n\nrcf_config\x18\x03 \x01(\x0b\x32\x11.config.RCFConfigH\x00\x12\'\n\nhst_config\x18\x04 \x01(\x0b\x32\x11.config.HSTConfigH\x00\x12)\n\x0bonnx_config\x18\x06 \x01(\x0b\x32\x12.config.ONNXConfigH\x00\x12H\n\x1bhoeffding_classifier_config\x18\x07 \x01(\x0b\x32!.config.HoeffdingClassifierConfigH\x00\x12\x46\n\x1ahoeffding_regressor_config\x18\x08 \x01(\x0b\x32 .config.HoeffdingRegressorConfigH\x00\x12<\n\x15\x61mf_classifier_config\x18\t \x01(\x0b\x32\x1b.config.AMFClassifierConfigH\x00\x12:\n\x14\x61mf_regressor_config\x18\n \x01(\x0b\x32\x1a.config.AMFRegressorConfigH\x00\x12<\n\x15\x66\x66m_classifier_config\x18\x0b \x01(\x0b\x32\x1b.config.FFMClassifierConfigH\x00\x12:\n\x14\x66\x66m_regressor_config\x18\x0c \x01(\x0b\x32\x1a.config.FFMRegressorConfigH\x00\x12\x31\n\x0fsnarimax_config\x18\r \x01(\x0b\x32\x16.config.SNARIMAXConfigH\x00\x12\x30\n\tnn_config\x18\x0e \x01(\x0b\x32\x1b.config.NeuralNetworkConfigH\x00\x12\'\n\nonn_config\x18\x0f \x01(\x0b\x32\x11.config.ONNConfigH\x00\x12Y\n$leveraging_bagging_classifier_config\x18\x10 \x01(\x0b\x32).config.LeveragingBaggingClassifierConfigH\x00\x12\x46\n\x1a\x61\x64\x61\x62oost_classifier_config\x18\x11 \x01(\x0b\x32 .config.AdaBoostClassifierConfigH\x00\x12<\n\x15random_sampler_config\x18\x12 \x01(\x0b\x32\x1b.config.RandomSamplerConfigH\x00\x12K\n\x1d\x62\x61ndit_model_selection_config\x18\x13 \x01(\x0b\x32\".config.BanditModelSelectionConfigH\x00\x12`\n(contextual_bandit_model_selection_config\x18\x14 \x01(\x0b\x32,.config.ContextualBanditModelSelectionConfigH\x00\x12-\n\rpython_config\x18\x15 \x01(\x0b\x32\x14.config.PythonConfigH\x00\x12\x39\n\x13preprocessor_config\x18\x16 \x01(\x0b\x32\x1a.config.PreProcessorConfigH\x00\x12\x37\n\x1aovr_model_selection_config\x18\x17 \x01(\x0b\x32\x11.config.OVRConfigH\x00\x12K\n\x18random_projection_config\x18\x18 \x01(\x0b\x32\'.config.RandomProjectionEmbeddingConfigH\x00\x12>\n\x16\x65mbedding_model_config\x18\x19 \x01(\x0b\x32\x1c.config.EmbeddingModelConfigH\x00\x12\x66\n+hetero_leveraging_bagging_classifier_config\x18\x1a \x01(\x0b\x32/.config.HeteroLeveragingBaggingClassifierConfigH\x00\x12S\n!hetero_adaboost_classifier_config\x18\x1b \x01(\x0b\x32&.config.HeteroAdaBoostClassifierConfigH\x00\x12<\n\x15sgt_classifier_config\x18\x1c \x01(\x0b\x32\x1b.config.SGTClassifierConfigH\x00\x12:\n\x14sgt_regressor_config\x18\x1d \x01(\x0b\x32\x1a.config.SGTRegressorConfigH\x00\x12\x37\n\x12multinomial_config\x18\x1e \x01(\x0b\x32\x19.config.MultinomialConfigH\x00\x12\x31\n\x0fgaussian_config\x18\x1f \x01(\x0b\x32\x16.config.GaussianConfigH\x00\x12@\n\x17\x61\x64\x61ptive_xgboost_config\x18  \x01(\x0b\x32\x1d.config.AdaptiveXGBoostConfigH\x00\x12:\n\x14\x61\x64\x61ptive_lgbm_config\x18! \x01(\x0b\x32\x1a.config.AdaptiveLGBMConfigH\x00\x12\x43\n\x16llama_embedding_config\x18\" \x01(\x0b\x32!.config.LLAMAEmbeddingModelConfigH\x00\x12=\n\x16rest_api_client_config\x18# \x01(\x0b\x32\x1b.config.RestAPIClientConfigH\x00\x12\x34\n\x11llama_text_config\x18$ \x01(\x0b\x32\x17.config.LlamaTextConfigH\x00\x12<\n\x15\x63lip_embedding_config\x18% \x01(\x0b\x32\x1b.config.ClipEmbeddingConfigH\x00\x12>\n\x16python_ensemble_config\x18\' \x01(\x0b\x32\x1c.config.PythonEnsembleConfigH\x00\x12\x36\n\x12grpc_client_config\x18( \x01(\x0b\x32\x18.config.GRPCClientConfigH\x00\x42\x0e\n\x0cmodel_configJ\x04\x08\x05\x10\x06\"^\n\x16\x46\x65\x61tureRetrievalConfig\x12\x15\n\rsql_statement\x18\x01 \x01(\t\x12\x18\n\x10placeholder_cols\x18\x02 \x03(\t\x12\x13\n\x0bresult_cols\x18\x03 \x03(\t\"r\n\x13KafkaProducerConfig\x12\x13\n\x0bwrite_topic\x18\x01 \x01(\t\x12\x17\n\x0fproto_file_name\x18\x02 \x01(\t\x12\x1a\n\x12proto_message_name\x18\x03 \x01(\t\x12\x11\n\tschema_id\x18\x04 \x01(\x05\"v\n\x13KafkaConsumerConfig\x12\x12\n\nread_topic\x18\x01 \x01(\t\x12\x17\n\x0fproto_file_name\x18\x02 \x01(\t\x12\x1a\n\x12proto_message_name\x18\x03 \x01(\t\x12\x16\n\x0e\x63onsumer_group\x18\x04 \x01(\t\"\x93\x01\n\x0bInputConfig\x12\x11\n\tkey_field\x18\x01 \x01(\t\x12\x13\n\x0blabel_field\x18\x02 \x01(\t\x12\x11\n\tnumerical\x18\x03 \x03(\t\x12\x13\n\x0b\x63\x61tegorical\x18\x04 \x03(\t\x12\x11\n\ttime_tick\x18\x05 \x01(\t\x12\x0f\n\x07textual\x18\x06 \x03(\t\x12\x10\n\x08imaginal\x18\x07 \x03(\t\"c\n\rLearnerConfig\x12\x17\n\x0fpredict_workers\x18\x01 \x01(\x05\x12\x19\n\x11update_batch_size\x18\x02 \x01(\x05\x12\x1e\n\x16synchronization_method\x18\x03 \x01(\t\"\xd4\x04\n\rTurboMLConfig\x12\x0f\n\x07\x62rokers\x18\x01 \x01(\t\x12\x32\n\rfeat_consumer\x18\x02 \x01(\x0b\x32\x1b.config.KafkaConsumerConfig\x12\x34\n\x0foutput_producer\x18\x03 \x01(\x0b\x32\x1b.config.KafkaProducerConfig\x12\x33\n\x0elabel_consumer\x18\x04 \x01(\x0b\x32\x1b.config.KafkaConsumerConfig\x12)\n\x0cinput_config\x18\x05 \x01(\x0b\x32\x13.config.InputConfig\x12*\n\rmodel_configs\x18\x06 \x03(\x0b\x32\x13.config.ModelConfig\x12\x18\n\x10initial_model_id\x18\x07 \x01(\t\x12\x10\n\x08\x61pi_port\x18\x08 \x01(\x05\x12\x12\n\narrow_port\x18\t \x01(\x05\x12\x39\n\x11\x66\x65\x61ture_retrieval\x18\n \x01(\x0b\x32\x1e.config.FeatureRetrievalConfig\x12\x36\n\x11\x63ombined_producer\x18\x0b \x01(\x0b\x32\x1b.config.KafkaProducerConfig\x12\x36\n\x11\x63ombined_consumer\x18\x0c \x01(\x0b\x32\x1b.config.KafkaConsumerConfig\x12-\n\x0elearner_config\x18\r \x01(\x0b\x32\x15.config.LearnerConfig\x12\"\n\x1a\x66ully_qualified_model_name\x18\x0e \x01(\t\"\xac\x01\n\x0eTrainJobConfig\x12\x19\n\x11initial_model_key\x18\x01 \x01(\t\x12)\n\x0cinput_config\x18\x02 \x01(\x0b\x32\x13.config.InputConfig\x12*\n\rmodel_configs\x18\x03 \x03(\x0b\x32\x13.config.ModelConfig\x12\x12\n\nmodel_name\x18\x04 \x01(\t\x12\x14\n\x0cversion_name\x18\x05 \x01(\t\"=\n\x0fModelConfigList\x12*\n\rmodel_configs\x18\x01 \x03(\x0b\x32\x13.config.ModelConfig')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'config_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_MSTREAMCONFIG']._serialized_start=24
  _globals['_MSTREAMCONFIG']._serialized_end=94
  _globals['_RCFCONFIG']._serialized_start=96
  _globals['_RCFCONFIG']._serialized_end=195
  _globals['_HSTCONFIG']._serialized_start=197
  _globals['_HSTCONFIG']._serialized_end=262
  _globals['_ONNXCONFIG']._serialized_start=264
  _globals['_ONNXCONFIG']._serialized_end=321
  _globals['_HOEFFDINGCLASSIFIERCONFIG']._serialized_start=324
  _globals['_HOEFFDINGCLASSIFIERCONFIG']._serialized_end=468
  _globals['_HOEFFDINGREGRESSORCONFIG']._serialized_start=470
  _globals['_HOEFFDINGREGRESSORCONFIG']._serialized_end=572
  _globals['_AMFCLASSIFIERCONFIG']._serialized_start=575
  _globals['_AMFCLASSIFIERCONFIG']._serialized_end=715
  _globals['_AMFREGRESSORCONFIG']._serialized_start=717
  _globals['_AMFREGRESSORCONFIG']._serialized_end=817
  _globals['_FFMCLASSIFIERCONFIG']._serialized_start=820
  _globals['_FFMCLASSIFIERCONFIG']._serialized_end=1000
  _globals['_FFMREGRESSORCONFIG']._serialized_start=1003
  _globals['_FFMREGRESSORCONFIG']._serialized_end=1182
  _globals['_SNARIMAXCONFIG']._serialized_start=1185
  _globals['_SNARIMAXCONFIG']._serialized_end=1320
  _globals['_NEURALNETWORKCONFIG']._serialized_start=1323
  _globals['_NEURALNETWORKCONFIG']._serialized_end=1659
  _globals['_NEURALNETWORKCONFIG_NEURALNETWORKLAYER']._serialized_start=1513
  _globals['_NEURALNETWORKCONFIG_NEURALNETWORKLAYER']._serialized_end=1659
  _globals['_ONNCONFIG']._serialized_start=1661
  _globals['_ONNCONFIG']._serialized_end=1788
  _globals['_OVRCONFIG']._serialized_start=1790
  _globals['_OVRCONFIG']._serialized_end=1842
  _globals['_RANDOMPROJECTIONEMBEDDINGCONFIG']._serialized_start=1844
  _globals['_RANDOMPROJECTIONEMBEDDINGCONFIG']._serialized_end=1923
  _globals['_EMBEDDINGMODELCONFIG']._serialized_start=1925
  _globals['_EMBEDDINGMODELCONFIG']._serialized_end=1969
  _globals['_BANDITMODELSELECTIONCONFIG']._serialized_start=1971
  _globals['_BANDITMODELSELECTIONCONFIG']._serialized_end=2036
  _globals['_CONTEXTUALBANDITMODELSELECTIONCONFIG']._serialized_start=2038
  _globals['_CONTEXTUALBANDITMODELSELECTIONCONFIG']._serialized_end=2123
  _globals['_LEVERAGINGBAGGINGCLASSIFIERCONFIG']._serialized_start=2126
  _globals['_LEVERAGINGBAGGINGCLASSIFIERCONFIG']._serialized_end=2269
  _globals['_HETEROLEVERAGINGBAGGINGCLASSIFIERCONFIG']._serialized_start=2272
  _globals['_HETEROLEVERAGINGBAGGINGCLASSIFIERCONFIG']._serialized_end=2403
  _globals['_ADABOOSTCLASSIFIERCONFIG']._serialized_start=2405
  _globals['_ADABOOSTCLASSIFIERCONFIG']._serialized_end=2504
  _globals['_HETEROADABOOSTCLASSIFIERCONFIG']._serialized_start=2506
  _globals['_HETEROADABOOSTCLASSIFIERCONFIG']._serialized_end=2593
  _globals['_SGTCLASSIFIERCONFIG']._serialized_start=2595
  _globals['_SGTCLASSIFIERCONFIG']._serialized_end=2684
  _globals['_SGTREGRESSORCONFIG']._serialized_start=2686
  _globals['_SGTREGRESSORCONFIG']._serialized_end=2774
  _globals['_RANDOMSAMPLERCONFIG']._serialized_start=2777
  _globals['_RANDOMSAMPLERCONFIG']._serialized_end=2923
  _globals['_MULTINOMIALCONFIG']._serialized_start=2925
  _globals['_MULTINOMIALCONFIG']._serialized_end=2978
  _globals['_GAUSSIANCONFIG']._serialized_start=2980
  _globals['_GAUSSIANCONFIG']._serialized_end=3015
  _globals['_PYTHONCONFIG']._serialized_start=3017
  _globals['_PYTHONCONFIG']._serialized_end=3105
  _globals['_PYTHONENSEMBLECONFIG']._serialized_start=3107
  _globals['_PYTHONENSEMBLECONFIG']._serialized_end=3203
  _globals['_PREPROCESSORCONFIG']._serialized_start=3206
  _globals['_PREPROCESSORCONFIG']._serialized_end=3397
  _globals['_ADAPTIVEXGBOOSTCONFIG']._serialized_start=3400
  _globals['_ADAPTIVEXGBOOSTCONFIG']._serialized_end=3671
  _globals['_ADAPTIVELGBMCONFIG']._serialized_start=3674
  _globals['_ADAPTIVELGBMCONFIG']._serialized_end=3912
  _globals['_RESTAPICLIENTCONFIG']._serialized_start=3914
  _globals['_RESTAPICLIENTCONFIG']._serialized_end=4030
  _globals['_GRPCCLIENTCONFIG']._serialized_start=4032
  _globals['_GRPCCLIENTCONFIG']._serialized_end=4145
  _globals['_LLAMAEMBEDDINGMODELCONFIG']._serialized_start=4147
  _globals['_LLAMAEMBEDDINGMODELCONFIG']._serialized_end=4227
  _globals['_LLAMATEXTCONFIG']._serialized_start=4229
  _globals['_LLAMATEXTCONFIG']._serialized_end=4291
  _globals['_CLIPEMBEDDINGCONFIG']._serialized_start=4293
  _globals['_CLIPEMBEDDINGCONFIG']._serialized_end=4359
  _globals['_MODELCONFIG']._serialized_start=4362
  _globals['_MODELCONFIG']._serialized_end=6761
  _globals['_FEATURERETRIEVALCONFIG']._serialized_start=6763
  _globals['_FEATURERETRIEVALCONFIG']._serialized_end=6857
  _globals['_KAFKAPRODUCERCONFIG']._serialized_start=6859
  _globals['_KAFKAPRODUCERCONFIG']._serialized_end=6973
  _globals['_KAFKACONSUMERCONFIG']._serialized_start=6975
  _globals['_KAFKACONSUMERCONFIG']._serialized_end=7093
  _globals['_INPUTCONFIG']._serialized_start=7096
  _globals['_INPUTCONFIG']._serialized_end=7243
  _globals['_LEARNERCONFIG']._serialized_start=7245
  _globals['_LEARNERCONFIG']._serialized_end=7344
  _globals['_TURBOMLCONFIG']._serialized_start=7347
  _globals['_TURBOMLCONFIG']._serialized_end=7943
  _globals['_TRAINJOBCONFIG']._serialized_start=7946
  _globals['_TRAINJOBCONFIG']._serialized_end=8118
  _globals['_MODELCONFIGLIST']._serialized_start=8120
  _globals['_MODELCONFIGLIST']._serialized_end=8181
# @@protoc_insertion_point(module_scope)

```

#### config_pb2.pyi
```python
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MStreamConfig(_message.Message):
    __slots__ = ("num_rows", "num_buckets", "factor")
    NUM_ROWS_FIELD_NUMBER: _ClassVar[int]
    NUM_BUCKETS_FIELD_NUMBER: _ClassVar[int]
    FACTOR_FIELD_NUMBER: _ClassVar[int]
    num_rows: int
    num_buckets: int
    factor: float
    def __init__(self, num_rows: _Optional[int] = ..., num_buckets: _Optional[int] = ..., factor: _Optional[float] = ...) -> None: ...

class RCFConfig(_message.Message):
    __slots__ = ("time_decay", "number_of_trees", "output_after", "sample_size")
    TIME_DECAY_FIELD_NUMBER: _ClassVar[int]
    NUMBER_OF_TREES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_AFTER_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_SIZE_FIELD_NUMBER: _ClassVar[int]
    time_decay: float
    number_of_trees: int
    output_after: int
    sample_size: int
    def __init__(self, time_decay: _Optional[float] = ..., number_of_trees: _Optional[int] = ..., output_after: _Optional[int] = ..., sample_size: _Optional[int] = ...) -> None: ...

class HSTConfig(_message.Message):
    __slots__ = ("n_trees", "height", "window_size")
    N_TREES_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WINDOW_SIZE_FIELD_NUMBER: _ClassVar[int]
    n_trees: int
    height: int
    window_size: int
    def __init__(self, n_trees: _Optional[int] = ..., height: _Optional[int] = ..., window_size: _Optional[int] = ...) -> None: ...

class ONNXConfig(_message.Message):
    __slots__ = ("model_save_name", "model_data")
    MODEL_SAVE_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_DATA_FIELD_NUMBER: _ClassVar[int]
    model_save_name: str
    model_data: bytes
    def __init__(self, model_save_name: _Optional[str] = ..., model_data: _Optional[bytes] = ...) -> None: ...

class HoeffdingClassifierConfig(_message.Message):
    __slots__ = ("delta", "tau", "grace_period", "n_classes", "leaf_pred_method", "split_method")
    DELTA_FIELD_NUMBER: _ClassVar[int]
    TAU_FIELD_NUMBER: _ClassVar[int]
    GRACE_PERIOD_FIELD_NUMBER: _ClassVar[int]
    N_CLASSES_FIELD_NUMBER: _ClassVar[int]
    LEAF_PRED_METHOD_FIELD_NUMBER: _ClassVar[int]
    SPLIT_METHOD_FIELD_NUMBER: _ClassVar[int]
    delta: float
    tau: float
    grace_period: int
    n_classes: int
    leaf_pred_method: str
    split_method: str
    def __init__(self, delta: _Optional[float] = ..., tau: _Optional[float] = ..., grace_period: _Optional[int] = ..., n_classes: _Optional[int] = ..., leaf_pred_method: _Optional[str] = ..., split_method: _Optional[str] = ...) -> None: ...

class HoeffdingRegressorConfig(_message.Message):
    __slots__ = ("delta", "tau", "grace_period", "leaf_pred_method")
    DELTA_FIELD_NUMBER: _ClassVar[int]
    TAU_FIELD_NUMBER: _ClassVar[int]
    GRACE_PERIOD_FIELD_NUMBER: _ClassVar[int]
    LEAF_PRED_METHOD_FIELD_NUMBER: _ClassVar[int]
    delta: float
    tau: float
    grace_period: int
    leaf_pred_method: str
    def __init__(self, delta: _Optional[float] = ..., tau: _Optional[float] = ..., grace_period: _Optional[int] = ..., leaf_pred_method: _Optional[str] = ...) -> None: ...

class AMFClassifierConfig(_message.Message):
    __slots__ = ("n_classes", "n_estimators", "step", "use_aggregation", "dirichlet", "split_pure")
    N_CLASSES_FIELD_NUMBER: _ClassVar[int]
    N_ESTIMATORS_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    USE_AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    DIRICHLET_FIELD_NUMBER: _ClassVar[int]
    SPLIT_PURE_FIELD_NUMBER: _ClassVar[int]
    n_classes: int
    n_estimators: int
    step: float
    use_aggregation: bool
    dirichlet: float
    split_pure: bool
    def __init__(self, n_classes: _Optional[int] = ..., n_estimators: _Optional[int] = ..., step: _Optional[float] = ..., use_aggregation: bool = ..., dirichlet: _Optional[float] = ..., split_pure: bool = ...) -> None: ...

class AMFRegressorConfig(_message.Message):
    __slots__ = ("n_estimators", "step", "use_aggregation", "dirichlet")
    N_ESTIMATORS_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    USE_AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    DIRICHLET_FIELD_NUMBER: _ClassVar[int]
    n_estimators: int
    step: float
    use_aggregation: bool
    dirichlet: float
    def __init__(self, n_estimators: _Optional[int] = ..., step: _Optional[float] = ..., use_aggregation: bool = ..., dirichlet: _Optional[float] = ...) -> None: ...

class FFMClassifierConfig(_message.Message):
    __slots__ = ("n_factors", "l1_weight", "l2_weight", "l1_latent", "l2_latent", "intercept", "intercept_lr", "clip_gradient")
    N_FACTORS_FIELD_NUMBER: _ClassVar[int]
    L1_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    L2_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    L1_LATENT_FIELD_NUMBER: _ClassVar[int]
    L2_LATENT_FIELD_NUMBER: _ClassVar[int]
    INTERCEPT_FIELD_NUMBER: _ClassVar[int]
    INTERCEPT_LR_FIELD_NUMBER: _ClassVar[int]
    CLIP_GRADIENT_FIELD_NUMBER: _ClassVar[int]
    n_factors: int
    l1_weight: float
    l2_weight: float
    l1_latent: float
    l2_latent: float
    intercept: float
    intercept_lr: float
    clip_gradient: float
    def __init__(self, n_factors: _Optional[int] = ..., l1_weight: _Optional[float] = ..., l2_weight: _Optional[float] = ..., l1_latent: _Optional[float] = ..., l2_latent: _Optional[float] = ..., intercept: _Optional[float] = ..., intercept_lr: _Optional[float] = ..., clip_gradient: _Optional[float] = ...) -> None: ...

class FFMRegressorConfig(_message.Message):
    __slots__ = ("n_factors", "l1_weight", "l2_weight", "l1_latent", "l2_latent", "intercept", "intercept_lr", "clip_gradient")
    N_FACTORS_FIELD_NUMBER: _ClassVar[int]
    L1_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    L2_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    L1_LATENT_FIELD_NUMBER: _ClassVar[int]
    L2_LATENT_FIELD_NUMBER: _ClassVar[int]
    INTERCEPT_FIELD_NUMBER: _ClassVar[int]
    INTERCEPT_LR_FIELD_NUMBER: _ClassVar[int]
    CLIP_GRADIENT_FIELD_NUMBER: _ClassVar[int]
    n_factors: int
    l1_weight: float
    l2_weight: float
    l1_latent: float
    l2_latent: float
    intercept: float
    intercept_lr: float
    clip_gradient: float
    def __init__(self, n_factors: _Optional[int] = ..., l1_weight: _Optional[float] = ..., l2_weight: _Optional[float] = ..., l1_latent: _Optional[float] = ..., l2_latent: _Optional[float] = ..., intercept: _Optional[float] = ..., intercept_lr: _Optional[float] = ..., clip_gradient: _Optional[float] = ...) -> None: ...

class SNARIMAXConfig(_message.Message):
    __slots__ = ("horizon", "p", "d", "q", "m", "sp", "sd", "sq", "num_children")
    HORIZON_FIELD_NUMBER: _ClassVar[int]
    P_FIELD_NUMBER: _ClassVar[int]
    D_FIELD_NUMBER: _ClassVar[int]
    Q_FIELD_NUMBER: _ClassVar[int]
    M_FIELD_NUMBER: _ClassVar[int]
    SP_FIELD_NUMBER: _ClassVar[int]
    SD_FIELD_NUMBER: _ClassVar[int]
    SQ_FIELD_NUMBER: _ClassVar[int]
    NUM_CHILDREN_FIELD_NUMBER: _ClassVar[int]
    horizon: int
    p: int
    d: int
    q: int
    m: int
    sp: int
    sd: int
    sq: int
    num_children: int
    def __init__(self, horizon: _Optional[int] = ..., p: _Optional[int] = ..., d: _Optional[int] = ..., q: _Optional[int] = ..., m: _Optional[int] = ..., sp: _Optional[int] = ..., sd: _Optional[int] = ..., sq: _Optional[int] = ..., num_children: _Optional[int] = ...) -> None: ...

class NeuralNetworkConfig(_message.Message):
    __slots__ = ("layers", "dropout", "loss_function", "optimizer", "learning_rate", "batch_size")
    class NeuralNetworkLayer(_message.Message):
        __slots__ = ("input_size", "output_size", "activation", "dropout", "residual_connections", "use_bias")
        INPUT_SIZE_FIELD_NUMBER: _ClassVar[int]
        OUTPUT_SIZE_FIELD_NUMBER: _ClassVar[int]
        ACTIVATION_FIELD_NUMBER: _ClassVar[int]
        DROPOUT_FIELD_NUMBER: _ClassVar[int]
        RESIDUAL_CONNECTIONS_FIELD_NUMBER: _ClassVar[int]
        USE_BIAS_FIELD_NUMBER: _ClassVar[int]
        input_size: int
        output_size: int
        activation: str
        dropout: float
        residual_connections: _containers.RepeatedScalarFieldContainer[int]
        use_bias: bool
        def __init__(self, input_size: _Optional[int] = ..., output_size: _Optional[int] = ..., activation: _Optional[str] = ..., dropout: _Optional[float] = ..., residual_connections: _Optional[_Iterable[int]] = ..., use_bias: bool = ...) -> None: ...
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    DROPOUT_FIELD_NUMBER: _ClassVar[int]
    LOSS_FUNCTION_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    layers: _containers.RepeatedCompositeFieldContainer[NeuralNetworkConfig.NeuralNetworkLayer]
    dropout: float
    loss_function: str
    optimizer: str
    learning_rate: float
    batch_size: int
    def __init__(self, layers: _Optional[_Iterable[_Union[NeuralNetworkConfig.NeuralNetworkLayer, _Mapping]]] = ..., dropout: _Optional[float] = ..., loss_function: _Optional[str] = ..., optimizer: _Optional[str] = ..., learning_rate: _Optional[float] = ..., batch_size: _Optional[int] = ...) -> None: ...

class ONNConfig(_message.Message):
    __slots__ = ("max_num_hidden_layers", "qtd_neuron_hidden_layer", "n_classes", "b", "n", "s")
    MAX_NUM_HIDDEN_LAYERS_FIELD_NUMBER: _ClassVar[int]
    QTD_NEURON_HIDDEN_LAYER_FIELD_NUMBER: _ClassVar[int]
    N_CLASSES_FIELD_NUMBER: _ClassVar[int]
    B_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    S_FIELD_NUMBER: _ClassVar[int]
    max_num_hidden_layers: int
    qtd_neuron_hidden_layer: int
    n_classes: int
    b: float
    n: float
    s: float
    def __init__(self, max_num_hidden_layers: _Optional[int] = ..., qtd_neuron_hidden_layer: _Optional[int] = ..., n_classes: _Optional[int] = ..., b: _Optional[float] = ..., n: _Optional[float] = ..., s: _Optional[float] = ...) -> None: ...

class OVRConfig(_message.Message):
    __slots__ = ("n_classes", "num_children")
    N_CLASSES_FIELD_NUMBER: _ClassVar[int]
    NUM_CHILDREN_FIELD_NUMBER: _ClassVar[int]
    n_classes: int
    num_children: int
    def __init__(self, n_classes: _Optional[int] = ..., num_children: _Optional[int] = ...) -> None: ...

class RandomProjectionEmbeddingConfig(_message.Message):
    __slots__ = ("n_embeddings", "type_embedding")
    N_EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    TYPE_EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    n_embeddings: int
    type_embedding: str
    def __init__(self, n_embeddings: _Optional[int] = ..., type_embedding: _Optional[str] = ...) -> None: ...

class EmbeddingModelConfig(_message.Message):
    __slots__ = ("num_children",)
    NUM_CHILDREN_FIELD_NUMBER: _ClassVar[int]
    num_children: int
    def __init__(self, num_children: _Optional[int] = ...) -> None: ...

class BanditModelSelectionConfig(_message.Message):
    __slots__ = ("bandit", "metric_name")
    BANDIT_FIELD_NUMBER: _ClassVar[int]
    METRIC_NAME_FIELD_NUMBER: _ClassVar[int]
    bandit: str
    metric_name: str
    def __init__(self, bandit: _Optional[str] = ..., metric_name: _Optional[str] = ...) -> None: ...

class ContextualBanditModelSelectionConfig(_message.Message):
    __slots__ = ("contextualbandit", "metric_name")
    CONTEXTUALBANDIT_FIELD_NUMBER: _ClassVar[int]
    METRIC_NAME_FIELD_NUMBER: _ClassVar[int]
    contextualbandit: str
    metric_name: str
    def __init__(self, contextualbandit: _Optional[str] = ..., metric_name: _Optional[str] = ...) -> None: ...

class LeveragingBaggingClassifierConfig(_message.Message):
    __slots__ = ("n_models", "n_classes", "w", "bagging_method", "seed", "num_children")
    N_MODELS_FIELD_NUMBER: _ClassVar[int]
    N_CLASSES_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    BAGGING_METHOD_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    NUM_CHILDREN_FIELD_NUMBER: _ClassVar[int]
    n_models: int
    n_classes: int
    w: float
    bagging_method: str
    seed: int
    num_children: int
    def __init__(self, n_models: _Optional[int] = ..., n_classes: _Optional[int] = ..., w: _Optional[float] = ..., bagging_method: _Optional[str] = ..., seed: _Optional[int] = ..., num_children: _Optional[int] = ...) -> None: ...

class HeteroLeveragingBaggingClassifierConfig(_message.Message):
    __slots__ = ("n_classes", "w", "bagging_method", "seed", "num_children")
    N_CLASSES_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    BAGGING_METHOD_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    NUM_CHILDREN_FIELD_NUMBER: _ClassVar[int]
    n_classes: int
    w: float
    bagging_method: str
    seed: int
    num_children: int
    def __init__(self, n_classes: _Optional[int] = ..., w: _Optional[float] = ..., bagging_method: _Optional[str] = ..., seed: _Optional[int] = ..., num_children: _Optional[int] = ...) -> None: ...

class AdaBoostClassifierConfig(_message.Message):
    __slots__ = ("n_models", "n_classes", "seed", "num_children")
    N_MODELS_FIELD_NUMBER: _ClassVar[int]
    N_CLASSES_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    NUM_CHILDREN_FIELD_NUMBER: _ClassVar[int]
    n_models: int
    n_classes: int
    seed: int
    num_children: int
    def __init__(self, n_models: _Optional[int] = ..., n_classes: _Optional[int] = ..., seed: _Optional[int] = ..., num_children: _Optional[int] = ...) -> None: ...

class HeteroAdaBoostClassifierConfig(_message.Message):
    __slots__ = ("n_classes", "seed", "num_children")
    N_CLASSES_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    NUM_CHILDREN_FIELD_NUMBER: _ClassVar[int]
    n_classes: int
    seed: int
    num_children: int
    def __init__(self, n_classes: _Optional[int] = ..., seed: _Optional[int] = ..., num_children: _Optional[int] = ...) -> None: ...

class SGTClassifierConfig(_message.Message):
    __slots__ = ("delta", "grace_period", "gamma")
    DELTA_FIELD_NUMBER: _ClassVar[int]
    GRACE_PERIOD_FIELD_NUMBER: _ClassVar[int]
    LAMBDA_FIELD_NUMBER: _ClassVar[int]
    GAMMA_FIELD_NUMBER: _ClassVar[int]
    delta: float
    grace_period: int
    gamma: float
    def __init__(self, delta: _Optional[float] = ..., grace_period: _Optional[int] = ..., gamma: _Optional[float] = ..., **kwargs) -> None: ...

class SGTRegressorConfig(_message.Message):
    __slots__ = ("delta", "grace_period", "gamma")
    DELTA_FIELD_NUMBER: _ClassVar[int]
    GRACE_PERIOD_FIELD_NUMBER: _ClassVar[int]
    LAMBDA_FIELD_NUMBER: _ClassVar[int]
    GAMMA_FIELD_NUMBER: _ClassVar[int]
    delta: float
    grace_period: int
    gamma: float
    def __init__(self, delta: _Optional[float] = ..., grace_period: _Optional[int] = ..., gamma: _Optional[float] = ..., **kwargs) -> None: ...

class RandomSamplerConfig(_message.Message):
    __slots__ = ("n_classes", "desired_dist", "sampling_method", "sampling_rate", "seed", "num_children")
    N_CLASSES_FIELD_NUMBER: _ClassVar[int]
    DESIRED_DIST_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_METHOD_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_RATE_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    NUM_CHILDREN_FIELD_NUMBER: _ClassVar[int]
    n_classes: int
    desired_dist: _containers.RepeatedScalarFieldContainer[float]
    sampling_method: str
    sampling_rate: float
    seed: int
    num_children: int
    def __init__(self, n_classes: _Optional[int] = ..., desired_dist: _Optional[_Iterable[float]] = ..., sampling_method: _Optional[str] = ..., sampling_rate: _Optional[float] = ..., seed: _Optional[int] = ..., num_children: _Optional[int] = ...) -> None: ...

class MultinomialConfig(_message.Message):
    __slots__ = ("n_classes", "alpha")
    N_CLASSES_FIELD_NUMBER: _ClassVar[int]
    ALPHA_FIELD_NUMBER: _ClassVar[int]
    n_classes: int
    alpha: float
    def __init__(self, n_classes: _Optional[int] = ..., alpha: _Optional[float] = ...) -> None: ...

class GaussianConfig(_message.Message):
    __slots__ = ("n_classes",)
    N_CLASSES_FIELD_NUMBER: _ClassVar[int]
    n_classes: int
    def __init__(self, n_classes: _Optional[int] = ...) -> None: ...

class PythonConfig(_message.Message):
    __slots__ = ("module_name", "class_name", "venv_name", "code")
    MODULE_NAME_FIELD_NUMBER: _ClassVar[int]
    CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
    VENV_NAME_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    module_name: str
    class_name: str
    venv_name: str
    code: str
    def __init__(self, module_name: _Optional[str] = ..., class_name: _Optional[str] = ..., venv_name: _Optional[str] = ..., code: _Optional[str] = ...) -> None: ...

class PythonEnsembleConfig(_message.Message):
    __slots__ = ("module_name", "class_name", "venv_name", "code")
    MODULE_NAME_FIELD_NUMBER: _ClassVar[int]
    CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
    VENV_NAME_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    module_name: str
    class_name: str
    venv_name: str
    code: str
    def __init__(self, module_name: _Optional[str] = ..., class_name: _Optional[str] = ..., venv_name: _Optional[str] = ..., code: _Optional[str] = ...) -> None: ...

class PreProcessorConfig(_message.Message):
    __slots__ = ("preprocessor_name", "text_categories", "num_children", "gguf_model_id", "max_tokens_per_input", "image_sizes", "channel_first")
    PREPROCESSOR_NAME_FIELD_NUMBER: _ClassVar[int]
    TEXT_CATEGORIES_FIELD_NUMBER: _ClassVar[int]
    NUM_CHILDREN_FIELD_NUMBER: _ClassVar[int]
    GGUF_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_PER_INPUT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_SIZES_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIRST_FIELD_NUMBER: _ClassVar[int]
    preprocessor_name: str
    text_categories: _containers.RepeatedScalarFieldContainer[int]
    num_children: int
    gguf_model_id: str
    max_tokens_per_input: int
    image_sizes: _containers.RepeatedScalarFieldContainer[int]
    channel_first: bool
    def __init__(self, preprocessor_name: _Optional[str] = ..., text_categories: _Optional[_Iterable[int]] = ..., num_children: _Optional[int] = ..., gguf_model_id: _Optional[str] = ..., max_tokens_per_input: _Optional[int] = ..., image_sizes: _Optional[_Iterable[int]] = ..., channel_first: bool = ...) -> None: ...

class AdaptiveXGBoostConfig(_message.Message):
    __slots__ = ("n_classes", "learning_rate", "max_depth", "max_window_size", "min_window_size", "max_buffer", "pre_train", "detect_drift", "use_updater", "trees_per_train", "percent_update_trees")
    N_CLASSES_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    MAX_DEPTH_FIELD_NUMBER: _ClassVar[int]
    MAX_WINDOW_SIZE_FIELD_NUMBER: _ClassVar[int]
    MIN_WINDOW_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAX_BUFFER_FIELD_NUMBER: _ClassVar[int]
    PRE_TRAIN_FIELD_NUMBER: _ClassVar[int]
    DETECT_DRIFT_FIELD_NUMBER: _ClassVar[int]
    USE_UPDATER_FIELD_NUMBER: _ClassVar[int]
    TREES_PER_TRAIN_FIELD_NUMBER: _ClassVar[int]
    PERCENT_UPDATE_TREES_FIELD_NUMBER: _ClassVar[int]
    n_classes: int
    learning_rate: float
    max_depth: int
    max_window_size: int
    min_window_size: int
    max_buffer: int
    pre_train: int
    detect_drift: bool
    use_updater: bool
    trees_per_train: int
    percent_update_trees: float
    def __init__(self, n_classes: _Optional[int] = ..., learning_rate: _Optional[float] = ..., max_depth: _Optional[int] = ..., max_window_size: _Optional[int] = ..., min_window_size: _Optional[int] = ..., max_buffer: _Optional[int] = ..., pre_train: _Optional[int] = ..., detect_drift: bool = ..., use_updater: bool = ..., trees_per_train: _Optional[int] = ..., percent_update_trees: _Optional[float] = ...) -> None: ...

class AdaptiveLGBMConfig(_message.Message):
    __slots__ = ("n_classes", "learning_rate", "max_depth", "max_window_size", "min_window_size", "max_buffer", "pre_train", "detect_drift", "use_updater", "trees_per_train")
    N_CLASSES_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    MAX_DEPTH_FIELD_NUMBER: _ClassVar[int]
    MAX_WINDOW_SIZE_FIELD_NUMBER: _ClassVar[int]
    MIN_WINDOW_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAX_BUFFER_FIELD_NUMBER: _ClassVar[int]
    PRE_TRAIN_FIELD_NUMBER: _ClassVar[int]
    DETECT_DRIFT_FIELD_NUMBER: _ClassVar[int]
    USE_UPDATER_FIELD_NUMBER: _ClassVar[int]
    TREES_PER_TRAIN_FIELD_NUMBER: _ClassVar[int]
    n_classes: int
    learning_rate: float
    max_depth: int
    max_window_size: int
    min_window_size: int
    max_buffer: int
    pre_train: int
    detect_drift: bool
    use_updater: bool
    trees_per_train: int
    def __init__(self, n_classes: _Optional[int] = ..., learning_rate: _Optional[float] = ..., max_depth: _Optional[int] = ..., max_window_size: _Optional[int] = ..., min_window_size: _Optional[int] = ..., max_buffer: _Optional[int] = ..., pre_train: _Optional[int] = ..., detect_drift: bool = ..., use_updater: bool = ..., trees_per_train: _Optional[int] = ...) -> None: ...

class RestAPIClientConfig(_message.Message):
    __slots__ = ("server_url", "max_retries", "connection_timeout", "max_request_time")
    SERVER_URL_FIELD_NUMBER: _ClassVar[int]
    MAX_RETRIES_FIELD_NUMBER: _ClassVar[int]
    CONNECTION_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    MAX_REQUEST_TIME_FIELD_NUMBER: _ClassVar[int]
    server_url: str
    max_retries: int
    connection_timeout: int
    max_request_time: int
    def __init__(self, server_url: _Optional[str] = ..., max_retries: _Optional[int] = ..., connection_timeout: _Optional[int] = ..., max_request_time: _Optional[int] = ...) -> None: ...

class GRPCClientConfig(_message.Message):
    __slots__ = ("server_url", "max_retries", "connection_timeout", "max_request_time")
    SERVER_URL_FIELD_NUMBER: _ClassVar[int]
    MAX_RETRIES_FIELD_NUMBER: _ClassVar[int]
    CONNECTION_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    MAX_REQUEST_TIME_FIELD_NUMBER: _ClassVar[int]
    server_url: str
    max_retries: int
    connection_timeout: int
    max_request_time: int
    def __init__(self, server_url: _Optional[str] = ..., max_retries: _Optional[int] = ..., connection_timeout: _Optional[int] = ..., max_request_time: _Optional[int] = ...) -> None: ...

class LLAMAEmbeddingModelConfig(_message.Message):
    __slots__ = ("gguf_model_id", "max_tokens_per_input")
    GGUF_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_PER_INPUT_FIELD_NUMBER: _ClassVar[int]
    gguf_model_id: str
    max_tokens_per_input: int
    def __init__(self, gguf_model_id: _Optional[str] = ..., max_tokens_per_input: _Optional[int] = ...) -> None: ...

class LlamaTextConfig(_message.Message):
    __slots__ = ("gguf_model_id", "num_children")
    GGUF_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_CHILDREN_FIELD_NUMBER: _ClassVar[int]
    gguf_model_id: str
    num_children: int
    def __init__(self, gguf_model_id: _Optional[str] = ..., num_children: _Optional[int] = ...) -> None: ...

class ClipEmbeddingConfig(_message.Message):
    __slots__ = ("gguf_model_id", "num_children")
    GGUF_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_CHILDREN_FIELD_NUMBER: _ClassVar[int]
    gguf_model_id: str
    num_children: int
    def __init__(self, gguf_model_id: _Optional[str] = ..., num_children: _Optional[int] = ...) -> None: ...

class ModelConfig(_message.Message):
    __slots__ = ("algorithm", "num_children", "mstream_config", "rcf_config", "hst_config", "onnx_config", "hoeffding_classifier_config", "hoeffding_regressor_config", "amf_classifier_config", "amf_regressor_config", "ffm_classifier_config", "ffm_regressor_config", "snarimax_config", "nn_config", "onn_config", "leveraging_bagging_classifier_config", "adaboost_classifier_config", "random_sampler_config", "bandit_model_selection_config", "contextual_bandit_model_selection_config", "python_config", "preprocessor_config", "ovr_model_selection_config", "random_projection_config", "embedding_model_config", "hetero_leveraging_bagging_classifier_config", "hetero_adaboost_classifier_config", "sgt_classifier_config", "sgt_regressor_config", "multinomial_config", "gaussian_config", "adaptive_xgboost_config", "adaptive_lgbm_config", "llama_embedding_config", "rest_api_client_config", "llama_text_config", "clip_embedding_config", "python_ensemble_config", "grpc_client_config")
    ALGORITHM_FIELD_NUMBER: _ClassVar[int]
    NUM_CHILDREN_FIELD_NUMBER: _ClassVar[int]
    MSTREAM_CONFIG_FIELD_NUMBER: _ClassVar[int]
    RCF_CONFIG_FIELD_NUMBER: _ClassVar[int]
    HST_CONFIG_FIELD_NUMBER: _ClassVar[int]
    ONNX_CONFIG_FIELD_NUMBER: _ClassVar[int]
    HOEFFDING_CLASSIFIER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    HOEFFDING_REGRESSOR_CONFIG_FIELD_NUMBER: _ClassVar[int]
    AMF_CLASSIFIER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    AMF_REGRESSOR_CONFIG_FIELD_NUMBER: _ClassVar[int]
    FFM_CLASSIFIER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    FFM_REGRESSOR_CONFIG_FIELD_NUMBER: _ClassVar[int]
    SNARIMAX_CONFIG_FIELD_NUMBER: _ClassVar[int]
    NN_CONFIG_FIELD_NUMBER: _ClassVar[int]
    ONN_CONFIG_FIELD_NUMBER: _ClassVar[int]
    LEVERAGING_BAGGING_CLASSIFIER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    ADABOOST_CLASSIFIER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    RANDOM_SAMPLER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    BANDIT_MODEL_SELECTION_CONFIG_FIELD_NUMBER: _ClassVar[int]
    CONTEXTUAL_BANDIT_MODEL_SELECTION_CONFIG_FIELD_NUMBER: _ClassVar[int]
    PYTHON_CONFIG_FIELD_NUMBER: _ClassVar[int]
    PREPROCESSOR_CONFIG_FIELD_NUMBER: _ClassVar[int]
    OVR_MODEL_SELECTION_CONFIG_FIELD_NUMBER: _ClassVar[int]
    RANDOM_PROJECTION_CONFIG_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_MODEL_CONFIG_FIELD_NUMBER: _ClassVar[int]
    HETERO_LEVERAGING_BAGGING_CLASSIFIER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    HETERO_ADABOOST_CLASSIFIER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    SGT_CLASSIFIER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    SGT_REGRESSOR_CONFIG_FIELD_NUMBER: _ClassVar[int]
    MULTINOMIAL_CONFIG_FIELD_NUMBER: _ClassVar[int]
    GAUSSIAN_CONFIG_FIELD_NUMBER: _ClassVar[int]
    ADAPTIVE_XGBOOST_CONFIG_FIELD_NUMBER: _ClassVar[int]
    ADAPTIVE_LGBM_CONFIG_FIELD_NUMBER: _ClassVar[int]
    LLAMA_EMBEDDING_CONFIG_FIELD_NUMBER: _ClassVar[int]
    REST_API_CLIENT_CONFIG_FIELD_NUMBER: _ClassVar[int]
    LLAMA_TEXT_CONFIG_FIELD_NUMBER: _ClassVar[int]
    CLIP_EMBEDDING_CONFIG_FIELD_NUMBER: _ClassVar[int]
    PYTHON_ENSEMBLE_CONFIG_FIELD_NUMBER: _ClassVar[int]
    GRPC_CLIENT_CONFIG_FIELD_NUMBER: _ClassVar[int]
    algorithm: str
    num_children: int
    mstream_config: MStreamConfig
    rcf_config: RCFConfig
    hst_config: HSTConfig
    onnx_config: ONNXConfig
    hoeffding_classifier_config: HoeffdingClassifierConfig
    hoeffding_regressor_config: HoeffdingRegressorConfig
    amf_classifier_config: AMFClassifierConfig
    amf_regressor_config: AMFRegressorConfig
    ffm_classifier_config: FFMClassifierConfig
    ffm_regressor_config: FFMRegressorConfig
    snarimax_config: SNARIMAXConfig
    nn_config: NeuralNetworkConfig
    onn_config: ONNConfig
    leveraging_bagging_classifier_config: LeveragingBaggingClassifierConfig
    adaboost_classifier_config: AdaBoostClassifierConfig
    random_sampler_config: RandomSamplerConfig
    bandit_model_selection_config: BanditModelSelectionConfig
    contextual_bandit_model_selection_config: ContextualBanditModelSelectionConfig
    python_config: PythonConfig
    preprocessor_config: PreProcessorConfig
    ovr_model_selection_config: OVRConfig
    random_projection_config: RandomProjectionEmbeddingConfig
    embedding_model_config: EmbeddingModelConfig
    hetero_leveraging_bagging_classifier_config: HeteroLeveragingBaggingClassifierConfig
    hetero_adaboost_classifier_config: HeteroAdaBoostClassifierConfig
    sgt_classifier_config: SGTClassifierConfig
    sgt_regressor_config: SGTRegressorConfig
    multinomial_config: MultinomialConfig
    gaussian_config: GaussianConfig
    adaptive_xgboost_config: AdaptiveXGBoostConfig
    adaptive_lgbm_config: AdaptiveLGBMConfig
    llama_embedding_config: LLAMAEmbeddingModelConfig
    rest_api_client_config: RestAPIClientConfig
    llama_text_config: LlamaTextConfig
    clip_embedding_config: ClipEmbeddingConfig
    python_ensemble_config: PythonEnsembleConfig
    grpc_client_config: GRPCClientConfig
    def __init__(self, algorithm: _Optional[str] = ..., num_children: _Optional[int] = ..., mstream_config: _Optional[_Union[MStreamConfig, _Mapping]] = ..., rcf_config: _Optional[_Union[RCFConfig, _Mapping]] = ..., hst_config: _Optional[_Union[HSTConfig, _Mapping]] = ..., onnx_config: _Optional[_Union[ONNXConfig, _Mapping]] = ..., hoeffding_classifier_config: _Optional[_Union[HoeffdingClassifierConfig, _Mapping]] = ..., hoeffding_regressor_config: _Optional[_Union[HoeffdingRegressorConfig, _Mapping]] = ..., amf_classifier_config: _Optional[_Union[AMFClassifierConfig, _Mapping]] = ..., amf_regressor_config: _Optional[_Union[AMFRegressorConfig, _Mapping]] = ..., ffm_classifier_config: _Optional[_Union[FFMClassifierConfig, _Mapping]] = ..., ffm_regressor_config: _Optional[_Union[FFMRegressorConfig, _Mapping]] = ..., snarimax_config: _Optional[_Union[SNARIMAXConfig, _Mapping]] = ..., nn_config: _Optional[_Union[NeuralNetworkConfig, _Mapping]] = ..., onn_config: _Optional[_Union[ONNConfig, _Mapping]] = ..., leveraging_bagging_classifier_config: _Optional[_Union[LeveragingBaggingClassifierConfig, _Mapping]] = ..., adaboost_classifier_config: _Optional[_Union[AdaBoostClassifierConfig, _Mapping]] = ..., random_sampler_config: _Optional[_Union[RandomSamplerConfig, _Mapping]] = ..., bandit_model_selection_config: _Optional[_Union[BanditModelSelectionConfig, _Mapping]] = ..., contextual_bandit_model_selection_config: _Optional[_Union[ContextualBanditModelSelectionConfig, _Mapping]] = ..., python_config: _Optional[_Union[PythonConfig, _Mapping]] = ..., preprocessor_config: _Optional[_Union[PreProcessorConfig, _Mapping]] = ..., ovr_model_selection_config: _Optional[_Union[OVRConfig, _Mapping]] = ..., random_projection_config: _Optional[_Union[RandomProjectionEmbeddingConfig, _Mapping]] = ..., embedding_model_config: _Optional[_Union[EmbeddingModelConfig, _Mapping]] = ..., hetero_leveraging_bagging_classifier_config: _Optional[_Union[HeteroLeveragingBaggingClassifierConfig, _Mapping]] = ..., hetero_adaboost_classifier_config: _Optional[_Union[HeteroAdaBoostClassifierConfig, _Mapping]] = ..., sgt_classifier_config: _Optional[_Union[SGTClassifierConfig, _Mapping]] = ..., sgt_regressor_config: _Optional[_Union[SGTRegressorConfig, _Mapping]] = ..., multinomial_config: _Optional[_Union[MultinomialConfig, _Mapping]] = ..., gaussian_config: _Optional[_Union[GaussianConfig, _Mapping]] = ..., adaptive_xgboost_config: _Optional[_Union[AdaptiveXGBoostConfig, _Mapping]] = ..., adaptive_lgbm_config: _Optional[_Union[AdaptiveLGBMConfig, _Mapping]] = ..., llama_embedding_config: _Optional[_Union[LLAMAEmbeddingModelConfig, _Mapping]] = ..., rest_api_client_config: _Optional[_Union[RestAPIClientConfig, _Mapping]] = ..., llama_text_config: _Optional[_Union[LlamaTextConfig, _Mapping]] = ..., clip_embedding_config: _Optional[_Union[ClipEmbeddingConfig, _Mapping]] = ..., python_ensemble_config: _Optional[_Union[PythonEnsembleConfig, _Mapping]] = ..., grpc_client_config: _Optional[_Union[GRPCClientConfig, _Mapping]] = ...) -> None: ...

class FeatureRetrievalConfig(_message.Message):
    __slots__ = ("sql_statement", "placeholder_cols", "result_cols")
    SQL_STATEMENT_FIELD_NUMBER: _ClassVar[int]
    PLACEHOLDER_COLS_FIELD_NUMBER: _ClassVar[int]
    RESULT_COLS_FIELD_NUMBER: _ClassVar[int]
    sql_statement: str
    placeholder_cols: _containers.RepeatedScalarFieldContainer[str]
    result_cols: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, sql_statement: _Optional[str] = ..., placeholder_cols: _Optional[_Iterable[str]] = ..., result_cols: _Optional[_Iterable[str]] = ...) -> None: ...

class KafkaProducerConfig(_message.Message):
    __slots__ = ("write_topic", "proto_file_name", "proto_message_name", "schema_id")
    WRITE_TOPIC_FIELD_NUMBER: _ClassVar[int]
    PROTO_FILE_NAME_FIELD_NUMBER: _ClassVar[int]
    PROTO_MESSAGE_NAME_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_ID_FIELD_NUMBER: _ClassVar[int]
    write_topic: str
    proto_file_name: str
    proto_message_name: str
    schema_id: int
    def __init__(self, write_topic: _Optional[str] = ..., proto_file_name: _Optional[str] = ..., proto_message_name: _Optional[str] = ..., schema_id: _Optional[int] = ...) -> None: ...

class KafkaConsumerConfig(_message.Message):
    __slots__ = ("read_topic", "proto_file_name", "proto_message_name", "consumer_group")
    READ_TOPIC_FIELD_NUMBER: _ClassVar[int]
    PROTO_FILE_NAME_FIELD_NUMBER: _ClassVar[int]
    PROTO_MESSAGE_NAME_FIELD_NUMBER: _ClassVar[int]
    CONSUMER_GROUP_FIELD_NUMBER: _ClassVar[int]
    read_topic: str
    proto_file_name: str
    proto_message_name: str
    consumer_group: str
    def __init__(self, read_topic: _Optional[str] = ..., proto_file_name: _Optional[str] = ..., proto_message_name: _Optional[str] = ..., consumer_group: _Optional[str] = ...) -> None: ...

class InputConfig(_message.Message):
    __slots__ = ("key_field", "label_field", "numerical", "categorical", "time_tick", "textual", "imaginal")
    KEY_FIELD_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_FIELD_NUMBER: _ClassVar[int]
    NUMERICAL_FIELD_NUMBER: _ClassVar[int]
    CATEGORICAL_FIELD_NUMBER: _ClassVar[int]
    TIME_TICK_FIELD_NUMBER: _ClassVar[int]
    TEXTUAL_FIELD_NUMBER: _ClassVar[int]
    IMAGINAL_FIELD_NUMBER: _ClassVar[int]
    key_field: str
    label_field: str
    numerical: _containers.RepeatedScalarFieldContainer[str]
    categorical: _containers.RepeatedScalarFieldContainer[str]
    time_tick: str
    textual: _containers.RepeatedScalarFieldContainer[str]
    imaginal: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, key_field: _Optional[str] = ..., label_field: _Optional[str] = ..., numerical: _Optional[_Iterable[str]] = ..., categorical: _Optional[_Iterable[str]] = ..., time_tick: _Optional[str] = ..., textual: _Optional[_Iterable[str]] = ..., imaginal: _Optional[_Iterable[str]] = ...) -> None: ...

class LearnerConfig(_message.Message):
    __slots__ = ("predict_workers", "update_batch_size", "synchronization_method")
    PREDICT_WORKERS_FIELD_NUMBER: _ClassVar[int]
    UPDATE_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    SYNCHRONIZATION_METHOD_FIELD_NUMBER: _ClassVar[int]
    predict_workers: int
    update_batch_size: int
    synchronization_method: str
    def __init__(self, predict_workers: _Optional[int] = ..., update_batch_size: _Optional[int] = ..., synchronization_method: _Optional[str] = ...) -> None: ...

class TurboMLConfig(_message.Message):
    __slots__ = ("brokers", "feat_consumer", "output_producer", "label_consumer", "input_config", "model_configs", "initial_model_id", "api_port", "arrow_port", "feature_retrieval", "combined_producer", "combined_consumer", "learner_config", "fully_qualified_model_name")
    BROKERS_FIELD_NUMBER: _ClassVar[int]
    FEAT_CONSUMER_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PRODUCER_FIELD_NUMBER: _ClassVar[int]
    LABEL_CONSUMER_FIELD_NUMBER: _ClassVar[int]
    INPUT_CONFIG_FIELD_NUMBER: _ClassVar[int]
    MODEL_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    INITIAL_MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    API_PORT_FIELD_NUMBER: _ClassVar[int]
    ARROW_PORT_FIELD_NUMBER: _ClassVar[int]
    FEATURE_RETRIEVAL_FIELD_NUMBER: _ClassVar[int]
    COMBINED_PRODUCER_FIELD_NUMBER: _ClassVar[int]
    COMBINED_CONSUMER_FIELD_NUMBER: _ClassVar[int]
    LEARNER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    FULLY_QUALIFIED_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    brokers: str
    feat_consumer: KafkaConsumerConfig
    output_producer: KafkaProducerConfig
    label_consumer: KafkaConsumerConfig
    input_config: InputConfig
    model_configs: _containers.RepeatedCompositeFieldContainer[ModelConfig]
    initial_model_id: str
    api_port: int
    arrow_port: int
    feature_retrieval: FeatureRetrievalConfig
    combined_producer: KafkaProducerConfig
    combined_consumer: KafkaConsumerConfig
    learner_config: LearnerConfig
    fully_qualified_model_name: str
    def __init__(self, brokers: _Optional[str] = ..., feat_consumer: _Optional[_Union[KafkaConsumerConfig, _Mapping]] = ..., output_producer: _Optional[_Union[KafkaProducerConfig, _Mapping]] = ..., label_consumer: _Optional[_Union[KafkaConsumerConfig, _Mapping]] = ..., input_config: _Optional[_Union[InputConfig, _Mapping]] = ..., model_configs: _Optional[_Iterable[_Union[ModelConfig, _Mapping]]] = ..., initial_model_id: _Optional[str] = ..., api_port: _Optional[int] = ..., arrow_port: _Optional[int] = ..., feature_retrieval: _Optional[_Union[FeatureRetrievalConfig, _Mapping]] = ..., combined_producer: _Optional[_Union[KafkaProducerConfig, _Mapping]] = ..., combined_consumer: _Optional[_Union[KafkaConsumerConfig, _Mapping]] = ..., learner_config: _Optional[_Union[LearnerConfig, _Mapping]] = ..., fully_qualified_model_name: _Optional[str] = ...) -> None: ...

class TrainJobConfig(_message.Message):
    __slots__ = ("initial_model_key", "input_config", "model_configs", "model_name", "version_name")
    INITIAL_MODEL_KEY_FIELD_NUMBER: _ClassVar[int]
    INPUT_CONFIG_FIELD_NUMBER: _ClassVar[int]
    MODEL_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_NAME_FIELD_NUMBER: _ClassVar[int]
    initial_model_key: str
    input_config: InputConfig
    model_configs: _containers.RepeatedCompositeFieldContainer[ModelConfig]
    model_name: str
    version_name: str
    def __init__(self, initial_model_key: _Optional[str] = ..., input_config: _Optional[_Union[InputConfig, _Mapping]] = ..., model_configs: _Optional[_Iterable[_Union[ModelConfig, _Mapping]]] = ..., model_name: _Optional[str] = ..., version_name: _Optional[str] = ...) -> None: ...

class ModelConfigList(_message.Message):
    __slots__ = ("model_configs",)
    MODEL_CONFIGS_FIELD_NUMBER: _ClassVar[int]
    model_configs: _containers.RepeatedCompositeFieldContainer[ModelConfig]
    def __init__(self, model_configs: _Optional[_Iterable[_Union[ModelConfig, _Mapping]]] = ...) -> None: ...

```

# flink_pb2.py
-## Location -> root_directory.common.protos
```python
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: flink.proto
# Protobuf Python Version: 4.25.5
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import sources_pb2 as sources__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0b\x66link.proto\x12\x05\x66link\x1a\rsources.proto\"\'\n\x17\x46linkDeploymentResource\x12\x0c\n\x04name\x18\x01 \x02(\t\"\xba\x0f\n\x11\x44\x65ploymentRequest\x12\x17\n\x0f\x64\x65ployment_name\x18\x01 \x02(\t\x12@\n\x0esql_deployment\x18\x02 \x01(\x0b\x32&.flink.DeploymentRequest.SqlDeploymentH\x00\x12J\n\x13\x61rtifact_deployment\x18\x03 \x01(\x0b\x32+.flink.DeploymentRequest.ArtifactDeploymentH\x00\x12G\n\x10\x66link_properties\x18\x05 \x03(\x0b\x32-.flink.DeploymentRequest.FlinkPropertiesEntry\x12\x45\n\x12job_manager_config\x18\x06 \x01(\x0b\x32).flink.DeploymentRequest.JobManagerConfig\x12G\n\x13task_manager_config\x18\x07 \x01(\x0b\x32*.flink.DeploymentRequest.TaskManagerConfig\x12\x37\n\x08\x65nv_vars\x18\x08 \x03(\x0b\x32%.flink.DeploymentRequest.EnvVarsEntry\x12\x16\n\x0e\x66rom_savepoint\x18\t \x01(\t\x12\x1a\n\x12recreate_on_update\x18\n \x01(\x08\x12\x35\n\tsavepoint\x18\x0b \x01(\x0b\x32\".flink.DeploymentRequest.Savepoint\x12 \n\x18\x61llow_non_restored_state\x18\x0c \x01(\x08\x12 \n\x18take_savepoint_on_update\x18\r \x01(\x08\x12\x13\n\x0bparallelism\x18\x0e \x01(\r\x12\x16\n\x0erestart_policy\x18\x0f \x01(\t\x12\x16\n\x0e\x63leanup_policy\x18\x10 \x01(\t\x12\x1c\n\x14savepoint_generation\x18\x11 \x01(\x03\x12\x18\n\x10\x63\x61ncel_requested\x18\x12 \x01(\x08\x12\x17\n\x0flocal_time_zone\x18\x13 \x01(\t\x1a\xdb\x01\n\rSqlDeployment\x12\r\n\x05query\x18\x01 \x02(\t\x12)\n\x0c\x64\x61ta_sources\x18\x02 \x03(\x0b\x32\x13.sources.DataSource\x12\x38\n\x04udfs\x18\x03 \x03(\x0b\x32*.flink.DeploymentRequest.SqlDeployment.Udf\x12\x12\n\nsink_topic\x18\x04 \x02(\t\x12\x1f\n\x17sink_topic_message_name\x18\x05 \x02(\t\x1a!\n\x03Udf\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x0c\n\x04\x63ode\x18\x02 \x02(\t\x1a\xb4\x03\n\x12\x41rtifactDeployment\x12G\n\x08java_job\x18\x01 \x01(\x0b\x32\x33.flink.DeploymentRequest.ArtifactDeployment.JavaJobH\x00\x12K\n\npython_job\x18\x02 \x01(\x0b\x32\x35.flink.DeploymentRequest.ArtifactDeployment.PythonJobH\x00\x12\x17\n\x0f\x66iles_base_path\x18\x03 \x02(\t\x12Z\n\x10\x66link_properties\x18\x04 \x03(\x0b\x32@.flink.DeploymentRequest.ArtifactDeployment.FlinkPropertiesEntry\x1a/\n\x07JavaJob\x12\x10\n\x08jar_name\x18\x01 \x02(\t\x12\x12\n\nclass_name\x18\x02 \x02(\t\x1a\x1e\n\tPythonJob\x12\x11\n\tfile_name\x18\x01 \x02(\t\x1a\x36\n\x14\x46linkPropertiesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x42\n\n\x08job_type\x1a\x42\n\x17\x43ontainerResourceLimits\x12\x11\n\tcpu_limit\x18\x01 \x02(\t\x12\x14\n\x0cmemory_limit\x18\x02 \x02(\t\x1a\x83\x01\n\x10JobManagerConfig\x12V\n\x1cjob_manager_resources_limits\x18\x01 \x01(\x0b\x32\x30.flink.DeploymentRequest.ContainerResourceLimits\x12\x17\n\x0fnum_of_replicas\x18\x02 \x01(\x05\x1a\x85\x01\n\x11TaskManagerConfig\x12W\n\x1dtask_manager_resources_limits\x18\x01 \x01(\x0b\x32\x30.flink.DeploymentRequest.ContainerResourceLimits\x12\x17\n\x0fnum_of_replicas\x18\x02 \x01(\x05\x1a\x43\n\tSavepoint\x12\x1e\n\x16\x61uto_savepoint_seconds\x18\x01 \x01(\x05\x12\x16\n\x0esavepoints_dir\x18\x02 \x01(\t\x1a\x36\n\x14\x46linkPropertiesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a.\n\x0c\x45nvVarsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x42\x13\n\x11\x64\x65ployment_config\"\n\n\x08Response2\x96\x01\n\x05\x46link\x12\x46\n\x17SubmitDeploymentRequest\x12\x18.flink.DeploymentRequest\x1a\x0f.flink.Response\"\x00\x12\x45\n\x10\x44\x65leteDeployment\x12\x1e.flink.FlinkDeploymentResource\x1a\x0f.flink.Response\"\x00\x42\"\n\x11\x63om.turboml.flinkB\x0b\x46linkServerP\x01')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'flink_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\021com.turboml.flinkB\013FlinkServerP\001'
  _globals['_DEPLOYMENTREQUEST_ARTIFACTDEPLOYMENT_FLINKPROPERTIESENTRY']._options = None
  _globals['_DEPLOYMENTREQUEST_ARTIFACTDEPLOYMENT_FLINKPROPERTIESENTRY']._serialized_options = b'8\001'
  _globals['_DEPLOYMENTREQUEST_FLINKPROPERTIESENTRY']._options = None
  _globals['_DEPLOYMENTREQUEST_FLINKPROPERTIESENTRY']._serialized_options = b'8\001'
  _globals['_DEPLOYMENTREQUEST_ENVVARSENTRY']._options = None
  _globals['_DEPLOYMENTREQUEST_ENVVARSENTRY']._serialized_options = b'8\001'
  _globals['_FLINKDEPLOYMENTRESOURCE']._serialized_start=37
  _globals['_FLINKDEPLOYMENTRESOURCE']._serialized_end=76
  _globals['_DEPLOYMENTREQUEST']._serialized_start=79
  _globals['_DEPLOYMENTREQUEST']._serialized_end=2057
  _globals['_DEPLOYMENTREQUEST_SQLDEPLOYMENT']._serialized_start=867
  _globals['_DEPLOYMENTREQUEST_SQLDEPLOYMENT']._serialized_end=1086
  _globals['_DEPLOYMENTREQUEST_SQLDEPLOYMENT_UDF']._serialized_start=1053
  _globals['_DEPLOYMENTREQUEST_SQLDEPLOYMENT_UDF']._serialized_end=1086
  _globals['_DEPLOYMENTREQUEST_ARTIFACTDEPLOYMENT']._serialized_start=1089
  _globals['_DEPLOYMENTREQUEST_ARTIFACTDEPLOYMENT']._serialized_end=1525
  _globals['_DEPLOYMENTREQUEST_ARTIFACTDEPLOYMENT_JAVAJOB']._serialized_start=1378
  _globals['_DEPLOYMENTREQUEST_ARTIFACTDEPLOYMENT_JAVAJOB']._serialized_end=1425
  _globals['_DEPLOYMENTREQUEST_ARTIFACTDEPLOYMENT_PYTHONJOB']._serialized_start=1427
  _globals['_DEPLOYMENTREQUEST_ARTIFACTDEPLOYMENT_PYTHONJOB']._serialized_end=1457
  _globals['_DEPLOYMENTREQUEST_ARTIFACTDEPLOYMENT_FLINKPROPERTIESENTRY']._serialized_start=1459
  _globals['_DEPLOYMENTREQUEST_ARTIFACTDEPLOYMENT_FLINKPROPERTIESENTRY']._serialized_end=1513
  _globals['_DEPLOYMENTREQUEST_CONTAINERRESOURCELIMITS']._serialized_start=1527
  _globals['_DEPLOYMENTREQUEST_CONTAINERRESOURCELIMITS']._serialized_end=1593
  _globals['_DEPLOYMENTREQUEST_JOBMANAGERCONFIG']._serialized_start=1596
  _globals['_DEPLOYMENTREQUEST_JOBMANAGERCONFIG']._serialized_end=1727
  _globals['_DEPLOYMENTREQUEST_TASKMANAGERCONFIG']._serialized_start=1730
  _globals['_DEPLOYMENTREQUEST_TASKMANAGERCONFIG']._serialized_end=1863
  _globals['_DEPLOYMENTREQUEST_SAVEPOINT']._serialized_start=1865
  _globals['_DEPLOYMENTREQUEST_SAVEPOINT']._serialized_end=1932
  _globals['_DEPLOYMENTREQUEST_FLINKPROPERTIESENTRY']._serialized_start=1459
  _globals['_DEPLOYMENTREQUEST_FLINKPROPERTIESENTRY']._serialized_end=1513
  _globals['_DEPLOYMENTREQUEST_ENVVARSENTRY']._serialized_start=1990
  _globals['_DEPLOYMENTREQUEST_ENVVARSENTRY']._serialized_end=2036
  _globals['_RESPONSE']._serialized_start=2059
  _globals['_RESPONSE']._serialized_end=2069
  _globals['_FLINK']._serialized_start=2072
  _globals['_FLINK']._serialized_end=2222
# @@protoc_insertion_point(module_scope)

```

#### flink_pb2.pyi
```python
import sources_pb2 as _sources_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FlinkDeploymentResource(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class DeploymentRequest(_message.Message):
    __slots__ = ("deployment_name", "sql_deployment", "artifact_deployment", "flink_properties", "job_manager_config", "task_manager_config", "env_vars", "from_savepoint", "recreate_on_update", "savepoint", "allow_non_restored_state", "take_savepoint_on_update", "parallelism", "restart_policy", "cleanup_policy", "savepoint_generation", "cancel_requested", "local_time_zone")
    class SqlDeployment(_message.Message):
        __slots__ = ("query", "data_sources", "udfs", "sink_topic", "sink_topic_message_name")
        class Udf(_message.Message):
            __slots__ = ("name", "code")
            NAME_FIELD_NUMBER: _ClassVar[int]
            CODE_FIELD_NUMBER: _ClassVar[int]
            name: str
            code: str
            def __init__(self, name: _Optional[str] = ..., code: _Optional[str] = ...) -> None: ...
        QUERY_FIELD_NUMBER: _ClassVar[int]
        DATA_SOURCES_FIELD_NUMBER: _ClassVar[int]
        UDFS_FIELD_NUMBER: _ClassVar[int]
        SINK_TOPIC_FIELD_NUMBER: _ClassVar[int]
        SINK_TOPIC_MESSAGE_NAME_FIELD_NUMBER: _ClassVar[int]
        query: str
        data_sources: _containers.RepeatedCompositeFieldContainer[_sources_pb2.DataSource]
        udfs: _containers.RepeatedCompositeFieldContainer[DeploymentRequest.SqlDeployment.Udf]
        sink_topic: str
        sink_topic_message_name: str
        def __init__(self, query: _Optional[str] = ..., data_sources: _Optional[_Iterable[_Union[_sources_pb2.DataSource, _Mapping]]] = ..., udfs: _Optional[_Iterable[_Union[DeploymentRequest.SqlDeployment.Udf, _Mapping]]] = ..., sink_topic: _Optional[str] = ..., sink_topic_message_name: _Optional[str] = ...) -> None: ...
    class ArtifactDeployment(_message.Message):
        __slots__ = ("java_job", "python_job", "files_base_path", "flink_properties")
        class JavaJob(_message.Message):
            __slots__ = ("jar_name", "class_name")
            JAR_NAME_FIELD_NUMBER: _ClassVar[int]
            CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
            jar_name: str
            class_name: str
            def __init__(self, jar_name: _Optional[str] = ..., class_name: _Optional[str] = ...) -> None: ...
        class PythonJob(_message.Message):
            __slots__ = ("file_name",)
            FILE_NAME_FIELD_NUMBER: _ClassVar[int]
            file_name: str
            def __init__(self, file_name: _Optional[str] = ...) -> None: ...
        class FlinkPropertiesEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
        JAVA_JOB_FIELD_NUMBER: _ClassVar[int]
        PYTHON_JOB_FIELD_NUMBER: _ClassVar[int]
        FILES_BASE_PATH_FIELD_NUMBER: _ClassVar[int]
        FLINK_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
        java_job: DeploymentRequest.ArtifactDeployment.JavaJob
        python_job: DeploymentRequest.ArtifactDeployment.PythonJob
        files_base_path: str
        flink_properties: _containers.ScalarMap[str, str]
        def __init__(self, java_job: _Optional[_Union[DeploymentRequest.ArtifactDeployment.JavaJob, _Mapping]] = ..., python_job: _Optional[_Union[DeploymentRequest.ArtifactDeployment.PythonJob, _Mapping]] = ..., files_base_path: _Optional[str] = ..., flink_properties: _Optional[_Mapping[str, str]] = ...) -> None: ...
    class ContainerResourceLimits(_message.Message):
        __slots__ = ("cpu_limit", "memory_limit")
        CPU_LIMIT_FIELD_NUMBER: _ClassVar[int]
        MEMORY_LIMIT_FIELD_NUMBER: _ClassVar[int]
        cpu_limit: str
        memory_limit: str
        def __init__(self, cpu_limit: _Optional[str] = ..., memory_limit: _Optional[str] = ...) -> None: ...
    class JobManagerConfig(_message.Message):
        __slots__ = ("job_manager_resources_limits", "num_of_replicas")
        JOB_MANAGER_RESOURCES_LIMITS_FIELD_NUMBER: _ClassVar[int]
        NUM_OF_REPLICAS_FIELD_NUMBER: _ClassVar[int]
        job_manager_resources_limits: DeploymentRequest.ContainerResourceLimits
        num_of_replicas: int
        def __init__(self, job_manager_resources_limits: _Optional[_Union[DeploymentRequest.ContainerResourceLimits, _Mapping]] = ..., num_of_replicas: _Optional[int] = ...) -> None: ...
    class TaskManagerConfig(_message.Message):
        __slots__ = ("task_manager_resources_limits", "num_of_replicas")
        TASK_MANAGER_RESOURCES_LIMITS_FIELD_NUMBER: _ClassVar[int]
        NUM_OF_REPLICAS_FIELD_NUMBER: _ClassVar[int]
        task_manager_resources_limits: DeploymentRequest.ContainerResourceLimits
        num_of_replicas: int
        def __init__(self, task_manager_resources_limits: _Optional[_Union[DeploymentRequest.ContainerResourceLimits, _Mapping]] = ..., num_of_replicas: _Optional[int] = ...) -> None: ...
    class Savepoint(_message.Message):
        __slots__ = ("auto_savepoint_seconds", "savepoints_dir")
        AUTO_SAVEPOINT_SECONDS_FIELD_NUMBER: _ClassVar[int]
        SAVEPOINTS_DIR_FIELD_NUMBER: _ClassVar[int]
        auto_savepoint_seconds: int
        savepoints_dir: str
        def __init__(self, auto_savepoint_seconds: _Optional[int] = ..., savepoints_dir: _Optional[str] = ...) -> None: ...
    class FlinkPropertiesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class EnvVarsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DEPLOYMENT_NAME_FIELD_NUMBER: _ClassVar[int]
    SQL_DEPLOYMENT_FIELD_NUMBER: _ClassVar[int]
    ARTIFACT_DEPLOYMENT_FIELD_NUMBER: _ClassVar[int]
    FLINK_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    JOB_MANAGER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    TASK_MANAGER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    ENV_VARS_FIELD_NUMBER: _ClassVar[int]
    FROM_SAVEPOINT_FIELD_NUMBER: _ClassVar[int]
    RECREATE_ON_UPDATE_FIELD_NUMBER: _ClassVar[int]
    SAVEPOINT_FIELD_NUMBER: _ClassVar[int]
    ALLOW_NON_RESTORED_STATE_FIELD_NUMBER: _ClassVar[int]
    TAKE_SAVEPOINT_ON_UPDATE_FIELD_NUMBER: _ClassVar[int]
    PARALLELISM_FIELD_NUMBER: _ClassVar[int]
    RESTART_POLICY_FIELD_NUMBER: _ClassVar[int]
    CLEANUP_POLICY_FIELD_NUMBER: _ClassVar[int]
    SAVEPOINT_GENERATION_FIELD_NUMBER: _ClassVar[int]
    CANCEL_REQUESTED_FIELD_NUMBER: _ClassVar[int]
    LOCAL_TIME_ZONE_FIELD_NUMBER: _ClassVar[int]
    deployment_name: str
    sql_deployment: DeploymentRequest.SqlDeployment
    artifact_deployment: DeploymentRequest.ArtifactDeployment
    flink_properties: _containers.ScalarMap[str, str]
    job_manager_config: DeploymentRequest.JobManagerConfig
    task_manager_config: DeploymentRequest.TaskManagerConfig
    env_vars: _containers.ScalarMap[str, str]
    from_savepoint: str
    recreate_on_update: bool
    savepoint: DeploymentRequest.Savepoint
    allow_non_restored_state: bool
    take_savepoint_on_update: bool
    parallelism: int
    restart_policy: str
    cleanup_policy: str
    savepoint_generation: int
    cancel_requested: bool
    local_time_zone: str
    def __init__(self, deployment_name: _Optional[str] = ..., sql_deployment: _Optional[_Union[DeploymentRequest.SqlDeployment, _Mapping]] = ..., artifact_deployment: _Optional[_Union[DeploymentRequest.ArtifactDeployment, _Mapping]] = ..., flink_properties: _Optional[_Mapping[str, str]] = ..., job_manager_config: _Optional[_Union[DeploymentRequest.JobManagerConfig, _Mapping]] = ..., task_manager_config: _Optional[_Union[DeploymentRequest.TaskManagerConfig, _Mapping]] = ..., env_vars: _Optional[_Mapping[str, str]] = ..., from_savepoint: _Optional[str] = ..., recreate_on_update: bool = ..., savepoint: _Optional[_Union[DeploymentRequest.Savepoint, _Mapping]] = ..., allow_non_restored_state: bool = ..., take_savepoint_on_update: bool = ..., parallelism: _Optional[int] = ..., restart_policy: _Optional[str] = ..., cleanup_policy: _Optional[str] = ..., savepoint_generation: _Optional[int] = ..., cancel_requested: bool = ..., local_time_zone: _Optional[str] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

```

# flink_pb2_grpc.py
-## Location -> root_directory.common.protos
```python
# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import flink_pb2 as flink__pb2

GRPC_GENERATED_VERSION = '1.68.1'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in flink_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class FlinkStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SubmitDeploymentRequest = channel.unary_unary(
                '/flink.Flink/SubmitDeploymentRequest',
                request_serializer=flink__pb2.DeploymentRequest.SerializeToString,
                response_deserializer=flink__pb2.Response.FromString,
                _registered_method=True)
        self.DeleteDeployment = channel.unary_unary(
                '/flink.Flink/DeleteDeployment',
                request_serializer=flink__pb2.FlinkDeploymentResource.SerializeToString,
                response_deserializer=flink__pb2.Response.FromString,
                _registered_method=True)


class FlinkServicer(object):
    """Missing associated documentation comment in .proto file."""

    def SubmitDeploymentRequest(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteDeployment(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FlinkServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'SubmitDeploymentRequest': grpc.unary_unary_rpc_method_handler(
                    servicer.SubmitDeploymentRequest,
                    request_deserializer=flink__pb2.DeploymentRequest.FromString,
                    response_serializer=flink__pb2.Response.SerializeToString,
            ),
            'DeleteDeployment': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteDeployment,
                    request_deserializer=flink__pb2.FlinkDeploymentResource.FromString,
                    response_serializer=flink__pb2.Response.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'flink.Flink', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('flink.Flink', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class Flink(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def SubmitDeploymentRequest(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/flink.Flink/SubmitDeploymentRequest',
            flink__pb2.DeploymentRequest.SerializeToString,
            flink__pb2.Response.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def DeleteDeployment(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/flink.Flink/DeleteDeployment',
            flink__pb2.FlinkDeploymentResource.SerializeToString,
            flink__pb2.Response.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

```

# input_pb2.py
-## Location -> root_directory.common.protos
```python
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: input.proto
# Protobuf Python Version: 4.25.5
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0binput.proto\x12\x05input\"t\n\x05Input\x12\x0f\n\x07numeric\x18\x01 \x03(\x02\x12\r\n\x05\x63\x61teg\x18\x02 \x03(\x05\x12\x0c\n\x04text\x18\x03 \x03(\t\x12\x0e\n\x06images\x18\x04 \x03(\x0c\x12\x11\n\ttime_tick\x18\x05 \x01(\x05\x12\r\n\x05label\x18\x06 \x01(\x02\x12\x0b\n\x03key\x18\x07 \x01(\t')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'input_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_INPUT']._serialized_start=22
  _globals['_INPUT']._serialized_end=138
# @@protoc_insertion_point(module_scope)

```

#### input_pb2.pyi
```python
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Input(_message.Message):
    __slots__ = ("numeric", "categ", "text", "images", "time_tick", "label", "key")
    NUMERIC_FIELD_NUMBER: _ClassVar[int]
    CATEG_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    IMAGES_FIELD_NUMBER: _ClassVar[int]
    TIME_TICK_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    numeric: _containers.RepeatedScalarFieldContainer[float]
    categ: _containers.RepeatedScalarFieldContainer[int]
    text: _containers.RepeatedScalarFieldContainer[str]
    images: _containers.RepeatedScalarFieldContainer[bytes]
    time_tick: int
    label: float
    key: str
    def __init__(self, numeric: _Optional[_Iterable[float]] = ..., categ: _Optional[_Iterable[int]] = ..., text: _Optional[_Iterable[str]] = ..., images: _Optional[_Iterable[bytes]] = ..., time_tick: _Optional[int] = ..., label: _Optional[float] = ..., key: _Optional[str] = ...) -> None: ...

```

# metrics_pb2.py
-## Location -> root_directory.common.protos
```python
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: metrics.proto
# Protobuf Python Version: 4.25.5
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rmetrics.proto\x12\x07metrics\"(\n\x07Metrics\x12\r\n\x05index\x18\x01 \x01(\x05\x12\x0e\n\x06metric\x18\x02 \x01(\x02')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'metrics_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_METRICS']._serialized_start=26
  _globals['_METRICS']._serialized_end=66
# @@protoc_insertion_point(module_scope)

```

#### metrics_pb2.pyi
```python
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Metrics(_message.Message):
    __slots__ = ("index", "metric")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    METRIC_FIELD_NUMBER: _ClassVar[int]
    index: int
    metric: float
    def __init__(self, index: _Optional[int] = ..., metric: _Optional[float] = ...) -> None: ...

```

# ml_service_pb2.py
-## Location -> root_directory.common.protos
```python
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ml_service.proto
# Protobuf Python Version: 4.25.5
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import input_pb2 as input__pb2
import output_pb2 as output__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10ml_service.proto\x12\nml_service\x1a\x0binput.proto\x1a\x0coutput.proto\"\x07\n\x05\x45mpty2^\n\tMLService\x12(\n\x05Learn\x12\x0c.input.Input\x1a\x11.ml_service.Empty\x12\'\n\x07Predict\x12\x0c.input.Input\x1a\x0e.output.Output')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ml_service_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_EMPTY']._serialized_start=59
  _globals['_EMPTY']._serialized_end=66
  _globals['_MLSERVICE']._serialized_start=68
  _globals['_MLSERVICE']._serialized_end=162
# @@protoc_insertion_point(module_scope)

```

#### ml_service_pb2.pyi
```python
import input_pb2 as _input_pb2
import output_pb2 as _output_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

```

# ml_service_pb2_grpc.py
-## Location -> root_directory.common.protos
```python
# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import input_pb2 as input__pb2
import ml_service_pb2 as ml__service__pb2
import output_pb2 as output__pb2

GRPC_GENERATED_VERSION = '1.68.1'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in ml_service_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class MLServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Learn = channel.unary_unary(
                '/ml_service.MLService/Learn',
                request_serializer=input__pb2.Input.SerializeToString,
                response_deserializer=ml__service__pb2.Empty.FromString,
                _registered_method=True)
        self.Predict = channel.unary_unary(
                '/ml_service.MLService/Predict',
                request_serializer=input__pb2.Input.SerializeToString,
                response_deserializer=output__pb2.Output.FromString,
                _registered_method=True)


class MLServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Learn(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Predict(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MLServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Learn': grpc.unary_unary_rpc_method_handler(
                    servicer.Learn,
                    request_deserializer=input__pb2.Input.FromString,
                    response_serializer=ml__service__pb2.Empty.SerializeToString,
            ),
            'Predict': grpc.unary_unary_rpc_method_handler(
                    servicer.Predict,
                    request_deserializer=input__pb2.Input.FromString,
                    response_serializer=output__pb2.Output.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ml_service.MLService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('ml_service.MLService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class MLService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Learn(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/ml_service.MLService/Learn',
            input__pb2.Input.SerializeToString,
            ml__service__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Predict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/ml_service.MLService/Predict',
            input__pb2.Input.SerializeToString,
            output__pb2.Output.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

```

# output_pb2.py
-## Location -> root_directory.common.protos
```python
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: output.proto
# Protobuf Python Version: 4.25.5
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0coutput.proto\x12\x06output\"\x9a\x01\n\x06Output\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x02\x12\x15\n\rfeature_score\x18\x03 \x03(\x02\x12\x1b\n\x13\x63lass_probabilities\x18\x04 \x03(\x02\x12\x17\n\x0fpredicted_class\x18\x05 \x01(\x05\x12\x12\n\nembeddings\x18\x06 \x03(\x02\x12\x13\n\x0btext_output\x18\x07 \x01(\t')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'output_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_OUTPUT']._serialized_start=25
  _globals['_OUTPUT']._serialized_end=179
# @@protoc_insertion_point(module_scope)

```

#### output_pb2.pyi
```python
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Output(_message.Message):
    __slots__ = ("key", "score", "feature_score", "class_probabilities", "predicted_class", "embeddings", "text_output")
    KEY_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    FEATURE_SCORE_FIELD_NUMBER: _ClassVar[int]
    CLASS_PROBABILITIES_FIELD_NUMBER: _ClassVar[int]
    PREDICTED_CLASS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    TEXT_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    key: str
    score: float
    feature_score: _containers.RepeatedScalarFieldContainer[float]
    class_probabilities: _containers.RepeatedScalarFieldContainer[float]
    predicted_class: int
    embeddings: _containers.RepeatedScalarFieldContainer[float]
    text_output: str
    def __init__(self, key: _Optional[str] = ..., score: _Optional[float] = ..., feature_score: _Optional[_Iterable[float]] = ..., class_probabilities: _Optional[_Iterable[float]] = ..., predicted_class: _Optional[int] = ..., embeddings: _Optional[_Iterable[float]] = ..., text_output: _Optional[str] = ...) -> None: ...

```

# sources_pb2.py
-## Location -> root_directory.common.protos
```python
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: sources.proto
# Protobuf Python Version: 4.25.5
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rsources.proto\x12\x07sources\"P\n\x0bKafkaSource\x12\r\n\x05topic\x18\x01 \x01(\t\x12\x1a\n\x12proto_message_name\x18\x02 \x01(\t\x12\x16\n\x0eschema_version\x18\x03 \x01(\x05\"\"\n\x12\x46\x65\x61tureGroupSource\x12\x0c\n\x04name\x18\x01 \x01(\t\"\xb2\x01\n\x08S3Config\x12\x0e\n\x06\x62ucket\x18\x01 \x01(\t\x12\x1a\n\raccess_key_id\x18\x02 \x01(\tH\x00\x88\x01\x01\x12\x1e\n\x11secret_access_key\x18\x03 \x01(\tH\x01\x88\x01\x01\x12\x0e\n\x06region\x18\x04 \x01(\t\x12\x15\n\x08\x65ndpoint\x18\x05 \x01(\tH\x02\x88\x01\x01\x42\x10\n\x0e_access_key_idB\x14\n\x12_secret_access_keyB\x0b\n\t_endpoint\"\xbd\x01\n\nFileSource\x12\x0c\n\x04path\x18\x01 \x01(\t\x12*\n\x06\x66ormat\x18\x02 \x01(\x0e\x32\x1a.sources.FileSource.Format\x12&\n\ts3_config\x18\x03 \x01(\x0b\x32\x11.sources.S3ConfigH\x00\";\n\x06\x46ormat\x12\x07\n\x03\x43SV\x10\x00\x12\x08\n\x04JSON\x10\x01\x12\x08\n\x04\x41VRO\x10\x02\x12\x0b\n\x07PARQUET\x10\x03\x12\x07\n\x03ORC\x10\x04\x42\x10\n\x0estorage_config\"\xa8\x01\n\x0ePostgresSource\x12\x0c\n\x04host\x18\x01 \x01(\t\x12\x0c\n\x04port\x18\x02 \x01(\x05\x12\x10\n\x08username\x18\x03 \x01(\t\x12\x10\n\x08password\x18\x04 \x01(\t\x12\r\n\x05table\x18\x05 \x01(\t\x12\x10\n\x08\x64\x61tabase\x18\x06 \x01(\t\x12\x1b\n\x13incrementing_column\x18\x07 \x01(\t\x12\x18\n\x10timestamp_column\x18\x08 \x01(\t\"\xf1\x01\n\x15TimestampFormatConfig\x12>\n\x0b\x66ormat_type\x18\x01 \x01(\x0e\x32).sources.TimestampFormatConfig.FormatType\x12\x1a\n\rformat_string\x18\x02 \x01(\tH\x00\x88\x01\x01\x12\x16\n\ttime_zone\x18\x03 \x01(\tH\x01\x88\x01\x01\"D\n\nFormatType\x12\x0f\n\x0b\x45pochMillis\x10\x00\x12\x10\n\x0c\x45pochSeconds\x10\x01\x12\x13\n\x0fStringTimestamp\x10\x02\x42\x10\n\x0e_format_stringB\x0c\n\n_time_zone\"\x8e\x01\n\tWatermark\x12\x10\n\x08time_col\x18\x01 \x01(\t\x12\x1d\n\x15\x61llowed_delay_seconds\x18\x02 \x01(\x03\x12<\n\x0ftime_col_config\x18\x03 \x01(\x0b\x32\x1e.sources.TimestampFormatConfigH\x00\x88\x01\x01\x42\x12\n\x10_time_col_config\"\xc7\x03\n\nDataSource\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\nkey_fields\x18\x02 \x03(\t\x12\x18\n\x0b\x65nvironment\x18\x04 \x01(\tH\x01\x88\x01\x01\x12\x1b\n\x0e\x65ncoded_schema\x18\x05 \x01(\tH\x02\x88\x01\x01\x12\x30\n\rdelivery_mode\x18\x06 \x01(\x0e\x32\x19.sources.DataDeliveryMode\x12*\n\twatermark\x18\x07 \x01(\x0b\x32\x12.sources.WatermarkH\x03\x88\x01\x01\x12*\n\x0b\x66ile_source\x18\x08 \x01(\x0b\x32\x13.sources.FileSourceH\x00\x12\x32\n\x0fpostgres_source\x18\t \x01(\x0b\x32\x17.sources.PostgresSourceH\x00\x12,\n\x0ckafka_source\x18\n \x01(\x0b\x32\x14.sources.KafkaSourceH\x00\x12;\n\x14\x66\x65\x61ture_group_source\x18\x0b \x01(\x0b\x32\x1b.sources.FeatureGroupSourceH\x00\x42\x06\n\x04typeB\x0e\n\x0c_environmentB\x11\n\x0f_encoded_schemaB\x0c\n\n_watermark*+\n\x10\x44\x61taDeliveryMode\x12\n\n\x06STATIC\x10\x00\x12\x0b\n\x07\x44YNAMIC\x10\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'sources_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_DATADELIVERYMODE']._serialized_start=1535
  _globals['_DATADELIVERYMODE']._serialized_end=1578
  _globals['_KAFKASOURCE']._serialized_start=26
  _globals['_KAFKASOURCE']._serialized_end=106
  _globals['_FEATUREGROUPSOURCE']._serialized_start=108
  _globals['_FEATUREGROUPSOURCE']._serialized_end=142
  _globals['_S3CONFIG']._serialized_start=145
  _globals['_S3CONFIG']._serialized_end=323
  _globals['_FILESOURCE']._serialized_start=326
  _globals['_FILESOURCE']._serialized_end=515
  _globals['_FILESOURCE_FORMAT']._serialized_start=438
  _globals['_FILESOURCE_FORMAT']._serialized_end=497
  _globals['_POSTGRESSOURCE']._serialized_start=518
  _globals['_POSTGRESSOURCE']._serialized_end=686
  _globals['_TIMESTAMPFORMATCONFIG']._serialized_start=689
  _globals['_TIMESTAMPFORMATCONFIG']._serialized_end=930
  _globals['_TIMESTAMPFORMATCONFIG_FORMATTYPE']._serialized_start=830
  _globals['_TIMESTAMPFORMATCONFIG_FORMATTYPE']._serialized_end=898
  _globals['_WATERMARK']._serialized_start=933
  _globals['_WATERMARK']._serialized_end=1075
  _globals['_DATASOURCE']._serialized_start=1078
  _globals['_DATASOURCE']._serialized_end=1533
# @@protoc_insertion_point(module_scope)

```

#### sources_pb2.pyi
```python
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataDeliveryMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STATIC: _ClassVar[DataDeliveryMode]
    DYNAMIC: _ClassVar[DataDeliveryMode]
STATIC: DataDeliveryMode
DYNAMIC: DataDeliveryMode

class KafkaSource(_message.Message):
    __slots__ = ("topic", "proto_message_name", "schema_version")
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    PROTO_MESSAGE_NAME_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    topic: str
    proto_message_name: str
    schema_version: int
    def __init__(self, topic: _Optional[str] = ..., proto_message_name: _Optional[str] = ..., schema_version: _Optional[int] = ...) -> None: ...

class FeatureGroupSource(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class S3Config(_message.Message):
    __slots__ = ("bucket", "access_key_id", "secret_access_key", "region", "endpoint")
    BUCKET_FIELD_NUMBER: _ClassVar[int]
    ACCESS_KEY_ID_FIELD_NUMBER: _ClassVar[int]
    SECRET_ACCESS_KEY_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    bucket: str
    access_key_id: str
    secret_access_key: str
    region: str
    endpoint: str
    def __init__(self, bucket: _Optional[str] = ..., access_key_id: _Optional[str] = ..., secret_access_key: _Optional[str] = ..., region: _Optional[str] = ..., endpoint: _Optional[str] = ...) -> None: ...

class FileSource(_message.Message):
    __slots__ = ("path", "format", "s3_config")
    class Format(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        CSV: _ClassVar[FileSource.Format]
        JSON: _ClassVar[FileSource.Format]
        AVRO: _ClassVar[FileSource.Format]
        PARQUET: _ClassVar[FileSource.Format]
        ORC: _ClassVar[FileSource.Format]
    CSV: FileSource.Format
    JSON: FileSource.Format
    AVRO: FileSource.Format
    PARQUET: FileSource.Format
    ORC: FileSource.Format
    PATH_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    S3_CONFIG_FIELD_NUMBER: _ClassVar[int]
    path: str
    format: FileSource.Format
    s3_config: S3Config
    def __init__(self, path: _Optional[str] = ..., format: _Optional[_Union[FileSource.Format, str]] = ..., s3_config: _Optional[_Union[S3Config, _Mapping]] = ...) -> None: ...

class PostgresSource(_message.Message):
    __slots__ = ("host", "port", "username", "password", "table", "database", "incrementing_column", "timestamp_column")
    HOST_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    PASSWORD_FIELD_NUMBER: _ClassVar[int]
    TABLE_FIELD_NUMBER: _ClassVar[int]
    DATABASE_FIELD_NUMBER: _ClassVar[int]
    INCREMENTING_COLUMN_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_COLUMN_FIELD_NUMBER: _ClassVar[int]
    host: str
    port: int
    username: str
    password: str
    table: str
    database: str
    incrementing_column: str
    timestamp_column: str
    def __init__(self, host: _Optional[str] = ..., port: _Optional[int] = ..., username: _Optional[str] = ..., password: _Optional[str] = ..., table: _Optional[str] = ..., database: _Optional[str] = ..., incrementing_column: _Optional[str] = ..., timestamp_column: _Optional[str] = ...) -> None: ...

class TimestampFormatConfig(_message.Message):
    __slots__ = ("format_type", "format_string", "time_zone")
    class FormatType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        EpochMillis: _ClassVar[TimestampFormatConfig.FormatType]
        EpochSeconds: _ClassVar[TimestampFormatConfig.FormatType]
        StringTimestamp: _ClassVar[TimestampFormatConfig.FormatType]
    EpochMillis: TimestampFormatConfig.FormatType
    EpochSeconds: TimestampFormatConfig.FormatType
    StringTimestamp: TimestampFormatConfig.FormatType
    FORMAT_TYPE_FIELD_NUMBER: _ClassVar[int]
    FORMAT_STRING_FIELD_NUMBER: _ClassVar[int]
    TIME_ZONE_FIELD_NUMBER: _ClassVar[int]
    format_type: TimestampFormatConfig.FormatType
    format_string: str
    time_zone: str
    def __init__(self, format_type: _Optional[_Union[TimestampFormatConfig.FormatType, str]] = ..., format_string: _Optional[str] = ..., time_zone: _Optional[str] = ...) -> None: ...

class Watermark(_message.Message):
    __slots__ = ("time_col", "allowed_delay_seconds", "time_col_config")
    TIME_COL_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_DELAY_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TIME_COL_CONFIG_FIELD_NUMBER: _ClassVar[int]
    time_col: str
    allowed_delay_seconds: int
    time_col_config: TimestampFormatConfig
    def __init__(self, time_col: _Optional[str] = ..., allowed_delay_seconds: _Optional[int] = ..., time_col_config: _Optional[_Union[TimestampFormatConfig, _Mapping]] = ...) -> None: ...

class DataSource(_message.Message):
    __slots__ = ("name", "key_fields", "environment", "encoded_schema", "delivery_mode", "watermark", "file_source", "postgres_source", "kafka_source", "feature_group_source")
    NAME_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELDS_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_FIELD_NUMBER: _ClassVar[int]
    ENCODED_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    DELIVERY_MODE_FIELD_NUMBER: _ClassVar[int]
    WATERMARK_FIELD_NUMBER: _ClassVar[int]
    FILE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    POSTGRES_SOURCE_FIELD_NUMBER: _ClassVar[int]
    KAFKA_SOURCE_FIELD_NUMBER: _ClassVar[int]
    FEATURE_GROUP_SOURCE_FIELD_NUMBER: _ClassVar[int]
    name: str
    key_fields: _containers.RepeatedScalarFieldContainer[str]
    environment: str
    encoded_schema: str
    delivery_mode: DataDeliveryMode
    watermark: Watermark
    file_source: FileSource
    postgres_source: PostgresSource
    kafka_source: KafkaSource
    feature_group_source: FeatureGroupSource
    def __init__(self, name: _Optional[str] = ..., key_fields: _Optional[_Iterable[str]] = ..., environment: _Optional[str] = ..., encoded_schema: _Optional[str] = ..., delivery_mode: _Optional[_Union[DataDeliveryMode, str]] = ..., watermark: _Optional[_Union[Watermark, _Mapping]] = ..., file_source: _Optional[_Union[FileSource, _Mapping]] = ..., postgres_source: _Optional[_Union[PostgresSource, _Mapping]] = ..., kafka_source: _Optional[_Union[KafkaSource, _Mapping]] = ..., feature_group_source: _Optional[_Union[FeatureGroupSource, _Mapping]] = ...) -> None: ...

```

# sources_p2p.py
-## Location -> root_directory.common.sources
```python
# This is an automatically generated file, please do not change
# gen by protobuf_to_pydantic[v0.3.0.2](https://github.com/so1n/protobuf_to_pydantic)
# Protobuf Version: 5.29.1 
# Pydantic Version: 2.10.4 
import typing
from enum import IntEnum

from google.protobuf.message import Message  # type: ignore
from protobuf_to_pydantic.customer_validator import check_one_of
from pydantic import BaseModel, Field, model_validator


class DataDeliveryMode(IntEnum):
    STATIC = 0
    DYNAMIC = 1

class KafkaSource(BaseModel):
    topic: str = Field(default="")
    proto_message_name: str = Field(default="")
    schema_version: int = Field(default=0)

class FeatureGroupSource(BaseModel):
    name: str = Field(default="")

class S3Config(BaseModel):
    bucket: str = Field()
    access_key_id: typing.Optional[str] = Field(default="")
    secret_access_key: typing.Optional[str] = Field(default="")
    region: str = Field()
    endpoint: typing.Optional[str] = Field(default="")

class FileSource(BaseModel):
    class Format(IntEnum):
        CSV = 0
        JSON = 1
        AVRO = 2
        PARQUET = 3
        ORC = 4

    _one_of_dict = {"FileSource.storage_config": {"fields": {"s3_config"}, "required": True}}
    one_of_validator = model_validator(mode="before")(check_one_of)
    path: str = Field(default="")
    format: "FileSource.Format" = Field(default=0)
    s3_config: typing.Optional[S3Config] = Field(default=None)

class PostgresSource(BaseModel):
    host: str = Field(default="")
    port: int = Field(default=0)
    username: str = Field(default="")
    password: str = Field(default="")
    table: str = Field(default="")
    database: str = Field(default="")
    incrementing_column: str = Field(default="")
    timestamp_column: str = Field(default="")

class TimestampFormatConfig(BaseModel):
    class FormatType(IntEnum):
        EpochMillis = 0
        EpochSeconds = 1
        StringTimestamp = 2

    format_type: "TimestampFormatConfig.FormatType" = Field(default=0)
    format_string: typing.Optional[str] = Field(default="")
    time_zone: typing.Optional[str] = Field(default="")

class Watermark(BaseModel):
    time_col: str = Field(default="")
    allowed_delay_seconds: int = Field(default=0)
    time_col_config: typing.Optional[TimestampFormatConfig] = Field(default=None)

class DataSource(BaseModel):
    _one_of_dict = {"DataSource.type": {"fields": {"feature_group_source", "file_source", "kafka_source", "postgres_source"}, "required": True}}
    one_of_validator = model_validator(mode="before")(check_one_of)
    name: str = Field(default="", min_length=1, pattern="^[a-z]([a-z0-9_]{0,48}[a-z0-9])?$")
    key_fields: typing.List[str] = Field(default_factory=list)
    environment: typing.Optional[str] = Field(default="")
    encoded_schema: typing.Optional[str] = Field(default="")
    delivery_mode: DataDeliveryMode = Field(default=0)
    watermark: typing.Optional[Watermark] = Field(default=None)
    file_source: typing.Optional[FileSource] = Field(default=None)
    postgres_source: typing.Optional[PostgresSource] = Field(default=None)
    kafka_source: typing.Optional[KafkaSource] = Field(default=None)
    feature_group_source: typing.Optional[FeatureGroupSource] = Field(default=None)

```