#Note - reuses some code from chapter 3
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import StringType, IntegerType, DoubleType
from pyspark.sql import functions as f
from pyspark.sql.functions import pandas_udf, udf
from pprint import pprint

# Create spark context
sc = SparkContext("local", "Ch6BasicExampleApp")
# Get spark session
spark = SparkSession.builder.getOrCreate()

# # Get the data and place it in a spark dataframe
# data = spark.read.format("csv").option("sep", ";").option("inferSchema", "true").option("header", "true").load(
#     "../chapter1/stream-classifier/data/bank/bank.csv")


import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import os

# assume you have already run 'start-mlflow-server.sh'
mlflow.set_tracking_uri("http://localhost:5000")

stage = 'Production'
model_name = "sk-learn-std-scale-clf"
source_rel_path_prefix = '../chapter3/mlflow-advanced'

client = MlflowClient()
for mv in client.search_model_versions("name='{}'".format(model_name)):
    if mv.current_stage == 'Production':
        source_path = os.path.join(source_rel_path_prefix, mv.source)

# Fetch model based on stage name ...
model = mlflow.pyfunc.load_model(
    #model_uri=f"models:/{model_name}/{stage}"
    #model_uri=f"../chapter3/mlflow-advanced/artifacts/0/148ac6e95aad434488415f69aa791ee5/artifacts/sklearn-model/."
    model_uri=source_path
)
print(source_path)

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=source_path)

from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)

print(X)

from pyspark.sql.functions import struct
df = spark.createDataFrame(X.tolist())
# df.withColumn('my_predictions', loaded_model(struct(['_{}'.format(x) for x in range(1, 14)]))).show()

#@udf(returnType=IntegerType())

import pandas as pd
@pandas_udf(returnType=DoubleType())
def predict_pd_udf(*cols):
    X = pd.concat(cols, axis=1)
    return pd.Series(X[:,0])

col_names = ['_{}'.format(x) for x in range(1, 14)]

df_pred = df.select(predict_pd_udf(*col_names).alias('class'))
df_pred.take(5)










