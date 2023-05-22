'''
This DAG shows how to build your own Extract -> Transform -> Machine Learn -> Load pipeline, with the following tasks:

1. Extract the taxi ride data from json files located in AWS S3 using the boto3 library, pass this into a pandas dataframe called df.
2. Transform and perform ML on the data by clustering the taxi rides using DBSCAN, and adding the cluster labels to the dataframe.
3. Take the df['traffic'], df['weather'] and df['news'] text columns, format them in a prompt and send this to the OpenAI API to generate a summary of the text.
4. Combine the results of the above steps into a single dataframe and then export to JSON in AWS S3 using boto3.


'''
from __future__ import annotations

import json
from textwrap import dedent
import datetime

import pendulum

from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator

from extract import extract_cluster_save_data

bucket_name = "etml-data"

date = datetime.datetime.now().strftime("%Y%m%d")
file_name = f"taxi-rides-{date}.json"


with DAG(
    dag_id="etml_dag",
    start_date=pendulum.datetime(2021, 10, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:
    
    extract_cluster_save_task = PythonOperator(
        task_id="extract_cluster_save",
        python_callable=extract_cluster_save_data,
        op_kwargs={'bucket_name': bucket_name, 'file_name': file_name},
    )
    
    
    
    
    
    