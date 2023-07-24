'''
This file contains the code for the class that takes the taxi ride data from AWS S3 using boto3 and returns a pandas dataframe.
'''
import boto3
from cluster import cluster_and_label
from utils.extractor import Extractor
import datetime

    
def extract_cluster_save_data(bucket_name: str, file_name: str) -> None:
    extractor = Extractor(bucket_name, file_name)
    df = extractor.extract_data()
    df = cluster_and_label(df)
    date = datetime.datetime.now().strftime("%Y%m%d")
    boto3.client('s3').put_object(Body=df.to_json(), Bucket=bucket_name, Key=f"clustered_data_{date}.json")
