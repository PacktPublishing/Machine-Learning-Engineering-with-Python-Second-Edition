import pandas as pd
import boto3

class Extractor:
    def __init__(self, bucket_name: str, file_name: str) -> None:
        self.bucket_name = bucket_name
        self.file_name = file_name

    def extract_data(self) -> pd.DataFrame:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=self.bucket_name, Key=self.file_name)
        df = pd.read_json(obj['Body'])
        return df