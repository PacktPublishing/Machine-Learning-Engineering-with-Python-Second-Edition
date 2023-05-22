'''
This script will read in clustered taxi ride data from the clustered_data_{date}.json file and then 
use the OpenAI API to generate a summary of the text where the clustering returned a label of '-1' (i.e an outlier).

Once the summary is generated, it will be saved to a file called 'clustered_summarized_{date}.json' in the same AWS S3 bucket.

The textual data to be summarized is in the 'traffic', 'weather' and 'news' columns of the dataframe.

The prompt will be created using Langchain and will have the following format:

"
The following information describes conditions relevant to taxi journeys through a single day in Glasgow, Scotland.

News: {df['news'][i]}
Weather: {df['weather'][i]}
Traffic: {df['traffic'][i]}

Summarise the above information in 3 sentences or less.
"

The returned text will then be added to the pandas dataframe as df["summary"] and then saved to the clustered_summarized_{date}.json file in AWS S3.
'''

from utils.extractor import Extractor
from textwrap import dedent
import openai
import boto3

class LLMSummarizer:
    def __init__(self, bucket_name: str, file_name: str) -> None:
        self.bucket_name = bucket_name
        self.file_name = file_name

    def summarize(self) -> None:
        extractor = Extractor(self.bucket_name, self.file_name)
        df = extractor.extract_data()
        df['summary'] = ''
        for i in range(len(df)):
            if df['label'][i] == -1:
                prompt = dedent(f"""
                The following information describes conditions relevant to taxi journeys through a single day in Glasgow, Scotland.

                News: {df['news'][i]}
                Weather: {df['weather'][i]}
                Traffic: {df['traffic'][i]}

                Summarise the above information in 3 sentences or less.
                """)
                df['summary'][i] = self.generate_summary(prompt)
        date = datetime.datetime.now().strftime("%Y%m%d")
        boto3.client('s3').put_object(Body=df.to_json(), Bucket=self.bucket_name, Key=f"clustered_summarized_{date}.json")

    def generate_summary(self, prompt: str) -> str:
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0.3,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"]
        )
        return response['choices'][0]['text']