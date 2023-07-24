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
import datetime
import openai
import boto3
import os

openai.api_key = os.environ['OPENAI_API_KEY']

class LLMSummarizer:
    def __init__(self, bucket_name: str, file_name: str) -> None:
        self.bucket_name = bucket_name
        self.file_name = file_name

    def summarize(self) -> None:
        extractor = Extractor(self.bucket_name, self.file_name)
        df = extractor.extract_data()
        df['summary'] = ''
        
        df['prompt'] = df.apply(lambda x: self.format_prompt(x['news'], x['weather'], x['traffic']), axis=1)
        df.loc[df['label']==-1, 'summary'] = df.loc[df['label']==-1, 'prompt'].apply(lambda x: self.generate_summary(x))
        date = datetime.datetime.now().strftime("%Y%m%d")
        boto3.client('s3').put_object(
            Body=df.to_json(orient='records'), 
            Bucket=self.bucket_name, 
            Key=f"clustered_summarized_{date}.json"
        )
    
    def format_prompt(self, news: str, weather: str, traffic: str) -> str:
        prompt = dedent(f'''
            The following information describes conditions relevant to taxi journeys through a single day in Glasgow, Scotland.

            News: {news}
            Weather: {weather}
            Traffic: {traffic}

            Summarise the above information in 3 sentences or less.
            ''')
        return prompt
    def generate_summary(self, prompt: str) -> str:
        # Try chatgpt api and fall back if not working
        try:
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                temperature = 0.3,
                messages = [{"role": "user", "content": prompt}]
            )
            return response.choices[0].message['content']
        except:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt = prompt
            )
            return response['choices'][0]['text']