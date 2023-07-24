'''
This script is an example of how to perform transfer learning using PyTorch and well known
NLP models, with open source datasets.

This will include steps for loading an open dataset, transforming the data, freezing the appropriate layers
and then training the non-frozen layers.

We will use The Hewlett Foundation: Automated Essay Scoring dataset from Kaggle, retrieved
using the Kaggle API. The dataset can be found here: https://www.kaggle.com/c/asap-aes/data
'''
import pandas as pd
import kaggle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW

import logging
logging.basicConfig(level=logging.INFO)

# Retrieve the Hewlett essay dataset from Kaggle using the Kaggle API
def download_kaggle_dataset(kaggle_dataset: str ="asap-aes") -> None:
    api = kaggle.api
    print(api.get_config_value('username'))
    kaggle.api.dataset_download_files(kaggle_dataset, path="./data", unzip=True)
    
# load and preprocess the Hewlett essay dataset from the data folder    
def load_and_process(local_dataset_path: str = "./data/training_set_rel3.tsv") -> None: 
    # Load the dataset in to a pandas dataframe
    data = pd.read_csv(local_dataset_path, sep='\t', encoding='ISO-8859-1')

    # Create a new training dataset based on this data
    train_dataset = []

    # Retrieve the pre-trained BART tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    for _, row in data.iterrows():
        essay = row['essay']
        score = row['domain1_score']

        # Preprocess the essay and convert to summary format
        # You may need to modify this based on the specifics of the dataset
        encoded = tokenizer.encode_plus(
            essay,
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()

        summary = f"This essay scored {score} points."  # Create a summary using the score

        train_dataset.append((input_ids, attention_mask, summary))
        
if __name__ == "__main__":
        
    # If data present, read it in, otherwise, download it 
    file_path = './data/training_set_rel3.tsv'
    if os.path.exists(file_path):
        logging.info('Dataset found.')
    else:
        logging.info('Dataset not found, downloading ...')
        download_kaggle_dataset()   
    
    logging.info('Reading dataset into pandas dataframe.')
    df = load_and_process(file_path)