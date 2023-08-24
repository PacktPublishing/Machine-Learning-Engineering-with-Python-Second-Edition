#suggestion drom claude


import ray
from ray import data

# Load data into Ray DataFrame
df = data.read_csv("train.csv") 

# Define preprocessing function
@ray.remote
def prep_store_data(df, store_id, store_open):
  # Preprocess as before
  ...

# Preprocess data for each store
store_dfs = []
for store_id in store_ids:
  store_df = prep_store_data.remote(df, store_id, 1)
  store_dfs.append(store_df)

ray_dfs = ray.get(store_dfs)

# Concatenate store DataFrames
dataframe = data.concat(ray_dfs)

# Train models
results = [train_predict.remote(df, 0.8, seasonality) 
           for df in ray_dfs]

# Rest of code as before
#-----
import ray.dataframe as rdf
import ray
import pandas as pd
from prophet import Prophet
import logging
import os
import kaggle

def download_kaggle_dataset(kaggle_dataset: str ="pratyushakar/rossmann-store-sales") -> None:
    api = kaggle.api
    print(api.get_config_value('username'))
    kaggle.api.dataset_download_files(kaggle_dataset, path="./", unzip=True, quiet=False)

def prep_store_data(df: pd.DataFrame, store_id: int = 4, store_open: int = 1) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns= {'Date': 'ds', 'Sales': 'y'}, inplace=True)
    df_store = df[
        (df['Store'] == store_id) &\
        (df['Open'] == store_open)
    ].reset_index(drop=True)
    return df_store.sort_values('ds', ascending=True) 

@ray.remote
def train_predict(
    df: pd.DataFrame,
    train_fraction: float,
    seasonality: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    
    # grab split data
    train_index = int(train_fraction*df.shape[0])
    df_train = df.copy().iloc[0:train_index]
    df_test = df.copy().iloc[train_index:]

    #create Prophet model
    model=Prophet(
        yearly_seasonality=seasonality['yearly'],
        weekly_seasonality=seasonality['weekly'],
        daily_seasonality=seasonality['daily'],
        interval_width = 0.95
    )

    # train and predict
    model.fit(df_train)
    predicted = model.predict(df_test)
    return predicted, df_train, df_test, train_index

if __name__ == "__main__":
    # If data present, read it in, otherwise, download it 
    file_path = 'train.csv'
    if os.path.exists(file_path):
        logging.info('Dataset found, reading into pandas dataframe.')
        df = pd.read_csv(file_path)
    else:
        logging.info('Dataset not found, downloading ...')
        download_kaggle_dataset()
        logging.info('Reading dataset into pandas dataframe.')
        df = pd.read_csv(file_path)   
    
    # Transform dataset in preparation for feeding to Prophet
    dataset = prep_store_data(df) #inefficient running this on full dataset ... could make remote as well.

    # Convert the pandas DataFrame to a Ray DataFrame
    rdf_data = rdf.from_pandas(dataset, npartitions=10)

    # Define a function to filter the data by store ID
    @ray.remote
    def filter_by_store_id(store_id):
        store_data = rdf_data[rdf_data['store_id'] == store_id]
        return store_data.to_pandas()

    # Get the unique store IDs
    store_ids = dataset['store_id'].unique()
    # Spawn tasks to filter the data by store ID in parallel
    store_data = ray.get([filter_by_store_id.remote(store_id) for store_id in store_ids])

    # Define the parameters for the Prophet model
    seasonality = {
        'yearly': True,
        'weekly': True,
        'daily': False
    }
    # Spawn tasks to train the model for each store
    tasks = [train_predict.remote(df, 0.8, seasonality) for df in store_data]

    # Get the results of the tasks
    results = ray.get(tasks)

    # Concatenate the predicted dataframes for all stores
    predictions = pd.concat([result[0] for result in results])
    # Concatenate the training and test dataframes for all stores
    train_data = pd.concat([result[1] for result in results])
    test_data = pd.concat([result[2] for result in results])
    # Compute the training index for each store
    train_indices = [result[3] for result in results]