# #suggestion drom claude

# import ray
# from ray import data

# # Load data into Ray DataFrame
# dataset = data.read_csv("train.csv") 

# # Define preprocessing function
# @ray.remote
# def prep_store_data(df, store_id, store_open):
#     df['Date'] = pd.to_datetime(df['Date'])
#     df.rename(columns= {'Date': 'ds', 'Sales': 'y'}, inplace=True)
#     df_store = df[
#         (df['Store'] == store_id) &\
#         (df['Open'] == store_open)
#     ].reset_index(drop=True)
#     return df_store.sort_values('ds', ascending=True) 

# # Preprocess data for each store
# store_dfs = []
# for store_id in store_ids:
#   store_df = prep_store_data.remote(df, store_id, 1)
#   store_dfs.append(store_df)

# ray_dfs = ray.get(store_dfs)

# # Concatenate store DataFrames
# dataframe = data.concat(ray_dfs)

# # Train models
# results = [train_predict.remote(df, 0.8, seasonality) 
#            for df in ray_dfs]

# Rest of code as before
#-----
import ray.data
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

#@ray.remote
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

@ray.remote(num_returns=4)
def prep_train_predict(
    df: pd.DataFrame,
    store_id: int,
    store_open: int=1,
    train_fraction: float=0.8,
    seasonality: dict={'yearly': True, 'weekly': True, 'daily': False}
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    df = prep_store_data(df, store_id=store_id, store_open=store_open)
    return train_predict(df, train_fraction, seasonality)    
    
# Define a function to filter the data by store ID
@ray.remote
def filter_by_store_id(df, store_id):
    store_data = df[df['Store'] == store_id]
    return store_data#store_data.to_pandas()


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
    #df = prep_store_data(df) #inefficient running this on full dataset ... could make remote as well.

    # Convert the pandas DataFrame to a Ray DataFrame
    dataset = ray.data.from_pandas(df)
    #dataset = ray.data.read_csv(file_path)

    # Get the unique store IDs
    # store_ids = dataset.unique("Store")
    store_ids = df['Store'].unique()#[0:10] #remove [0:10]
    # Spawn tasks to filter the data by store ID in parallel
    #store_data = ray.get([filter_by_store_id.remote(dataset, store_id) for store_id in store_ids])
    #filtered_store_data = ray.get([dataset.filter(lambda x: x["Store"]==store_id) for store_id in store_ids])
    
    # Spawn tasks to prep data
    #prepped_store_data = ray.get([prep_store_data.remote(df, store_id, 1) for (df, store_id) in zip(filtered_store_data, store_ids)])

    # Define the parameters for the Prophet model
    seasonality = {
        'yearly': True,
        'weekly': True,
        'daily': False
    }
    # Spawn tasks to train the model for each store
    # train_predict_tasks = [
    #     prep_train_predict.remote(df, store_id) for store_id in store_ids
    #     ]
    
    pred_obj_refs, train_obj_refs, test_obj_refs, train_index_obj_refs = map(
        list,
        zip(
            *(
                [prep_train_predict.remote(df, store_id) for store_id in store_ids]
            )
        ),
    )

    predictions = ray.get(pred_obj_refs)
    train_data = ray.get(train_obj_refs)
    test_data = ray.get(test_obj_refs)
    train_indices = ray.get(train_index_obj_refs)
    
    # Get the results of the tasks
    #results = ray.get(train_predict_tasks)

    # # Concatenate the predicted dataframes for all stores
    # predictions = pd.concat([result[0] for result in results])
    # # Concatenate the training and test dataframes for all stores
    # train_data = pd.concat([result[1] for result in results])
    # test_data = pd.concat([result[2] for result in results])
    # # Compute the training index for each store
    # train_indices = [result[3] for result in results]