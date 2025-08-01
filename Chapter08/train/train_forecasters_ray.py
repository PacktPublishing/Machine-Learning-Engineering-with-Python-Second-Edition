import ray.data
import ray
import pandas as pd
from prophet import Prophet
import logging
import os
import kaggle

# for testing
import time

def download_kaggle_dataset(kaggle_dataset: str ="pratyushakar/rossmann-store-sales") -> None:
    api = kaggle.api
    print(api.get_config_value('username'))
    kaggle.api.dataset_download_files(kaggle_dataset, path="./", unzip=True, quiet=False)

def prep_store_data(
    df: pd.DataFrame, 
    store_id: int = 4, 
    store_open: int = 1
    ) -> pd.DataFrame:
    df_store = df[
        (df['Store'] == store_id) &\
        (df['Open'] == store_open)
    ].reset_index(drop=True)
    df_store['Date'] = pd.to_datetime(df_store['Date'])
    df_store.rename(columns= {'Date': 'ds', 'Sales': 'y'}, inplace=True)
    return df_store.sort_values('ds', ascending=True) 

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
    
    # Convert the pandas DataFrame to a Ray DataFrame if you need this.
    # dataset = ray.data.from_pandas(df)
    
    # You can read the data from a CSV file directly into a Ray DataFrame,
    # but you need this to be in a specific format that our file doesn't have 
    # For example, you'd need to remove the quotes from the csv header.
    # dataset = ray.data.read_csv(file_path) 

    # Get the unique store IDs
    # store_ids = dataset.unique("Store") # if you were using Ray DataFrame
    store_ids = df['Store'].unique()#[0:50] #for testing

    # Define the parameters for the Prophet model
    seasonality = {
        'yearly': True,
        'weekly': True,
        'daily': False
    }
    ray.init(num_cpus=4)
    df_id = ray.put(df)
    
    start = time.time()
    pred_obj_refs, train_obj_refs, test_obj_refs, train_index_obj_refs = map(
        list,
        zip(*([prep_train_predict.remote(df_id, store_id) for store_id in store_ids])),
    )
    
    #Note: could try this as a for loop for fairer comparison?
    ray_results = {
        'predictions': ray.get(pred_obj_refs),
        'train_data': ray.get(train_obj_refs),
        'test_data': ray.get(test_obj_refs),
        'train_indices': ray.get(train_index_obj_refs)
    }
    # predictions = ray.get(pred_obj_refs)
    # train_data = ray.get(train_obj_refs)
    # test_data = ray.get(test_obj_refs)
    # train_indices = ray.get(train_index_obj_refs)
    ray_core_time = time.time() - start

    # print(f'''Predictions: \n{predictions}''')
    # print(f'''Train Data: \n{train_data}''')
    # print(f'''Test Data: \n{test_data}''')
    # print(f'''Train Indices: \n{train_indices}''')
    ray.shutdown()
    #---------------------------------------------
    # Serial training of the Prophet models with a 
    # for loop for comparison
    #---------------------------------------------
    start = time.time()
    predictions = []
    train_data = []
    test_data = []
    train_indices = []
    for store_id in store_ids:
        df_store = prep_store_data(df, store_id=store_id)
        predicted, df_train, df_test, train_index = train_predict(
            df = df_store,
            train_fraction = 0.8,
            seasonality=seasonality
        )
        predictions.append(predicted)
        train_data.append(df_train)
        test_data.append(df_test)
        train_indices.append(train_index)
        
    serial_results = {
        'predictions': predictions,
        'train_data': train_data,
        'test_data': test_data,
        'train_indices': train_indices
    }
    serial_time = time.time() - start
    
    print(f"Models trained (Ray): {len(store_ids)}")
    print(f"Time taken (Ray): {ray_core_time/60:.2f} minutes")
    print(f"Models trained (serial): {len(store_ids)}")
    print(f"Time taken (serial): {serial_time/60:.2f} minutes")

    print("Done!")