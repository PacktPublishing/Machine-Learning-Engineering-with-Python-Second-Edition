import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow
from mlflow.client import MlflowClient


#plt.rcParams.update({'font.size': 22})

import prophet
from prophet import Prophet
import kaggle

def download_kaggle_dataset(
        kaggle_dataset: str ="pratyushakar/rossmann-store-sales",
        target_path = "./"
    ) -> None:
    api = kaggle.api
    print(api.get_config_value('username'))
    kaggle.api.dataset_download_files(kaggle_dataset, path=target_path, unzip=True, quiet=False)
    
def prep_store_data(df: pd.DataFrame, store_id: int = 4, store_open: int = 1) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns= {'Date': 'ds', 'Sales': 'y'}, inplace=True)
    df_store = df[
        (df['Store'] == store_id) &\
        (df['Open'] == store_open)
    ].reset_index(drop=True)
    return df_store.sort_values('ds', ascending=True)   
    

def train_test_split_forecaster(
    df: pd.DataFrame,
    train_fraction: float 
)->tuple[pd.DataFrame, pd.DataFrame]:
    # grab split data
    train_index = int(train_fraction*df.shape[0])
    df_train = df.copy().iloc[0:train_index]
    df_test = df.copy().iloc[train_index:]
    return df_train, df_test
    
    
def train_forecaster(
    df_train: pd.DataFrame,
    seasonality: dict 
) -> prophet.forecaster.Prophet:
    #create Prophet model
    forecaster=Prophet(
        yearly_seasonality=seasonality['yearly'],
        weekly_seasonality=seasonality['weekly'],
        daily_seasonality=seasonality['daily'],
        interval_width = 0.95
    )
    forecaster.fit(df_train)
    return forecaster

def test_forecaster(
    df_test: pd.DataFrame
) -> None:
    return None

def forecast(
    forecaster: prophet.forecaster.Prophet,
    forecast_index: pd.DataFrame
) -> pd.DataFrame:
    return forecaster.predict(forecast_index)
    
    
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


# def plot_forecast(df_train: pd.DataFrame, df_test: pd.DataFrame, predicted: pd.DataFrame) -> None:
#     fig, ax = plt.subplots(figsize=(20,10))
#     df_test.plot(
#         x='ds', 
#         y='y', 
#         ax=ax, 
#         label='Truth', 
#         linewidth=1, 
#         markersize=5, 
#         color='tab:blue',
#         alpha=0.9, 
#         marker='o'
#     )
#     predicted.plot(
#         x='ds', 
#         y='yhat', 
#         ax=ax, 
#         label='Prediction + 95% CI', 
#         linewidth=2, 
#         markersize=5, 
#         color='red'
#     )
#     ax.fill_between(
#         x=predicted['ds'], 
#         y1=predicted['yhat_upper'], 
#         y2=predicted['yhat_lower'], 
#         alpha=0.15, 
#         color='red',
#     )
#     df_train.iloc[train_index-100:].plot(
#         x='ds', 
#         y='y', 
#         ax=ax, 
#         color='tab:blue', 
#         label='_nolegend_', 
#         alpha=0.5, 
#         marker='o'
#     )
#     current_ytick_values = plt.gca().get_yticks()
#     plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_ytick_values])
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Sales')
#     plt.tight_layout()
#     plt.savefig('store_data_forecast.png')




if __name__ == "__main__":
    import logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
    logging.basicConfig(format = log_format, level = logging.INFO) 
    
    tracking_uri = "http://0.0.0.0:5001"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri) 
    logging.info("Defined MLFlowClient and set tracking URI.")

    #mlflow.set_experiment("prophet_models_05042023")
    #mlflow.autolog()
    
    import os
    
    # If data present, read it in, otherwise, download it 
    #file_path = 'rossman_store_data/train.csv'
    file_path = "./rossman_store_data/"
    if os.path.exists(file_path):
        logging.info('Dataset found, reading into pandas dataframe.')
        df = pd.read_csv(file_path + "train.csv")
    else:
        logging.info('Dataset not found, downloading ...')
        download_kaggle_dataset(target_path=file_path)
        logging.info('Reading dataset into pandas dataframe.')
        df = pd.read_csv(file_path)   
    
    logging.info("Training data retrieved.")
    
    # Transform dataset in preparation for feeding to Prophet
    df = prep_store_data(df)
    logging.info("Transformed data")
    
    with mlflow.start_run():
        logging.info("Started MLFlow run")
        model_name = 'prophet-retail-forecaster'
        mlflow.autolog()
        
        # Define main parameters for modelling
        seasonality = {
            'yearly': True,
            'weekly': True,
            'daily': False
        }

        logging.info("Splitting data")
        # Split the data
        df_train, df_test = train_test_split_forecaster(df=df, train_fraction=0.75)
        logging.info("Data split")
        
        # Train the model
        logging.info("Training model")
        forecaster = train_forecaster(df_train=df_train, seasonality=seasonality)
        run_id = mlflow.active_run().info.run_id
        logging.info("Model trained")
        
        mlflow.prophet.log_model(forecaster, artifact_path="model")
        logging.info("Logged actual model")
    
    # The default path where the MLflow autologging function stores the model
    artifact_path = "model"
    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
    
    # Register the model
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    logging.info("Model registered")
    
    # # Create new model version
    # mv = client.create_model_version(
    #     model_name, 
    #     model_uri, 
    #     run_id, 
    #     description="Prophet model for item demand.") 
    # logging.info("Model version created")
    
    # Transition model to production
    client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage='production',
    )
    logging.info("Model transitioned to prod stage")



