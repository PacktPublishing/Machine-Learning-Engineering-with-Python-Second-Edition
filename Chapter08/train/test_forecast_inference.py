import mlflow 
import mlflow.pyfunc
from mlflow.client import MlflowClient

import logging
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
logging.basicConfig(format = log_format, level = logging.INFO)

tracking_uri = "http://0.0.0.0:5001"
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri=tracking_uri) 

model_name = 'prophet-retail-forecaster'
latest_version_number = client.get_latest_versions(name="prophet-retail-forecaster")[0].version
model_production_uri = client.get_model_version_download_uri(name=model_name, version=latest_version_number)


#model_production_uri = "models:/{model_name}/production".format(model_name=model_name)

#print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))
# need to get this working remotely ...
model_production = mlflow.pyfunc.load_model("/Users/apmcm/dev/Machine-Learning-Engineering-with-Python-Second-Edition/Chapter07/register/artifacts/0/b0747ee336464e7085f95e590db770fd/artifacts/prophet-model")
logging.info("Downloaded model")

logging.info("trying again")
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/production")
logging.info("worked.")

logging.info("trying predict")
#model.predict()


def get_production_model(store_id:int):
    model_name = f"prophet-retail-forecaster-store-{store_id}"
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/production")
    return model

import datetime
import pandas as pd

def create_forecast_index(begin_date: str = None, end_date: str = None):
    # Convert forecast begin date
    if begin_date == None:
        begin_date = datetime.datetime.now().replace(tzinfo=None)
    else:
        begin_date = datetime.datetime.strptime(begin_date, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=None)
        
    # Convert forecast end date
    if end_date == None: 
        end_date = begin_date + datetime.timedelta(days=7)
    else:
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=None)

    return pd.date_range(start = begin_date, end = end_date, freq = 'D')    

for store_id in ['3', '4', '10']:
    #model_name = f"prophet-retail-forecaster-store-{store_id}"
    #model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/production")
    model = get_production_model(store_id)
    
    forecast_index = create_forecast_index(begin_date="2023-03-01T00:00:00Z", end_date="2023-03-07T00:00:00Z")
    df_ds = pd.DataFrame({'ds': forecast_index})
    print(model.predict(df_ds))
    
    