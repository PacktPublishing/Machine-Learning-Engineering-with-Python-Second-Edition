import mlflow.pyfunc
from mlflow.client import MlflowClient

import logging
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
logging.basicConfig(format = log_format, level = logging.INFO)

tracking_uri = "http://0.0.0.0:5001"
client = MlflowClient(tracking_uri=tracking_uri) 

model_name = 'prophet-retail-forecaster'


latest_version_number = client.get_latest_versions(name="prophet-retail-forecaster")[0].version

model_production_uri = client.get_model_version_download_uri(name=model_name, version=latest_version_number)


#model_production_uri = "models:/{model_name}/production".format(model_name=model_name)

#print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))
# need to get this working remotely ...
model_production = mlflow.pyfunc.load_model("/Users/apmcm/dev/Machine-Learning-Engineering-with-Python-Second-Edition/Chapter07/register/artifacts/0/b0747ee336464e7085f95e590db770fd/artifacts/prophet-model")
logging.info("Downloaded model")