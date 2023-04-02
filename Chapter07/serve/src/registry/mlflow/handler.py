from mlflow.client import MlflowClient
import pprint

# Handle this properly later ...
def check_mlflow_health():
    client = MlflowClient(tracking_uri="http://0.0.0.0:5001") 
    for rm in client.search_registered_models():
        pprint(dict(rm), indent=4)
    return 'OK'
