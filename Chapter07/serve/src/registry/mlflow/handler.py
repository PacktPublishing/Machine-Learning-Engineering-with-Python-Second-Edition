from mlflow.client import MlflowClient
from pprint import pprint

# Handle this properly later ...
def check_mlflow_health():
    client = MlflowClient(tracking_uri="http://0.0.0.0:5001") 
    try:
        experiments = client.search_experiments()   
        for rm in experiments:
            pprint(dict(rm), indent=4)
        return 'Service returning experiments'
    except:
        return 'Error calling MLFlow'
    