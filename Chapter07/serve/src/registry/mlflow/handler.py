from mlflow.client import MlflowClient
import mlflow
from mlflow.pyfunc import PyFuncModel
from pprint import pprint

class MLFlowHandler:
    def __init__(self) -> None:
        self.client = MlflowClient(tracking_uri="http://0.0.0.0:5001") 
    
    def check_mlflow_health(self) -> None:
        try:
            experiments = self.client.search_experiments()   
            for rm in experiments:
                pprint(dict(rm), indent=4)
                return 'Service returning experiments'
        except:
            return 'Error calling MLFlow'
        
    def get_production_model(self, store_id: int) -> PyFuncModel:
        model_name = f"prophet-retail-forecaster-store-{store_id}"
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/production")
        return model
            

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
    
    
    
    