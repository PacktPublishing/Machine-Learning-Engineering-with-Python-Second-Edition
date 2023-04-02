from fastapi import FastAPI
from pydantic import BaseModel
import datetime
import pandas as pd
import pprint
from helpers.request import ForecastRequest, create_forecast_index
from registry.mlflow.handler import check_mlflow_health
    
# # MLFLOW HANDLERS
# def check_mlflow_health():
#     client = MLFlowClient(tracking_uri="http://0.0.0.0:5001") 
#     for rm in client.search_registered_models():
#         pprint(dict(rm), indent=4)
#     return 'OK'
    

# KSERVE SERVICE HANDLERS

app = FastAPI()

@app.get("/health/", status_code=200)
async def healthcheck():
    return {
        "serviceStatus": "OK",
        "modelRegistryHealth": check_mlflow_health()
        }


@app.post("/forecast/", status_code=200)
async def parse_request(forecast_request: ForecastRequest):
    forecast_request_dict = forecast_request.dict()
    forecast_index = create_forecast_index(
        begin_date=forecast_request.begin_date, 
        end_date=forecast_request.end_date
        )
    # This is as a check for now, would like to retain the index
    forecast_request_dict.update({'forecast_index_strings': forecast_index.to_series(keep_tz=False).astype(str).tolist()})
    return forecast_request_dict
