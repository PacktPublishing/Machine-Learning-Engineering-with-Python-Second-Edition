from fastapi import FastAPI
from pydantic import BaseModel
import datetime
import pandas as pd
import pprint
from helpers.request import ForecastRequest, create_forecast_index
from registry.mlflow.handler import check_mlflow_health
    

app = FastAPI()

@app.get("/health/", status_code=200)
async def healthcheck():
    return {
        "serviceStatus": "OK",
        "modelTrackingHealth": check_mlflow_health(),
        "modelRegistryHealth": False
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
