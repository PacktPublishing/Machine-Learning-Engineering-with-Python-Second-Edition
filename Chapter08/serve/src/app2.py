from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from helpers.request import ForecastRequest, create_forecast_index
from registry.mlflow.handler import MLFlowHandler
from typing import List

from contextlib import asynccontextmanager

# Logging
import logging
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
logging.basicConfig(format = log_format, level = logging.INFO)

ml_models = {}
service_handlers = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create handlers
    service_handlers['mlflow'] = MLFlowHandler()
    logging.info("Initiatilised mlflow handler {}".format(type(service_handlers['mlflow'])))
    yield
    # Clean up handlers
    service_handlers.clear()
    ml_models.clear()
    logging.info("Handlers and ml models cleared")

#app = FastAPI(lifespan=lifespan)
app = FastAPI()



@app.get("/health/", status_code=200)
async def healthcheck():
    return {
        "serviceStatus": "OK",
        "modelTrackingHealth": service_handlers['mlflow'].check_mlflow_health(),
        "modelRegistryHealth": False
        }


@app.post("/forecast2/", status_code=200)
async def parse_request2(forecast_request: List[ForecastRequest]):
    '''
    1. Retrieve each model from model registry
    2. Forecast with it
    3. Cache it
    '''
    with lifespan(app=app):
        forecasts = []
        for item in forecast_request:
            model_name = 'prophet-retail-forecaster-store-{}'.format(item.store_id)
            if model_name not in ml_models.keys():
                ml_models[model_name] = service_handlers['mlflow'].get_production_model(item.store_id)
            else:
                pass
            forecast_input = create_forecast_index(
                begin_date=item.begin_date, 
                end_date=item.end_date
                )
            forecasts.append(ml_models[model_name].predict(forecast_input))
        return forecasts    


