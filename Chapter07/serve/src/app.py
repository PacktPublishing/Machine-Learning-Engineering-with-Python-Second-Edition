from fastapi import FastAPI
from pydantic import BaseModel
import datetime
import pandas as pd
import pprint
from helpers.request import ForecastRequest, create_forecast_index
from registry.mlflow.handler import MLFlowHandler, check_mlflow_health
from typing import List

# Caching
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

# Logging
import logging
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
logging.basicConfig(format = log_format, level = logging.INFO)

app = FastAPI()

handlers = {}
models = {}
MODEL_BASE_NAME = f"prophet-retail-forecaster-store-"

# Start the in memory cache
@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())
    logging.info("InMemory cache initiated")
    await get_service_handlers()
    logging.info("Updated global service handlers")

async def get_service_handlers():
    mlflow_handler = MLFlowHandler()
    global handlers
    handlers['mlflow'] = mlflow_handler
    logging.info("Retreving mlflow handler {}".format(mlflow_handler))
    return handlers #{'mlflow': mlflow_handler}
    
@app.get("/health/", status_code=200)
async def healthcheck():
    handlers = await get_service_handlers()
    logging.info("Got handlers in healthcheck.")
    return {
        "serviceStatus": "OK",
        "modelTrackingHealth": handlers['mlflow'].check_mlflow_health(),
        "modelRegistryHealth": False
        }

@cache(expire=30)
async def get_models(store_ids: list):
    global handlers
    global models
    models = []
    for store_id in store_ids:
        model = handlers['mlflow'].get_production_model(store_id=store_id)
        models.append(model)
    return models

#@cache(expire=30)
async def get_model(store_id: str):
    global handlers
    global models
    model_name = MODEL_BASE_NAME + f"{store_id}"
    if model_name not in models:
        models[model_name] = handlers['mlflow'].get_production_model(store_id=store_id)
    # logging.info("Got handlers in get_model")
    # logging.info("handlers type = {}".format(type(handlers)))
    # logging.info("handler type = {}".format(type(handlers['mlflow'])))
    # logging.info("handlers dict {}".format(handlers))
    #model = handlers['mlflow'].get_production_model(store_id=store_id)
    #logging.info("model type = {}".format(type(model)))
    return models[model_name]

@app.get("/testcache", status_code=200)
async def test_cache():
    #handlers = await get_service_handlers()
    global handlers
    logging.info(pprint.pprint(str(handlers['mlflow'])))
    return {'type': str(handlers['mlflow'].get_production_model(store_id='4'))}
    

@app.post("/forecast/", status_code=200)
async def parse_request2(forecast_request: List[ForecastRequest]):
    '''
    1. Retrieve each model from model registry
    2. Forecast with it
    3. Cache it
    '''
    forecasts = []
    for item in forecast_request:
        model = await get_model(item.store_id)
        forecast_input = create_forecast_index(
            begin_date=item.begin_date, 
            end_date=item.end_date
            )
        forecast_result = {}
        forecast_result['request'] = item.dict()
        forecast_result['forecast'] = model.predict(forecast_input)[['ds', 'yhat']].to_dict('records')
        forecasts.append(forecast_result)
    return forecasts    