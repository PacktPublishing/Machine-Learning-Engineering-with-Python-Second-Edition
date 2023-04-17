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

app = FastAPI()

# Start the in memory cache
@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())

@cache(expire=30)  
async def get_service_handlers():
    mlflow_handler = MLFlowHandler()
    return {'mlflow': mlflow_handler}
    
@app.get("/health/", status_code=200)
async def healthcheck():
    handlers = await get_service_handlers()
    return {
        "serviceStatus": "OK",
        "modelTrackingHealth": handlers['mlflow'].check_mlflow_health(),
        "modelRegistryHealth": False
        }

@cache(expire=30)
async def get_models(store_ids: list):
    handlers = await get_service_handlers()
    models = []
    for store_id in store_ids:
        model = handlers['mlflow'].get_production_model(store_id=store_id)
        models.append(model)
    return {'models': models}

@cache(expire=30)
async def get_model(store_id: str):
    handlers = await get_service_handlers()
    model = handlers['mlflow'].get_production_model(store_id=store_id)
    return {'model': model}
        
@app.post("/forecast2/", status_code=200)
async def parse_request2(forecast_request: List[ForecastRequest]):
    '''
    1. Retrieve each model from model registry
    2. Forecast with it
    3. Cache it
    '''
    #store_ids = []
    forecasts = []
    for item in forecast_request:
        #store_ids.append(item.store_id)
        model_dict = await get_model(item.store_id)
        forecast_input = create_forecast_index(
            begin_date=item.begin_date, 
            end_date=item.end_date
            )
        forecasts.append(model_dict['model'].predict(forecast_input))
    return forecasts    
    # # models_dict = await get_models(store_ids=store_ids)
    # # models = models_dict['models']
    # return "".join([str(x) for x in models])


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






