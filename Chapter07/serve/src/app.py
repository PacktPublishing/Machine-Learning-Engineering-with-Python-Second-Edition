from fastapi import FastAPI
from pydantic import BaseModel
import datetime
import pandas as pd

class ForecastRequest(BaseModel):
    store_id: str
    begin_date: str | None = None
    end_date: str | None = None
    
def create_forecast_index(begin_date: str = None, end_date: str = None):
    # if begin_date == None:
    #     begin_date = datetime.datetime.now()
    # else:
    #     begin_date = datetime.datetime.strptime(begin_date, '%Y-%m-%dT%H:%M:%SZ')
        
    # if end_date == None: 
    #     end_date = begin_date + datetime.timedelta(days=7) 
   
    return pd.date_range(start = begin_date, end = end_date, freq = 'D')     
    

app = FastAPI()

@app.get("/")
async def root():
    return {"serviceStatus": "OK"}


@app.post("/forecast/", status_code=200)
async def parse_request(forecast_request: ForecastRequest):
    forecast_request_dict = forecast_request.dict()
    forecast_index = create_forecast_index(begin_date=forecast_request.begin_date, end_date=forecast_request.end_date)
    forecast_request_dict.update({'forecast_index': forecast_index})
    return forecast_request_dict
    # return {
    #     'forecastIndex': create_forecast_index(
    #         begin_date = forecast_request.begin_date, 
    #         end_date = forecast_request.end_date
    #         ), 
    #     forecast_request.dict()
    #     }

# @app.get("/users")
# async def users():
#     users = [
#         {
#             "name": "Mars Kule",
#             "age": 25,
#             "city": "Lagos, Nigeria"
#         },

#         {
#             "name": "Mercury Lume",
#             "age": 23,
#             "city": "Abuja, Nigeria"
#         },

#          {
#             "name": "Jupiter Dume",
#             "age": 30,
#             "city": "Kaduna, Nigeria"
#         }
#     ]

#     return users