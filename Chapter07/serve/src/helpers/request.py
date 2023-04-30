from pydantic import BaseModel
import datetime
import pandas as pd
 
 # REQUEST UTILITIES
class ForecastRequest(BaseModel):
    store_id: str
    begin_date: str | None = None
    end_date: str | None = None

def create_forecast_index(begin_date: str = None, end_date: str = None):
    # Convert forecast begin date
    if begin_date == None:
        begin_date = datetime.datetime.now().replace(tzinfo=None)
    else:
        begin_date = datetime.datetime.strptime(begin_date, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=None)
        
    # Convert forecast end date
    if end_date == None: 
        end_date = begin_date + datetime.timedelta(days=7)
    else:
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=None)

    forecast_index = pd.date_range(start = begin_date, end = end_date, freq = 'D')
    # Format for Prophet to consume
    return pd.DataFrame({'ds': forecast_index})

     