import investpy 
import pandas as pd 
from datetime import datetime, timedelta
from dateutil.relativedelta import *

def get_daily_close(names, start, end):
    indices = {}
    for name in names:
        indices[name] = investpy.get_index_historical_data(index=name, country='United States', from_date=start, to_date=end).Close
    indices = pd.DataFrame(indices)
    return indices

def get_daily_returns(names, start, end):

    indices = get_daily_close(names, start, end)
    indices = indices.pct_change().dropna()*100

    return indices 

def get_monthly_close(names, start, end):
    indices = get_daily_close(names, start, end)
    indices = indices[~indices.index.to_period('m').duplicated()]
    end = datetime.strptime(start ,'%d/%m/%Y')+ relativedelta(months= len(indices)-1)
    end_p = end.strftime('%Y/%m/%d')
    indices.index = pd.date_range(start= start, end= end_p , freq='MS')

    return indices

def get_monthly_returns(names, start, end):
    indices = get_daily_close(names, start, end)
    indices = indices[~indices.index.to_period('m').duplicated()]
    end = datetime.strptime(start ,'%d/%m/%Y')+ relativedelta(months= len(indices)-1)
    end_p = end.strftime('%Y/%m/%d')
    indices.index = pd.date_range(start= start, end= end_p , freq='MS')
    indices = indices.pct_change().dropna()*100
    return indices.round(2)

def convert_return_to_price(start_point , prediction):
    predicted_price = pd.DataFrame(index = prediction.index)
    predicted_price['prediction'] = start_point * (1 +  (prediction/100)).cumprod()
    
    return predicted_price