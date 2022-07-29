from fredapi import Fred
import pandas as pd 
import http.client
import json
from datetime import datetime,timedelta
fred = Fred(api_key='177ad7df08f3dc82a72e19844c282ad7')
import investpy 
from dateutil.relativedelta import *

def get_indicators(series, start, end):
    a = datetime.strptime(start,'%d/%m/%Y')
    b = datetime.strptime(end,'%d/%m/%Y')
    start = a.strftime('%m/%d/%Y')
    end = b.strftime('%m/%d/%Y')
    indicators = pd.DataFrame()
    indicators.index = pd.date_range(start=start, end=end, freq='MS')
    for key, value in series.items():
        indic = fred.get_series(value, observation_start=start).dropna()
        indic.index = pd.DatetimeIndex(indic.index)
        indic = indic[~indic.index.to_period('m').duplicated()].iloc[:len(indicators.index)]
        if len(indic) < len(indicators.index):
            indic = indic.append([indic[-1:]]*(len(indicators.index) - len(indic)),ignore_index=True)
        indic.index = indicators.index
        indicators[key] = indic
    return indicators

def get_indicators_change(series, start, end):
    indicators = get_indicators(series, start, end)
    indicators = indicators.pct_change().dropna()*100
    return indicators.round(2)



def get_gdp():
    conn = http.client.HTTPSConnection("sbcharts.investing.com")

    headers = { 'cookie': "udid=f92ac5f987d748ac19f54a7fd6648d8b; smd=f92ac5f987d748ac19f54a7fd6648d8b-1658937030; __cf_bm=84Yik1wF4QyCI7vhNYGb.gQdqSQPekJb9op.YIlxfWg-1658937030-0-AYYxo34miq20Y8qt0wtFq67eto8kysvPl99aHX0zBOd7EhvO0ssenXzaL0KXTkLYSGBdPtbadzO0e%2Frk20NOevI%3D; __cflb=02DiuGJ2571ivhYYHJPXE4RT4BvGX6a9JaX8dphan3buv" }

    conn.request("GET", "/events_charts/eu/343.json", payload, headers)

    res = conn.getresponse()
    data = res.read()

    gdp = dict()
    data = json.loads(data.decode("utf-8"))
    for x in data['data']:
        date =datetime.fromtimestamp(int(str(x[0])[:-3])).strftime('%d-%m-%y')
        gdp[date]=x[1]
        
        return gdp