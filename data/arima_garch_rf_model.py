import pmdarima
import arch
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import investpy 
import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from dateutil.relativedelta import *
import sys 
sys.path.append('..')
from data.indicators import get_indicators, get_indicators_change
from data.indices import get_daily_close, get_daily_returns, get_monthly_close, get_monthly_returns


series = {   
"Manufacturing employement" : "MANEMP",
"Unemployement rate"        : "UNRATE",        
"Consumer Sentiment"        : "UMCSENT",        
"Auto Sales"                : "TOTALNSA",        
"CPI Urban"                 : "CPIAUCNS",        
"Employement rate"          : "LREM64TTUSM156N",        
"Auto Production"           : "G17MVSFAUTOS",        
"CPI Oil"                   : "CUUR0000SEHE",        
"CPI Less Food & Energy"    : "CPILFENS",        
"Effr"                      : "EFFR",        
"M1"                        : "M1NS",        
"M2"                        : "M2NS",        
"M3"                        : "MABMM301USM189S",        
"Mortgage rate 30"          : "MORTGAGE30US",        
"Mortgage rate 15"          : "MORTGAGE30US",        
"Home price"                : "CSUSHPISA",        
"House start"               : "HOUSTNSA",        
"10-2 years rate"           : "T10Y2Y",        
"Deposits"                  : "DPSACBM027NBOG",        
"Industrial Production"     : "INDPRO",        
"PMI Manufacturing"         : "PCUOMFGOMFG",       
}


class arima_garch_rf_model:
    
    def __init__(self,sector,ind_start,tr_end,start1,ind_end,series = series):
        self.sector = sector
        self.ind_start = ind_start
        self.ind_end = ind_end
        self.tr_end = tr_end
        self.start1 = start1
        self.series = series
        
        
    def simulate_GARCH(self,observations, omega, alpha, beta = 0):
        
        np.random.seed(4)
        # Initialize the parameters
        white_noise = np.random.normal(size = observations)
        resid = np.zeros_like(white_noise)
        variance = np.zeros_like(white_noise)

        for t in range(1, observations):
            # Simulate the variance (sigma squared)
            variance[t] = omega + alpha * resid[t-1]**2 + beta * variance[t-1]
            # Simulate the residuals
            resid[t] = np.sqrt(variance[t]) * white_noise[t]    

        return resid, variance


    def arima_garch_model(self):
        
        try: 
            a = datetime.strptime(self.tr_end,'%d/%m/%Y')
            self.tr_end = a.strftime('%Y/%m/%d')
        except : self.tr_end = self.tr_end    

        a= datetime.strptime(self.tr_end ,'%Y/%m/%d')+ relativedelta(months=1)
        self.start = a.strftime('%Y/%m/%d')

        global sp_return
        sp_return = get_monthly_returns([self.sector], self.ind_start, self.ind_end) 

        sp_return_trn = sp_return.loc[:self.tr_end]

        global test_date
        test_date = sp_return.loc[self.start:].index

        global df_Close_monthly
        df_Close_monthly = get_monthly_close([self.sector], self.ind_start, self.ind_end) 

        # fit a GARCH(1,1) model on the residuals of the ARIMA model
        garch1 = arch.arch_model(sp_return_trn, p=1, q=1);
        garch_fitted1 = garch1.fit();

        # fit ARIMA on returns 
        arima_model_fitted = pmdarima.auto_arima(sp_return_trn);
        arima_residuals = arima_model_fitted.arima_res_.resid

        # fit a GARCH(1,1) model on the residuals of the ARIMA model
        garch = arch.arch_model(arima_residuals, p=1, q=1);
        garch_fitted = garch.fit();

        # Use ARIMA to predict mu
        Model = ARIMA(sp_return_trn, order=arima_model_fitted.order)
        Fited_model =  Model.fit()
        arima_predict = pd.DataFrame(columns=[])
        arima_predict['predicted'] = Fited_model.forecast(steps=len(sp_return))  
        arima_predict['Date'] = sp_return.index 
        arima_predict= arima_predict.set_index('Date')

        # Use GARCH to predict the residual
        sim_resid, sim_variance= self.simulate_GARCH(observations =len(test_date) ,omega = garch_fitted.params[1], alpha = garch_fitted.params[2], beta = garch_fitted.params[3]) 

        error_monthly = pd.DataFrame()
        error_monthly['Residual'] = pd.concat([garch_fitted1.resid, pd.Series(sim_resid) ],ignore_index = True)
        error_monthly['date']= sp_return.index
        error_monthly = error_monthly.set_index('date')

        # Combine both models' output: yt = mu + et
        adj_prediction_monthly_arima_garch = arima_predict.predicted + error_monthly.Residual

        return adj_prediction_monthly_arima_garch

    
    def important_indicators(self):

        dates={'tr_end' : self.tr_end ,'start' : self.tr_end ,'start1' : self.start1 ,'tst_end' : self.start1 }

        #set a unique date format D/ M/Y : 
        for key in dates:
            if key == 'start' or key=='tst_end':
                try:
                    dates[key] = datetime.strptime(dates[key] ,'%d/%m/%Y')+ relativedelta(months=1)
                    dates[key] = dates[key].strftime('%Y/%m/%d')
                except:continue    
            else:continue
                #a = datetime.strptime(dates[key],'%d/%m/%Y')
                #dates[key] = a.strftime('%Y/%m/%d')
                   

        self.tr_end = list(dates.values())[0] 
        self.start =list(dates.values())[1] 
        self.start1 = list(dates.values())[2] 
        self.tst_end = list(dates.values())[3]     


        sp_return = get_monthly_returns([self.sector], self.ind_start, self.ind_end) 

        sp_return_trn = sp_return.loc[:self.tr_end]
        sp_return_tst1 = sp_return.loc[self.start:self.start1]
        sp_return_tst2 = sp_return.loc[self.tst_end:]

        test_date = sp_return.loc[self.start:].index

        global df_Close_monthly
        df_Close_monthly = get_monthly_close([self.sector], self.ind_start, self.ind_end)  


        # Combine both models' output: yt = mu + et
        global adj_prediction_monthly
        adj_prediction_monthly = self.arima_garch_model()

        RF= np.array(sp_return).flatten()-np.array(adj_prediction_monthly).flatten()
        RF_error = pd.DataFrame(columns =[])
        RF_error['RF_error']=RF
        RF_error['date']= adj_prediction_monthly.index
        RF_error=RF_error.set_index('date')
        
        global RF_error_trn ,RF_error_tst1 ,RF_error_tst2
        RF_error_trn = RF_error.loc[:self.start]
        RF_error_tst1 = RF_error.loc[self.start:self.start1]
        RF_error_tst2 = RF_error.loc[self.tst_end:]

        indicators = get_indicators(self.series, self.ind_start, self.ind_end)  


        #gdp = indicators.gdp.dropna() under work

        indicators = 100*indicators.pct_change().dropna()#.loc[:,'gold':'pmi']

        indicators.loc['2019-10-01']['10-2 years rate'] = 0 

        #indicators['gdp'] = get_gdp() #gdp.dropna() under work 

        x_train  = indicators.loc[self.start:self.start1]

        y_train = RF_error_tst1

        x_test = indicators.loc[:,:].loc[self.tst_end:]

        y_test = RF_error_tst2


        sel = SelectFromModel(RandomForestRegressor(n_estimators = 100,random_state=42))
        sel.fit(x_train, y_train)
        selected_feat= x_train.columns[(sel.get_support())]
        l = list(selected_feat)
        
        #indicators = indicators[l]
        
        return indicators[l]
    


    def adj_prediction_monthly_rf(self):
        
        indicators = self.important_indicators()
        
        if 'm3' in indicators:
            indicators.loc[:,'m3']['2022-04-01'] = 0
        indicators = indicators.dropna() 

        x_train  = indicators.loc[self.start:self.start1]

        y_train =   RF_error_tst1 

        x_test = indicators.loc[self.tst_end:]

        y_test = RF_error_tst2

        #y_train = y_train[tst_end:]


        # Instantiate model with 1000 decision trees
        rf = RandomForestRegressor(n_estimators = 10, random_state = 42)
        # Train the model on training data
        rf.fit(x_train, y_train);

        # Use the forest's predict method on the test data
        predictions = rf.predict(x_test)
        df_predictions = pd.DataFrame(columns=['value'],data=predictions)
        df_predictions['date'] = RF_error_tst2.index
        df_predictions = df_predictions.set_index('date')

        adj_argaf = adj_prediction_monthly.loc[self.tst_end:]+predictions
        start_point =  df_Close_monthly.loc[self.tst_end] 

        adj_prediction_monthly_rf =pd.DataFrame(columns=[])
        adj_prediction_monthly_rf['final_pred'] = pd.concat([adj_prediction_monthly.loc[:self.start1], pd.Series(adj_argaf) ],ignore_index = True)
        adj_prediction_monthly_rf['date']=pd.concat([pd.Series(adj_prediction_monthly.loc[:self.start1].index) , pd.Series(adj_argaf.index)],ignore_index=True)
        adj_prediction_monthly_rf = adj_prediction_monthly_rf.set_index('date')

        return adj_prediction_monthly_rf, start_point
