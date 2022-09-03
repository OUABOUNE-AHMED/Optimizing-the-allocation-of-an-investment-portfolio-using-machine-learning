import pandas as pd
import numpy as np 
from tqdm import tqdm
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
from time import sleep 


def get_fig():
    sleep(5)
    mus = pd.read_csv('Estimated_returns_variance.csv', index_col='Index')
    indices = pd.read_csv('All_Sectors.csv', index_col='Date')
    indices = indices.pct_change().dropna()
    cov = pd.read_csv('Indices_Returns_Cov.csv', index_col='Unnamed: 0')


    #- How many assests to include in each portfolio
    n_assets = 5
    #-- How many portfolios to generate
    n_portfolios = 1000

    #-- Initialize empty list to store mean-variance pairs for plotting
    mean_variance_pairs = []

    np.random.seed(75)
    #-- Loop through and generate lots of random portfolios
    for i in range(n_portfolios):
        #- Choose assets randomly without replacement
        assets = np.random.choice(list(indices.columns.drop('NASDAQ')), n_assets, replace=False)
        #- Choose weights randomly
        weights = np.random.rand(n_assets)
        #- Ensure weights sum to 1
        weights = weights/sum(weights)

        #-- Loop over asset pairs and compute portfolio return and variance
        #- https://quant.stackexchange.com/questions/43442/portfolio-variance-explanation-for-equation-investments-by-zvi-bodie
        portfolio_E_Variance = 0
        portfolio_E_Return = 0
        for i in range(len(assets)):
            portfolio_E_Return += weights[i] * mus['Estimated Return'].loc[assets[i]]
            for j in range(len(assets)):
                #-- Add variance/covariance for each asset pair
                #- Note that when i==j this adds the variance
                portfolio_E_Variance += weights[i] * weights[j] * cov.loc[assets[i], assets[j]]
                
        #-- Add the mean/variance pairs to a list for plotting
        mean_variance_pairs.append([portfolio_E_Return, portfolio_E_Variance])
        

    #-- Plot the risk vs. return of randomly generated portfolios
    #-- Convert the list from before into an array for easy plotting
    mean_variance_pairs = np.array(mean_variance_pairs)

    risk_free_rate=0 #-- Include risk free rate here

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_variance_pairs[:,1]**0.5, y=mean_variance_pairs[:,0], 
                        marker=dict(color=(mean_variance_pairs[:,0]-risk_free_rate)/(mean_variance_pairs[:,1]**0.5), 
                                    showscale=True, 
                                    size=7,
                                    line=dict(width=1),
                                    colorscale="RdBu",
                                    colorbar=dict(title="Sharpe<br>Ratio")
                                    ), 
                        mode='markers'))
    fig.update_layout(template='plotly_white',
                    xaxis=dict(title='Annualised Risk (Volatility)'),
                    yaxis=dict(title='Annualised Return'),
                    title='Sample of Random Portfolios',
                    width=850,
                    height=500)
    fig.update_xaxes(range=[0.15, 0.26])
    fig.update_yaxes(range=[0.035,0.09])
    fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))

    return fig 