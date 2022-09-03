import dash
import dash_core_components as dcc
import dash_html_components as html
from flask_login.utils import login_required
import plotly.express as px
import pandas as pd
import investpy
from logo_scraper import get_logo
from markowitz import get_fig

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


def create_dash_application(flask_app):
    dash_app = dash.Dash(server=flask_app, name="Dashboard", url_base_pathname="/dash/",
                         external_stylesheets=external_stylesheets)

    fig1 = get_fig()
    data = {'Indices': ['SP500', 'Energy', 'NASDAQ', 'Financials', 'Consumer Staples'], 'perc': [0.2, 0.29, 0.09, 0.01,0.41]}
    df = pd.DataFrame(data)
    fig5 = px.pie(df, values='perc', names='Indices')



    dash_app.layout = html.Div(children=[
        # All elements from the top of the page
        html.Div([
                dcc.Graph(
                    id='graph0',
                    figure=fig1
                ),
            ], className='row'),
        html.Div([
                dcc.Graph(
                    id='graph1',
                    figure=fig5
                ),
            ], className='row')
    ])

    for view_function in dash_app.server.view_functions:
        if view_function.startswith(dash_app.config.url_base_pathname):
            dash_app.server.view_functions[view_function] = login_required(
                dash_app.server.view_functions[view_function]
            )

    return dash_app
