# -*- coding: utf-8 -*-
'''
Automatize: Multi-Aspect Trajectory Data Mining Tool Library
The present application offers a tool, called AutoMATize, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. The AutoMATize integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
License GPL v.3 or superior

@author: Tarlis Portela
'''
import sys, os 
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join('.')))

import base64
import datetime
import io

import dash
from dash import dash_table
from dash import dcc
from dash import html
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State

from automatize.assets.routes.page_analysis import render_page_analysis
from automatize.assets.routes.page_datasets import render_page_datasets
from automatize.assets.routes.page_experiments import render_page_experiments
from automatize.assets.routes.page_results import render_page_results

from automatize.assets.app_base import *
from automatize.assets.config import *
# ------------------------------------------------------------

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return render_page_home()
    elif pathname == '/analysis':
        return render_page_analysis()
    elif pathname == '/methods':
        return render_markdown_file(PAGES_ROUTE+'/pages/methods.md', div=True)
    elif '/datasets' in pathname:
        return render_page_datasets(pathname)
    elif pathname == '/experiments':
        return render_page_experiments(pathname)
    elif pathname == '/results':
        return render_page_results(pathname) #render_markdown_file('automatize/assets/experiments.md')
    elif pathname == '/publications':
        return render_markdown_file(PAGES_ROUTE+'/pages/publications.md', div=True)
    else:
        file = PAGES_ROUTE+ pathname+'.md'
#         print(pathname, file)
        if os.path.exists(file):
            return render_markdown_file(file, div=True)
        else:
            return underDev(pathname)
    # You could also return a 404 "URL not found" page here
    
light_logo = True
app.layout = html.Div(id = 'parent', children = [
        html.Nav(className='navbar navbar-expand-lg navbar-dark bg-primary', 
            style={'padding-left': '10px', 'padding-right': '10px'},
            id='app-page-header',
            children=[
                # represents the URL bar, doesn't render anything
                dcc.Location(id='url', refresh=False),
                html.A(className='navbar-brand',
                    children=[
                        html.Img(src='/assets/favicon.ico', width="30", height="30"),
                        page_title,
                    ],
                    href="/",
                ),
                html.Div(style={'flex': 'auto'}),#, children=[
                html.Ul(className='navbar-nav', children=[
#                     html.Li(className='nav-item', children=[
#                         html.A(className='nav-link',
#                             children=['Home'],
#                             href="/",
#                         ),
#                     ]),
                    html.Li(className='nav-item', children=[
                        html.A(className='nav-link',
                            children=['Datasets'],
                            href="/datasets",
                        ),
                    ]),
                    html.Li(className='nav-item', children=[
                        html.A(className='nav-link',
                            children=['Methods'],
                            href="/methods",
                        ),
                    ]),
                    html.Li(className='nav-item', children=[
                        html.A(className='nav-link',
                            children=['Scripting'],
                            href="/experiments",
                        ),
                    ]),
                    html.Li(className='nav-item', children=[
                        html.A(className='nav-link',
                            children=['Results'],
                            href="/results",
                        ),
                    ]),
                    html.Li(className='nav-item', children=[
                        html.A(className='nav-link',
                            children=['Analysis'],
                            href="/analysis",
                        ),
                    ]),
                    html.Li(className='nav-item', children=[
                        html.A(className='nav-link',
                            children=['Publications'],
                            href="/publications",
                        ),
                    ]),
                    html.Li(className='nav-item', children=[
                        html.A(className='nav-link',
                            children=['About Me'],
                            href="http://tarlis.com.br/",
                        ),
                    ]),
                    html.Li(className='nav-item', children=[
                        html.A(className='nav-link nav-link-btn',
                            id='gh-link',
                            children=['View on GitHub'],
                            href="https://github.com/ttportela/automatize",
                        ),
                    ]),
                ]),
            ],
        ),
    
        html.Div(id='page-content'),
    ]
)

def render_page_home():
    y = datetime.datetime.now().date().year
#     return render_markdown_file(README)
    return html.Div(id='content-home', children=[ 
        render_markdown_file(README),
        html.Hr(),
        html.Span('© '+str(y)+' Beta version, by '),
        html.A(
            children=['Tarlis Tortelli Portela'],
            href="https://tarlis.com.br",
        ),
        html.Span('.'),
    ], style={'margin': '20px'})

if __name__ == '__main__':
#     sess.init_app(app)
    
    app.run_server(host=HOST, port=PORT, debug=DEBUG)