# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.path.abspath('.'))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
#sys.path.insert(0, os.path.abspath(os.path.join('.')))

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
        return render_markdown_file(PAGES_ROUTE+'assets/pages/methods.md', div=True)
    elif '/datasets' in pathname:
        return render_page_datasets(pathname)
    elif pathname == '/experiments':
        return render_page_experiments(pathname)
    elif pathname == '/results':
        return render_page_results(pathname) #render_markdown_file('./assets/experiments.md')
    elif pathname == '/publications':
        return render_markdown_file(PAGES_ROUTE+'assets/pages/publications.md', div=True)
    elif pathname == '/tutorial':
        return html.Div(id='content-home', children=[html.Iframe(
            src="assets/examples/Automatize_Sample_Code.html", width="100%", height="100vh",
            style={"height": "100vh", "width": "100%"},
        )])
    else:
        file = ASSETS_ROUTE + pathname+'.md'
        #print(os.path.abspath(file))
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
                            children=['Tutorial'],
                            href="/tutorial",
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
        render_markdown_file(README, div=True),
        html.Hr(),
        html.Span('© '+str(y)+' '+VERSION+' version, by '),
        html.A(
            children=['Tarlis Tortelli Portela'],
            href="https://tarlis.com.br",
        ),
        html.Span('.'),
#     ], style={'margin': '20px'})
    ])

if __name__ == '__main__':
#     sess.init_app(app)
    app.run_server(host=HOST, port=PORT, debug=DEBUG)