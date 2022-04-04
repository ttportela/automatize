# -*- coding: utf-8 -*-
'''
Automatize: Multi-Aspect Trajectory Data Mining Tool Library
The present application offers a tool, called AutoMATize, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. The AutoMATize integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
License GPL v.3 or superior

@author: Tarlis Portela
'''
import dash
import dash_bootstrap_components as dbc

import flask
from automatize.assets.config import page_title

# Server Config
HOST = '0.0.0.0'
PORT = 8050
DEBUG = True


# Boostrap CSS.
external_stylesheets=[dbc.themes.BOOTSTRAP]

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets, prevent_initial_callbacks=True, 
#                 title=page_title, suppress_callback_exceptions=True)
# server = app.server

server = flask.Flask('automatize')

app = dash.Dash('automatize', server=server,external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
app.title = page_title
app._favicon = 'favicon.ico'
