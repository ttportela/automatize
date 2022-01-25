# -*- coding: utf-8 -*-
'''
Created on Dec, 2021

@author: Tarlis Portela
'''
# import sys, os 
# sys.path.insert(0, os.path.abspath(os.path.join('.')))

import dash

import dash_bootstrap_components as dbc

# Boostrap CSS.
external_stylesheets=[dbc.themes.BOOTSTRAP]

# app = dash.Dash(__name__, suppress_callback_exceptions=True)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, prevent_initial_callbacks=True, 
                title='Multiple Aspect Trajectory Analytics Tool', suppress_callback_exceptions=True)

server = app.server
