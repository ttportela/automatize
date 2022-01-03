# -*- coding: utf-8 -*-
'''
Created on Dec, 2021

@author: Tarlis Portela
'''
import sys, os 
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
import dash_cytoscape as cyto

import pandas as pd
import networkx as nx

from automatize.movelets import *

# ------------------------------------------------------------
# attributes = ['None']
# sel_attributes = []
# ------------------------------------------------------------
def render_model_filter(movelets=[], model='', from_value=0, to_value=100, attributes=[], sel_attribute=''):
    return html.Div([
            html.Strong('Range of Movelets ('+str(len(movelets))+'): '),
            dcc.RangeSlider(
                id='input-range-mov-graph',
                min=0,
                max=len(movelets) if len(movelets) > 0 else 100,
                value=[from_value, to_value],
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div([
                html.Div([
                    html.Strong('Attributes: '),
                    dcc.Dropdown(
                        id='input-attr-mov-graph',
                        options=[
                            {'label': attr, 'value': attr} for attr in attributes
    #                         {'label': 'All attributes', 'value': str(sel_attribute)}
                        ],
                        value=sel_attribute,
                        style = {'width':'100%'},
                    ),
                ], style={'width': '50%', 'padding': 10, 'flex': 1}),
                html.Div([
                    html.Strong('Graph Format: '),
                    dcc.Dropdown(
                        id='input-format-mov-graph',
                        options=[
                            {'label': 'Sankey Model',    'value': 'sankey'},
                            {'label': 'Markov Model',    'value': 'markov'},
    #                         {'label': 'Tree (as graph)', 'value': 'graph'},
                            {'label': 'Tree (as text)',  'value': 'tree'},
                        ],
                        value=model,
                        style = {'width':'100%'},
                    ),
                ], style={'width': '50%', 'padding': 10, 'flex': 1}),
            ], style={'display': 'flex', 'flex-direction': 'row'}),
            html.Hr(),
        ], style={'width': '100%'})

def render_model(movelets=[], model='', from_value=0, to_value=100, attributes=[], sel_attribute=None):
    
    if sel_attribute == '':
        sel_attribute = None
    
    ls_movs = movelets[from_value : 
            (to_value if to_value <= len(movelets) else len(movelets))]
    
    if model == 'markov':
#         G = movelets_markov(ls_movs, sel_attribute)
        name, nodes, edges, groups, no_colors, ed_colors = movelets_markov2(ls_movs, sel_attribute)
        G = graph_nx(name, nodes, edges, groups, no_colors, ed_colors, draw=False)
        
        fig = cyto.Cytoscape(
            id='graph-'+model,
#             layout={'name': 'preset'},
            layout={'name': 'circle'},
            style={'width': '100%', 'height': '800px'}, #, 'height': '400px'
            elements=G,
            stylesheet=[
                {
                    'selector': 'node',
                    'style': {
                        'background-color': 'data(color)',
                        'line-color': 'data(color)',
                        'label': 'data(label)',
                    }
                },
                {
                    'selector': 'edge',
                    'style': {
                        'curve-style': 'bezier',
                        'background-color': 'data(color)',
                        'line-color': 'data(color)',
                        'target-arrow-color': 'data(color)',
                        'target-arrow-shape': 'triangle',
                        'label': 'data(weight)',
                    }
                }
            ]
        )
    elif model == 'sankey':
        G = movelets_sankey(ls_movs, sel_attribute)
        fig = dcc.Graph(
            id='graph-'+model,
            style = {'width':'100%'},
            figure=G
        )
#     elif model == 'graph':
#         G = []
#         if len(ls_movs) > 0:
#             tree = createTree(ls_movs)
# #             G = convert2anytree(tree)
#             G = convert2digraph(tree)
#         fig = dcc.Graph(
#             id='graph-'+model,
#             style = {'width':'100%'},
#             figure=G
#         )
    elif model == 'tree':
        fig = html.Div(render_tree(ls_movs.copy()))
    else:
        fig = html.H4('...')
    
    return [
        render_model_filter(movelets, model, from_value, to_value, attributes, sel_attribute),
        html.Div(style = {'width':'100%'}, children = [fig])
    ]

def render_tree(ls_movs):
    components = []
    
    if len(ls_movs) > 0:
        tree = createTree(ls_movs)
        s = tree.traversePrint()
        for line in s.split('\n'):
            components += [
                html.Br(),
                html.Span(line),
            ]
    
    return components
