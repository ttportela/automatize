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
                            {'label': 'Class Heat Map', 'value': 'attrHeatMap'},
                            {'label': 'Tree Model',  'value': 'tree'},
                            {'label': 'Quality Tree', 'value': 'qtree'},
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
    elif model == 'qtree':
        fig = html.Div(render_quality_tree(ls_movs))
    elif model == 'attrHeatMap':
        fig = html.Div(render_attr_heatmap(ls_movs))        
    else:
        fig = html.H4('...')
    
    return [
        render_model_filter(movelets, model, from_value, to_value, attributes, sel_attribute),
        html.Div(style = {'width':'100%'}, children = [fig])
    ]


## ------------------------------------------------------------------------------------------------
def render_tree(ls_movs):
    ncor = 7
    def getTitleElem(x, ident=1):
        return html.A(id='tree-link', children=getMoveletBox(x.data, ident))
#         return html.A(id='tree-link', children=html.Span('{:3.2f}'.format(x.data.quality)+'%'))
    
    def render_element(root, ident=1):
        if len(root.children) > 0:
            return [getTitleElem(root, ident),
                    html.Ul(
                        [html.Li(render_element(x, ident+1)) for x in root.children]
                    )]
        else:
            return getTitleElem(root, ident)
    
    if len(ls_movs) > 0:
        tree = createTree(ls_movs)    
        return [ html.Div(html.Ul([html.Li(render_element(tree))]), className='tree') ]
    
    return [html.Span('No movelets to render a tree')]


## ------------------------------------------------------------------------------------------------
def render_quality_tree(ls_movs):
    ncor = 7
    def getTitleElem(x, ident=1):
        return html.A(id='tree-link', children=[
            html.Span('Mov-'+str(x.data.mid)),
            html.Br(),
            html.Span('{:3.2f}'.format(x.data.quality)+'%')
        ])
    
    def render_element(root, ident=1):
        if len(root.children) > 0:
            return [getTitleElem(root, ident),
                    html.Ul(
                        [html.Li(render_element(x, ident+1)) for x in root.children]
                    )]
        else:
            return getTitleElem(root, ident)
    
    if len(ls_movs) > 0:
        tree = createTree(ls_movs)    
        return [ html.Div(html.Ul([html.Li(render_element(tree))]), className='tree') ]
    
    return [html.Span('No movelets to render a tree')]


## ------------------------------------------------------------------------------------------------
def render_attr_heatmap(ls_movs):
    if len(ls_movs) > 0:
        G = moveletsHeatMap(ls_movs)
        
        buf = io.BytesIO() # in-memory files
        G.savefig(buf, format="png") # save to the above file object
#        G.close()
        data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
        G = "data:image/png;base64,{}".format(data)
        
        fig = html.Img(
            id='graph-movelets-heatmap',
            style = {'width':'100%'},
            src=G
        )
        return [ fig ]
    
    return [html.Span('No movelets to render a tree')]

## ------------------------------------------------------------------------------------------------
def getMoveletBox(m, ident):
    
    n = m.size
    feats = list(m.attributes())
    
    WD = 10
    for i in range(n):
        for attr in feats:
            WD = max(WD, len(str(m.data[i][attr])))
    WD = WD * 12
    
    HG = 20
    
    return html.Div(
        html.Div(
            html.Div([
                html.Span('{:3.2f}'.format(m.quality)+'%', style={"float":"left", "position":"relative", "top":"-5px"}),
                html.Div(''
                         , className="rc-slider-rail"
                         , style={"left": str(int((100/n)))+"%","width":str(int((n-1)*100/n))+"%",}
                ),
                html.Div('', 
                         className="rc-slider-track rc-slider-track-1", 
                         style={"left": str(int((100/n)))+"%","right":"auto","width":str(int((n-1)*100/n))+"%",}
                ),
                
                html.Div([
                    html.Span('', className="", style={"left": "0%"})
                ]+[
                    html.Span('', className="rc-slider-dot rc-slider-dot-active", style={"left": str(int((i+1)*100/n))+"%"})
                for i in range(n)], className="rc-slider-step"),
                
                html.Div([
                    html.Span('mov-'+str(m.mid)
                    , className="rc-slider-mark-text rc-slider-mark-text-active"
                    , style={"transform": "translateX(-50%)", "left": "0%"})
                ]+[
                    html.Span('p'+str(1+m.start+i)
                    , className="rc-slider-mark-text rc-slider-mark-text-active"
                    , style={"transform": "translateX(-50%)", "left": str(int((i+1)*100/n))+"%"})
                for i in range(n)], className="rc-slider-mark"),
                
            ] + \
            [
                
                html.Div([
                    html.Span(str(feats[k])
                    , className="rc-slider-mark-text rc-slider-mark-text-active"
                    , style={"transform": "translateX(-50%)", "left": "0%", "top": str(int(HG*(k+1)))+'px'})
                ]+[
                    html.Span(str(m.data[i][feats[k]])
                    , className="rc-slider-mark-text rc-slider-mark-text-active"
                    , style={"transform": "translateX(-50%)", 
                             "left": str(int((i+1)*100/n))+"%", 
                             "top": str(int(HG*(k+1)))+'px',
                             "width":"max-content",
                            })
                for i in range(n)], className="rc-slider-mark")
                
            for k in range(len(feats))]
            , className="rc-slider rc-slider-with-marks"
            , style={"position":"relative"})
        , style={"padding":"0px 25px 25px"})
    , style={"width": str(int(n*WD))+"px", "height": str(int(HG*(len(feats)+2)))+'px'})


## ------------------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------------------
def old_getMoveletBox(m, ident):
#         ncor = 7
    colClass = ''#'col-sm-2'
    boxStyle = style={"padding-bottom": "25px", "margin-bottom": "5px", 
#            "width": str((len(m.data)+1)*150)+"px", 
           "width": "max-content",
           "text-align": "center"}
    return html.Div([dbc.Row(
        [dbc.Col([ # FIRST COL (ATTR names)
            html.Div([
#                     html.Div('{:3.2f}'.format(m.quality)+'%', 
# #                         className="rc-slider-handle rc-slider-handle-1",
#                         style={"right": "auto",},
# #                         style={"right": "auto", "margin-top": "0"},
#                     ),
                html.Div(
                    html.Span(
                        '{:3.2f}'.format(m.quality)+'%', 
                        className="rc-slider-mark-text rc-slider-mark-text-active",
                        style={"color":"black", "transform": "translateX(-50%)"},
                    ), 
                    className="rc-slider-mark",
                    style={"right": "auto", "top":"0"},
                ),
                html.Div(
                    html.Span(
                        'size: '+str(m.size), 
                        className="rc-slider-mark-text rc-slider-mark-text-active",
                        style={"transform": "translateX(-50%)"},
                    ), 
                    className="rc-slider-mark",
                    style={"right": "auto"},
                ),],
                style={"position": "relative"},
                className="rc-slider rc-slider-with-marks",
            )
        ], className=colClass)] + \
        [dbc.Col([ # SUBSEQ COLs (points)
            html.Div([
                html.Div('', 
                    className="rc-slider-handle rc-slider-handle-1",
                    style={"right": "auto", "transform": "translateX(-50%)"},
#                         style={"right": "auto", "margin-top": "0"},
                ),
                html.Div(
                    html.Span(
                        'p'+str(m.start+i+1), 
                        className="rc-slider-mark-text rc-slider-mark-text-active",
                        style={"transform": "translateX(-50%)"},
                    ), 
                    className="rc-slider-mark",
                    style={"right": "auto"},
                ),],
                style={"position": "relative"},
                className="rc-slider rc-slider-with-marks",
            )
        ], className=colClass) for i in range(len(m.data))],
        style={"padding": "0px 25px 0px"}, 
    )] + \
    [dbc.Row(
        [dbc.Col([ # FIRST COL (attr name)
            html.Div([
                html.Div(
                    html.Span(
                        str(k), 
                        className="rc-slider-mark-text rc-slider-mark-text-active",
                        style={"transform": "translateX(-50%)"},
                    ), 
                    className="rc-slider-mark",
                    style={"right": "auto"},
                ),],
                style={"position": "relative"},
                className="rc-slider rc-slider-with-marks",
            )], 
            className=colClass,
        )] + \
        [dbc.Col([ # SUBSEQ COLs (values)
            html.Div([
#                     html.Div('', 
#                         className="rc-slider-handle rc-slider-handle-1",
#                         style={"right": "auto", "transform": "translateX(-50%)"},
# #                         style={"right": "auto", "margin-top": "0"},
#                     ),
                html.Div(
                    html.Span(
                        str(m.data[i][k]), 
                        className="rc-slider-mark-text rc-slider-mark-text-active",
                        style={"transform": "translateX(-50%)"},
                    ), 
                    className="rc-slider-mark",
                    style={"right": "auto"},
                ),],
                style={"position": "relative", "width": "max-content"},
                className="rc-slider rc-slider-with-marks",
            )],
        className=colClass) for i in range(len(m.data)) ],
        style={"padding": "0px 25px 0px"},
    ) for k in m.attributes()],
    style=boxStyle,
    className='movelet-box')

def old_render_tree(ls_movs):        
    def getTitleElem(x, ident=1):
        return html.Summary(html.A(id='tree-link', children=
                      html.Span('Mov-'+str(x.data.mid) + ' ({:3.2f}'.format(x.data.quality)+'%)' )
               ))
    
    def render_element(root, ident=1):
        if len(root.children) > 0:
            return html.Details([
                getTitleElem(root, ident),
                getMoveletBox(root.data, ident),
                dbc.ListGroup(
                    [dbc.ListGroupItem(render_element(x, ident+1)) for x in root.children]
                )]
            )
        else:
            return html.Div([getTitleElem(root, ident), getMoveletBox(root.data, ident),])
    
#     components = []
    
    if len(ls_movs) > 0:
        tree = createTree(ls_movs)
#         print(tree.traversePrint())

        return [html.Div(dbc.ListGroup([dbc.ListGroupItem(render_element(tree))])) ]
    
    return [html.Span('No movelets to render a tree')]