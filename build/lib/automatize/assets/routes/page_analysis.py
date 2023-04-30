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
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd

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

from automatize.assets.routes.subpage_trajectories import *
from automatize.assets.routes.subpage_models import *

from automatize.assets.app_base import app, sess, gess
from automatize.assets.config import PACKAGE_NAME

from automatize.preprocessing import readDataset, organizeFrame
# ------------------------------------------------------------
# from_trajs = 0
# to_trajs = 100
# sel_attributes = []
# sel_traj = ''
# # ------------------------------------------------------------
# ls_tids  = set()
# ls_trajs = []
# ls_movs  = []
# ------------------------------------------------------------

def reset():
#     global from_trajs, to_trajs, sel_attributes, sel_traj, ls_tids, ls_trajs, ls_movs
    sess('from_trajs', 0)
    sess('to_trajs', 50)
    sess('sel_attributes', [])
    sess('sel_traj', '')
    # ------------------------------------------------------------
    sess('ls_tids', [])
    sess('ls_trajs', [])
    sess('ls_movs', [])

# app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})

# app = dash.Dash()   #initialising dash app
# df = px.data.stocks() #reading stock price dataset 

def render_statistics(ls_tids, ls_trajs, ls_movs):
#     global ls_tids, ls_trajs, ls_movs
    # ------------------------------------------------------------
    ls_tids        = set(gess('ls_tids', []))
    ls_trajs       = gess('ls_trajs', [])
    ls_movs        = gess('ls_movs', [])
    
    # Update Screen:
    components = []
    components.append(html.Hr())
    if len(ls_movs) > 0:
#         used_features, df_stats = movelets_statistics(ls_movs)
#         df_stats = movelets_statistics_bylabel(used_features, df_stats)
        df_stats = movelets_statistics(ls_movs)
        df_stats = movelets_statistics_bylabel(df_stats)
        
#        components.append(html.Div(style = {'margin':10, 'display':'grid'}, children = [
#            html.Div(style = {'display':'inline'}, children = [
#                html.Strong('Number of Movelets: '),
#                html.Span(str(len(ls_movs))),
#            ]),
#        ]))
#        components.append(dash_table.DataTable(
#            id='table-movs-stats',
#            columns=[{"name": i, "id": i} for i in df_stats.columns],
#            data=df_stats.to_dict('records'),
##             css=[{'selector': 'table', 'rule': 'table-layout: fixed; width: 50%'}],
##             style_cell={
##                 'width': '{}%'.format(len(df_stats.columns)*2),
##                 'textOverflow': 'ellipsis',
##                 'overflow': 'hidden'
##             }
#        ))
        components.append(html.Details(style = {'margin':10, 'display':'grid'}, open=True, children = 
            [
                html.Summary(
                    html.A(children=[
                        html.Strong('Total Number of Movelets: '),
                        html.Span(str(len(ls_movs))),
                    ]),
                ),
                html.Div(list(map(lambda d: 
                     html.Details(open=True, children = [
                        html.Summary([
                            html.Strong('Label: '),
                            html.Span(str(d['label'])),
                        ]),
                        html.Div(list(map(lambda x:
                            html.Div([
                                html.Strong(x[0]+': '),
                                html.Span(str(x[1])),
                            ])
                        , list(d.items())[1:])),
                        style={'text-indent':'4em'})
                     ]), df_stats.to_dict('records'))),
                    style={'text-indent':'2em'}
                )
            ]
        ))
        components.append(html.Hr())
    
    if len(ls_trajs) > 0:
        #labels, samples, top, bot, npoints, avg_size, diff_size, attr, num_attr, classes, stats = trajectory_statistics(ls_trajs)
        labels, samples, top, bot, npoints, avg_size, diff_size, attr, num_attr, classes = trajectory_statistics(ls_trajs)
        
        ds_stats = html.Div(style = {'margin':10}, children = [
            dbc.Accordion(
                [
                    dbc.AccordionItem(
                        [
                            html.Div(style = {'margin':10, 'display':'grid'}, children = [
                                html.Div(style = {'display':'inline'}, children = [
                                    html.Strong('Number os Trajectories: '),
                                    html.Span(str(samples)),
                    #                 html.Br(),
                    #                 html.Span(', '.join(['Class '+str(k)+': '+str(v) for k,v in classes.items()])),
                                ]),
                                html.Div(style = {'display':'inline'}, children = [
                                    html.Strong('Attributes: '),
                                    html.Span(str(num_attr)),
                                    html.Br(),
                                    html.Span('[' + (', '.join(attr)) + ']'),
                                ]),
                                html.Div(style = {'display':'inline'}, children = [
                                    html.Strong('Trajectories Size: '),
                                    html.Span(str(avg_size) + ' | from ' + str(bot) + ' to ' + str(top) + ' | ±' + str(diff_size)),
                                ]),
                                html.Div(style = {'display':'inline'}, children = [
                                    html.Strong('Number of Points: '),
                                    html.Span(str(npoints)),
                                ]),
                            ])
                        ],
                        title="Dataset Statistics",
                    ),
                    dbc.AccordionItem(
                        [
                            html.Div(style = {'display':'inline'}, children = [
                                #html.Strong('Classes: '),
                                #html.Span(str(len(labels))),
                                #html.Br(),
                                html.Span(', '.join(labels)),
                            ]),
                        ],
                        title="Classes",
                    ),
                    dbc.AccordionItem(
                        [
                            html.Div(style = {'display':'inline'}, children = [
                                #html.Br(),
                                #html.Strong('Trajectories per Class: '),
                                #html.Br(),
                                html.Span(', '.join(['C('+str(k)+'): '+str(v) for k,v in classes.items()])),
                            ])
                        ],
                        title="Trajectories per Class",
                    ),
                ],
            )]
        )
        components.append(ds_stats)
        components.append(html.Hr())
        
    return components

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    # DECODE DATAFRAME:
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename or '.zip' in filename or '.ts' in filename:

            df = pd.DataFrame()
            if '.csv' in filename:
                decoded = io.StringIO(decoded.decode('utf-8'))
                df = pd.read_csv(decoded, na_values='?')
            elif '.zip' in filename:
                from automatise.preprocessing import read_zip
                from zipfile import ZipFile
                decoded = io.BytesIO(decoded)
                df = read_zip(ZipFile(decoded, "r"))
            elif '.ts' in filename:
                from automatise.io.ts_io import load_from_tsfile
                decoded = io.StringIO(decoded.decode('utf-8'))
                df = load_from_tsfile(decoded, replace_missing_vals_with="?")

            df.columns = df.columns.astype(str)

            df, columns_order_zip, columns_order_csv = organizeFrame(df)
            update_trajectories(df[columns_order_csv])
        elif 'json' in filename:
            update_movelets(io.BytesIO(decoded))
        else:
            return dbc.Alert("This file format is not accepted.", color="warning", style = {'margin':10})
    except Exception as e:
        raise e
        print(e)
        return dbc.Alert("There was an error processing this file.", color="danger", style = {'margin':10})

    return dbc.Alert("File "+filename+" loaded ("+str(datetime.datetime.fromtimestamp(date))+").", color="info", style = {'margin':10})

def update_trajectories(df):
    # TRANSFORM TRAJECTORIES:
#     global ls_tids, ls_trajs
    # ------------------------------------------------------------
    ls_tids        = set(gess('ls_tids', []))
    ls_trajs       = gess('ls_trajs', [])
    
    ls_aux = parse_trajectories(df)
    #for T in ls_aux:
    def processT(T):
        nonlocal ls_tids, ls_trajs
        if T.tid not in ls_tids:
            ls_tids.add(T.tid)
            ls_trajs.append(T)
            
    list(map(lambda T: processT(T), ls_aux))
            
    sess('ls_tids', list(ls_tids))
    sess('ls_trajs', ls_trajs)
            
def update_movelets(data):
    # TRANSFORM Movelets:
#     global ls_movs
    ls_movs       = gess('ls_movs', [])
    
    ls_aux = parse_movelets(data)
    #for m in ls_aux:
    #    ls_movs.append(m)
    
    ls_movs = ls_movs + ls_aux 
        
    sess('ls_movs', ls_movs)

# ------------------------------------------------------------
# @du.callback(
#     output=Output('output-data-upload', 'children'),
#     id='upload-data',
# )
# def get_a_list(filenames):
#     return html.Ul([html.Li(filenames)])


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              State('output-data-upload', 'children'),
)
def update_statistic(list_of_contents, list_of_names, list_of_dates, components):
    ls_tids        = set(gess('ls_tids', []))
    ls_trajs       = gess('ls_trajs', [])
    ls_movs        = gess('ls_movs', [])
    
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        
#         return html.Div(children + render_statistics(ls_tids, ls_trajs, ls_movs))
        return html.Div(render_statistics(ls_tids, ls_trajs, ls_movs))
    
@app.callback(
    Output(component_id='output-data-trajs', component_property='children'),
    Input(component_id='output-data-upload', component_property='children'),
#     Input('input-from-traj', 'value'),
#     Input('input-to-traj', 'value'),
    Input('input-range-trajs', 'value'),
    Input('input-attr-traj', 'value'),
)
def update_trajs_list(input_value, range_value, sel_attributes):
    ls_trajs       = gess('ls_trajs', [])
    ls_movs        = gess('ls_movs', [])
    return render_page_trajectories(ls_trajs, range_value, ls_movs, sel_attributes)


    
@app.callback(
    Output(component_id='output-traj-movelet', component_property='children'),
    Input('input-traj', 'value'),
)
def update_traj_view(input_value):
#     global sel_traj, ls_movs
    ls_trajs       = gess('ls_trajs', [])
    ls_movs        = gess('ls_movs', [])
#     sel_traj       = gess('sel_traj', '')
    
    traj = None #ls_trajs[0] if len(ls_trajs) > 0 else None
    if input_value != '':
        for T in ls_trajs:
            if str(input_value) == str(T.tid):
                traj = T
                break 
    if traj:
        return render_page_movelets(traj, ls_movs)
    else:
        return dbc.Alert("Select a valid TID.", color="info", style = {'margin':10})


@app.callback(Output('output-graph-mov', 'children'),
              Input('upload-data', 'contents'),
              Input('input-range-mov-graph', 'value'),
              Input('input-attr-mov-graph', 'value'),
              Input('input-format-mov-graph', 'value'),
#               Input('input-from-mov-graph', 'value'),
#               Input('input-to-mov-graph', 'value'),
)
def update_mov_view(list_of_contents, range_value, sel_attribute, model):
    
    from_trajs     = gess('from_trajs', 0)
    to_trajs       = gess('to_trajs', 100)
    
#     ls_tids        = gess('ls_tids', set())
    ls_trajs       = gess('ls_trajs', [])
    ls_movs        = gess('ls_movs', [])
    
    if len(ls_movs) > 0:
        from_trajs, to_trajs = range_value
        from_trajs = int(from_trajs) if from_trajs != '' else 0
        to_trajs = int(to_trajs) if to_trajs != '' else 10
        
        sess('from_trajs', from_trajs)
        sess('to_trajs', to_trajs)

        if len(ls_trajs) > 0:
            attributes = ls_trajs[0].attributes
        else:
            attributes = list(set([x for m in ls_movs for x in m.attributes()]))
            attributes.sort()

#         if sel_attribute == '':
#             sel_attribute = None

        return html.Div(style = {'width':'100%'}, 
            children = render_model(ls_movs, model, from_trajs, to_trajs, attributes, sel_attribute))
#     except Exception as e:
#         print(e)
    
    return render_model(ls_movs, '', 0, 100, '')


def render_page_analysis():
    reset()
    return dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Movelets Statistics', value='tab-1', children=[render_content('tab-1')]),
        dcc.Tab(label='Trajectories and Movelets', value='tab-2', children=[render_content('tab-2')]),
        dcc.Tab(label='Movelets', value='tab-3', children=[render_content('tab-3')]),
        dcc.Tab(label='Movelets Graph', value='tab-4', children=[render_content('tab-4')]),
#             dcc.Tab(label='Movelets Sankey', value='tab-5', children=[render_content('tab-5')]),
#             dcc.Tab(label='Movelets Tree', value='tab-6', children=[render_content('tab-6')]),
    ]),

# @app.callback(Output('tabs-content', 'children'),
#               Input('tabs', 'value'))
def render_content(tab):
    
    from_trajs     = gess('from_trajs', 0)
    to_trajs       = gess('to_trajs', 100)
#     sel_attributes = sess('sel_attributes', [])
    sel_traj       = gess('sel_traj', '')    
    
    if tab == 'tab-1':
        return html.Div([
#             html.H4('Tab content', style = {'margin':10}),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
    #                 'width': '90%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '20px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
#             du.Upload(
#                 id='upload-data',
#                 filetypes=['csv', 'json', 'zip'],
#             ),
            html.Div(id='output-data-upload'),
        ])
    elif tab == 'tab-2':
        return html.Div(style = {'margin':10}, children=[
            html.H4('Trajectories and Movelets Visualization'), 
#             html.Div(style = {'display':'inline'}, children = [
                
#             ]),
            html.Div(id='output-data-trajs', children=[
                page_trajectories_filter([], from_trajs, to_trajs, [], []),
#                 html.H6('Attributes: '),
#                 dcc.Dropdown(
#                     id='input-attr-traj',
#                     options=[
# #                         {'label': 'None', 'value': 'None'}
#                     ],
#                     multi=True,
#                     value=""
#                 ),
            ]),
        ])
    elif tab == 'tab-3':
        return html.Div(style = {'margin':10}, children=[
            html.H4('Movelets Visualization'), 
            html.Div(style = {'display':'inline'}, children = [
                html.Strong('Trajectory ID: '),
                dcc.Input(
                    id='input-traj',
                    placeholder='TID...',
                    type='text',
                    value=str(sel_traj)
                ),
            ]),
            html.Div(id='output-traj-movelet'),
        ])
    elif tab == 'tab-4':
        return html.Div(style = {'margin':10}, children=[
            html.H4('Movelets Graph Visualization'), 
            html.Div(id='output-graph-mov', children=render_model_filter()),
        ])
#     elif tab == 'tab-5':
#         return html.Div(style = {'margin':10}, children=[
# #             html.H4('Trajectories and Movelets Visualization'), 
#             html.Div(id='output-mov-sankey', children=render_model('output-mov-sankey', [])),
#         ])
#     elif tab == 'tab-6':
#         return html.Div(style = {'margin':10}, children=[
#             html.H4('Movelets Tree View'), 
#             html.Div(id='output-mov-tree', children=render_tree([])),
#         ])
    else:
        return html.Div([
            dbc.Alert("Content in development.", color="info", style = {'margin':10})
        ])