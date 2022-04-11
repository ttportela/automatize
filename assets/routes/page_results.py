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
import glob2 as glob
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join('.')))

import io
import base64
from datetime import datetime, date

import dash
from dash import dash_table
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State

from automatize.helper.CDDiagram import draw_cd_diagram 
from automatize.results import format_hour, format_float

from automatize.assets.app_base import app, gess, sess
from automatize.assets.config import *
from automatize.helper.script_inc import METHODS_NAMES, CLASSIFIERS_NAMES
# ------------------------------------------------------------
# EXP_PATH='../../workdir/'

# DATA = None
# ------------------------------------------------------------

def render_page_results(pathname):
    content = []
    
#     if pathname == '/results':
#     global DATA
#     DATA = None
    sess('DATA', None)

    content = render_experiments()
#     else:
#         content = render_method(pathname)
        
    return html.Div(children=[
#         html.H3('Experimental Evaluations', style={'margin':10}),
        render_markdown_file(PAGES_ROUTE+'/pages/results.md', div=True),
        html.Div(id='output-results', children=content)
    ])

def render_method(pathname):
    return [underDev(pathname)]
#     file = glob.glob(os.path.join(DATA_PATH, '*', '*', os.path.basename(pathname)+'.md'))[0]
#     f = open(file, "r")
#     return html.Div(dcc.Markdown(f.read()), style={'margin': '20px'})

# def render_dataset(pathname):
#     file = glob.glob(os.path.join(DATA_PATH, '*', '*', os.path.basename(pathname)+'.md'))[0]
#     f = open(file, "r")
#     return html.Div(dcc.Markdown(f.read()), style={'margin': '20px'})
    
@app.callback(
    Output(component_id='output-results', component_property='children'),
    Input('input-results-datasets', 'value'),
    Input('input-results-methods', 'value'),
    Input('input-results-classifiers', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
)
def render_experiments_call(sel_datasets=None, sel_methods=None, sel_classifiers=None, 
                            contents=None, filename=None, fdate=None):
    if contents is not None:
#         global DATA
        DATA = gess('DATA')
#         print(filename, date)
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
#                 df = pd.DataFrame()
                decoded = io.StringIO(decoded.decode('utf-8'))
                DATA = [pd.read_csv(decoded), filename, datetime.fromtimestamp(fdate)]
                sess('DATA', DATA)
                return render_experiments(None, None, None)
            else:
                return [dbc.Alert("File format invalid (use CSV exported with 'automatize.results.history').", color="danger", style = {'margin':10})] + \
                    render_experiments(None, None, None)
        except Exception as e:
#             raise e
            print(e)
            return [dbc.Alert("There was an error processing this file.", color="danger", style = {'margin':10})] + \
                    render_experiments(None, None, None)
    
    return render_experiments(sel_datasets, sel_methods, sel_classifiers)

def filter_results(sel_datasets=None, sel_methods=None, sel_classifiers=None, file=RESULTS_FILE):
    #     hsf = os.path.join('automatize', 'assets', 'experiments_history.csv')
#     from automatize.results import history
#     df = history(path)
#     df.to_csv(hsf)
#     global DATA
    DATA = gess('DATA')

    time = date.today()
    if DATA:
        df = DATA[0].copy()
        time = DATA[2]
    else:
        df = pd.read_csv(file, index_col=0)
        time = datetime.fromtimestamp(os.path.getmtime(file))
    
    df['set'] = df['dataset'] #+ '-' + df['subset']
    
#     df['accuracy'] = df['accuracy'] * 100
    
    datasets    = list(df['set'].unique())
    methods     = list(df['method'].unique())
    classifiers = list(df['classifier'].unique())
    names       = list(df['name'].unique())
    dskeys      = list(df['key'].unique())
    
    if sel_datasets == None or sel_datasets == []:
        sel_datasets = datasets
    if sel_methods == None or sel_methods == []:
        sel_methods = methods
    if sel_classifiers == None or sel_classifiers == []:
        sel_classifiers = classifiers

    f1 = df['set'].isin(sel_datasets)
    f2 = df['method'].isin(sel_methods)
    f3 = df['classifier'].isin(sel_classifiers)
    f4 = df['name'].isin(names)
    f5 = df['key'].isin(dskeys)
    df = df[f1 & f2 & f3 & f4 & f5]
                   
    return df, DATA, time, datasets, methods, classifiers, names, dskeys, sel_datasets, sel_methods, sel_classifiers
  
def render_experiments(sel_datasets=None, sel_methods=None, sel_classifiers=None, file=RESULTS_FILE):

    df, DATA, time, datasets, methods, classifiers, names, dskeys, sel_datasets, sel_methods, sel_classifiers = filter_results(sel_datasets, sel_methods, sel_classifiers, file)
    
    return [
        html.Div([
            html.Div([
                html.Div([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Visualizing '+DATA[1] if DATA else 'Provide results in CSV file',
                            html.A(' (Drag and Drop or Select Files)')
                        ]),
                        style={
            #                 'width': '90%',
                            'height': '50px',
                            'lineHeight': '50px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '20px'
                        },
                        # Allow multiple files to be uploaded
                        multiple=False
                    ),
                    html.Strong('Datasets: '),
                    dcc.Dropdown(
                        id='input-results-datasets',
                        options=[
                            {'label': x, 'value': x} for x in datasets
                        ],
                        multi=True,
                        value=sel_datasets,
                        style = {'width':'100%'},
                    ),
                ], style={'width': '50%', 'padding': 10, 'flex': 1}),
                html.Div([
                    html.Strong('Classifiers: '),
                    dcc.Dropdown(
                        id='input-results-classifiers',
                        options=[
                            {'label': CLASSIFIERS_NAMES[x] if x in CLASSIFIERS_NAMES.keys() else x, 
                             'value': x} for x in classifiers
                        ],
                        multi=True,
                        value=sel_classifiers,
                        style = {'width':'100%'},
                    ),
                    html.Strong('Methods: '),
                    dcc.Dropdown(
                        id='input-results-methods',
                        options=[
                            {'label': METHODS_NAMES[x] if x in METHODS_NAMES.keys() else x, 
                             'value': x} for x in methods
                        ],
                        multi=True,
                        value=sel_methods,
                        style = {'width':'100%'},
                    ),
                ], style={'width': '50%', 'padding': 10, 'flex': 1}),
            ], style={'display': 'flex', 'flex-direction': 'row'}),
        ], style={'margin':10}),
        html.Hr(),
        render_experiments_panels(df),
        html.Br(),
        html.Span("Last Update: " + time.strftime("%d/%m/%Y, %H:%M:%S"), style={'margin':10}),
        dbc.Button("download results", id="download-results-btn", color="light"),
#         html.A('[download results]', href=file, id='download-results-btn'),
        dcc.Download(id="download-results-csv"),
        html.Br(), html.Br(),
    ]  

@app.callback(
    Output("download-results-csv", "data"),
    Input("download-results-btn", "n_clicks"),
    State('input-results-datasets', 'value'),
    State('input-results-methods', 'value'),
    State('input-results-classifiers', 'value'),
    prevent_initial_call=True,
)
def download_results_csv(n_clicks, sel_datasets=None, sel_methods=None, sel_classifiers=None):
    df, *x = filter_results(sel_datasets, sel_methods, sel_classifiers, RESULTS_FILE)
    return dcc.send_data_frame(df.to_csv, "automatize_experimental_history.csv")

def render_experiments_panels(df):
    return dcc.Tabs(id="results-tabs", value='tab-1', children=[
        dcc.Tab(label='Critical Difference', value='tab-1', children=[render_expe_graph(df.copy())]),
        dcc.Tab(label='Average Ranking', value='tab-2', children=[render_ranks(df.copy())]),
        dcc.Tab(label='Raw Results', value='tab-3', children=[render_expe_table(df.copy())]),
    ])
    
def render_expe_graph(df):     
    components = []
    
    try:
        fig = draw_cd_diagram(df, 'name', 'key', 'accuracy', title='Accuracy', labels=True)
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches='tight', format = "png") # save to the above file object
        fig.close()
        fig = base64.b64encode(buf.getbuffer()).decode("utf8")
        components.append(html.Div(html.Img(src="data:image/png;base64,{}".format(fig)), style={'padding':10}))
    except Exception as e:
        print('Accuracy', 'results not possible:', str(e))
        components.append(alert('Accuracy Graph not possible with these parameters.'))
        
    try:
        fig = draw_cd_diagram(df[~df['method'].isin(['MARC', 'POI', 'NPOI', 'WPOI'])], 'name', 'key', 'cls_runtime', title='Classification Time', labels=True, ascending=False)
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches='tight', format = "png") # save to the above file object
        fig.close()
        fig = base64.b64encode(buf.getbuffer()).decode("utf8")
        components.append(html.Div(html.Img(src="data:image/png;base64,{}".format(fig)), style={'padding':10}))
    except Exception as e:
        print('Classification Time', 'results not possible:', str(e))
        components.append(alert('Classification Time Graph not possible with these parameters.'))
        
    try:
        fig = draw_cd_diagram(df, 'name', 'key', 'total_time', title='Total Time', labels=True, ascending=False)
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches='tight', format = "png") # save to the above file object
        fig.close()
        fig = base64.b64encode(buf.getbuffer()).decode("utf8")
        components.append(html.Div(html.Img(src="data:image/png;base64,{}".format(fig)), style={'padding':10}))
    except Exception as e:
        print('Total Time', 'results not possible:', str(e))
        components.append(alert('Total Time Graph not possible with these parameters.'))
        
    return html.Div(components + [
        html.Br(),
        html.Span("* Some methods may not appear due to different number of mesurements between methods and datasets.", style={'margin':10}),
        html.Br(),
        html.Span("** Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm to reject the null\'s hypothesis.", style={'margin':10}),
        html.Br(),
        html.Span("*** At least 3 sets of measurements must be given for Friedman test.", style={'margin':10}),
        html.Br(),
        html.Span("Critical Difference Diagram adapted from:", style={'margin':10}),
        html.A("hfawaz/cd-diagram", href='https://github.com/hfawaz/cd-diagram'),
        html.Br(),
        ])
def render_ranks(df): 
    return html.Div([
        html.H4('Accuracy Ranks:'),
        render_avg_rank(df, rank_col='accuracy', ascending=False),
        html.Hr(), html.Br(),
        html.H4('Total Time Ranks:'),
        render_avg_rank(df, rank_col='total_time', ascending=True, format_func=format_hour),
        html.Hr(), html.Br(),
        html.H4('Classification Time Ranks:'),
        render_avg_rank(df[df['cls_runtime'] > 0], rank_col='cls_runtime', ascending=True, format_func=format_hour),
        html.Br(),
    ], style={'margin':10})

def render_avg_rank(df, rank_col='accuracy', ascending=False, format_func=format_float): 
    cls_name = 'method'
    ds_key = 'key'
    
    components = [html.Br()]
    
    for dataset in df['dataset'].unique():
        dfx = df[df['dataset'] == dataset]
        dfx['rank'] = dfx[rank_col].rank(ascending=ascending)
#         print(1, dfx)
        dfx = dfx.groupby([cls_name, 'classifier']).mean(['rank'])
#         print(2, dfx)
        dfx = dfx.sort_values(['rank']).reset_index()
    #     print(3, dfx)

        rankItems = []
        for i in range(len(dfx)):
            rankItems.append(dbc.ListGroupItem(
                dbc.Row(
                    [
                        dbc.Col(dbc.Badge(str(i+1)+'º', color="light", text_color="primary", className="ms-1")),
                        dbc.Col(html.Span( METHODS_NAMES[dfx[cls_name][i]] 
                                          if dfx[cls_name][i] in METHODS_NAMES.keys() else dfx[cls_name][i] )),
                        dbc.Col(html.Span( CLASSIFIERS_NAMES[dfx['classifier'][i]] 
                                          if dfx['classifier'][i] in CLASSIFIERS_NAMES.keys() else dfx['classifier'][i] )),
                        dbc.Col(html.Span( format_func(dfx[rank_col][i]) )),
                        dbc.Col(html.Span(str(dfx['rank'][i]) + ' (Avg Rank)')),
                    ]
                ),
#                 [
# #                 html.Span(str(i+1)+'º'),
                
#                 dbc.Badge(str(i+1)+'º', color="light", text_color="primary", className="ms-1"),
#                 html.Span(dfx['rank'][i]),
#                 html.Span(dfx[cls_name][i]),
#                 html.Span(dfx['classifier'][i]),
#                 html.Span(dfx[rank_col][i]),
#             ], 
                color="info" if i==0 else "light", style={'width': 'auto'}))
#             print(i+1, dfx['rank'][i], dfx[cls_name][i], dfx['classifier'][i], dfx[rank_col][i])
        
#         components.append(html.Br())
        components.append(html.H6(dataset+':'))
        components.append(dbc.ListGroup(rankItems))
#         components.append(html.Hr()) 
        components.append(html.Br())
    
    return html.Div(components)#, style={'margin':10})

def render_expe_table(df):
    
    dfx = df.drop(['#','timestamp','file','random','set','error','name','key'], axis=1)
    
    dfx['method'] = [METHODS_NAMES[x] if x in METHODS_NAMES.keys() else x for x in dfx['method']]
    dfx['runtime'] = [format_hour(x) for x in dfx['runtime']]
    dfx['cls_runtime'] = [format_hour(x) for x in dfx['cls_runtime']]
    dfx['total_time'] = [format_hour(x) for x in dfx['total_time']]
    
    return html.Div([
        dash_table.DataTable(
            id='table-results',
    #         columns=[{"name": i, "id": i, "presentation": "html"} for i in df.columns[:-1]],
    #         columns=[{
    #             'id': 'Name',
    #             'name': 'Dataset',
    #             'type': 'any',
    #             "presentation": "markdown",
    #         }, {
    #             'id': 'Category',
    #             'name': 'Category',
    #             'type': 'text',
    #             "presentation": "markdown",
    # #             'format': FormatTemplate.money(0)
    #         }],
            columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in dfx.columns],
            data=dfx.to_dict('records'),
    #         markdown_options={'link_target': '_self', 'html': True},
    #         data=df[df.columns[:-1]].to_dict('records'),
    #         css=[{'selector': 'td', 'rule': 'text-align: left !important;'},
    #              {'selector': 'th', 'rule': 'text-align: left !important; font-weight: bold'}
    #         ],
    #             style_cell={
    #                 'width': '{}%'.format(len(df_stats.columns)*2),
    #                 'textOverflow': 'ellipsis',
    #                 'overflow': 'hidden'
    #             }
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
        )
    ], style={'margin':10})