import sys, os 
import pandas as pd
import glob2 as glob
sys.path.insert(0, os.path.abspath(os.path.join('.')))

from datetime import datetime

import dash
from dash import dash_table
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State

from automatize.app_base import app
from automatize.assets.config import *
# ------------------------------------------------------------
# EXP_PATH='../../workdir/'

def render_page_experiments(pathname):
    content = []
    
    if pathname == '/experiments':
        content = render_experiments()
    else:
        content = render_method(pathname)
        
    return html.Div(children=[
        html.H3('Experimental Evaluations', style={'margin':10}),
        html.Div(id='output-experiments', children=content)
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
    Output(component_id='output-experiments', component_property='children'),
    Input('input-experiments-datasets', 'value'),
    Input('input-experiments-methods', 'value'),
    Input('input-experiments-classifiers', 'value'),
)
def render_experiments_call(sel_datasets=None, sel_methods=None, sel_classifiers=None):
    return render_experiments(sel_datasets, sel_methods, sel_classifiers)
    
def render_experiments(sel_datasets=None, sel_methods=None, sel_classifiers=None, path=EXP_PATH):
    hsf = os.path.join('automatize', 'assets', 'experiments_history.csv')
#     from automatize.results import history
#     df = history(path)
#     df.to_csv(hsf)

    time = datetime.fromtimestamp(os.path.getmtime(hsf))
    df = pd.read_csv(hsf,index_col=0)
    
    datasets    = list(df['dataset'].unique())
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

    f1 = df['dataset'].isin(sel_datasets)
    f2 = df['method'].isin(sel_methods)
    f3 = df['classifier'].isin(sel_classifiers)
    f4 = df['name'].isin(names)
    f5 = df['key'].isin(dskeys)
    df = df[f1 & f2 & f3 & f4 & f5]
    
    df.drop(['#','timestamp','file'], axis=1, inplace=True)
    
    return [
        html.Div([
            html.Div([
                html.Div([
                    html.Strong('Datasets: '),
                    dcc.Dropdown(
                        id='input-experiments-datasets',
                        options=[
                            {'label': x, 'value': x} for x in datasets
                        ],
                        multi=True,
                        value=sel_datasets,
                        style = {'width':'100%'},
                    ),
                    html.Strong('Classifiers: '),
                    dcc.Dropdown(
                        id='input-experiments-classifiers',
                        options=[
                            {'label': x, 'value': x} for x in classifiers
                        ],
                        multi=True,
                        value=sel_classifiers,
                        style = {'width':'100%'},
                    ),
                ], style={'width': '50%', 'padding': 10, 'flex': 1}),
                html.Div([
                    html.Strong('Methods: '),
                    dcc.Dropdown(
                        id='input-experiments-methods',
                        options=[
                            {'label': x, 'value': x} for x in methods
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
        html.Br(),
    ]  

def render_experiments_panels(df):
    return dcc.Tabs(id="experiments-tabs", value='tab-1', children=[
        dcc.Tab(label='Ranking Graph', value='tab-1', children=[render_expe_graph(df)]),
        dcc.Tab(label='Methods Rank', value='tab-2', children=[underDev('Methods Rank')]),
        dcc.Tab(label='Methods Results', value='tab-3', children=[render_expe_table(df)]),
    ])
    
def render_expe_graph(df): 
    return underDev('Ranking Graph')
    
def render_expe_table(df):
    return html.Div([
        dash_table.DataTable(
            id='table-experiments',
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
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
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
        )
    ], style={'margin':10})