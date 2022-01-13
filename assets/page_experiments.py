import sys, os 
import pandas as pd
import glob2 as glob
sys.path.insert(0, os.path.abspath(os.path.join('.')))

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
    if pathname == '/experiments':
        return html.Div(style = {'margin':10}, children=[
            html.H3('Experimental Evaluations'), 
            render_experiments(),
        ])
    else:
        return render_method(pathname)

def render_method(pathname):
    return html.H3('Under dev...')
#     file = glob.glob(os.path.join(DATA_PATH, '*', '*', os.path.basename(pathname)+'.md'))[0]
#     f = open(file, "r")
#     return html.Div(dcc.Markdown(f.read()), style={'margin': '20px'})

# def render_dataset(pathname):
#     file = glob.glob(os.path.join(DATA_PATH, '*', '*', os.path.basename(pathname)+'.md'))[0]
#     f = open(file, "r")
#     return html.Div(dcc.Markdown(f.read()), style={'margin': '20px'})
        

def render_experiments(path=EXP_PATH):
    hsf = os.path.join('automatize', 'assets', 'experiments_history.csv')
#     from automatize.results import history
#     df = history(path)
#     df.to_csv(hsf)

    df = pd.read_csv(hsf,index_col=0)
    return dash_table.DataTable(
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
    
