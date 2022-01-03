import sys, os 
import pandas as pd
import glob2 as glob
sys.path.insert(0, os.path.abspath(os.path.join('.')))

import base64
import datetime
import io

import dash
from dash import dash_table
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State

from automatize.app_base import app
# ------------------------------------------------------------
DATA_PATH='../../datasets/data'

def render_page_datasets(pathname):
    if pathname == '/datasets':
        return html.Div(style = {'margin':10}, children=[
            html.H4('Datasets'), 
            html.Div(children=render_datasets()),
        ])
    else:
        return render_dataset(pathname)

def render_dataset(pathname):
    file = glob.glob(os.path.join(DATA_PATH, '*', '*', os.path.basename(pathname)+'.me'))[0]
    f = open(file, "r")
    return html.Div(dcc.Markdown(f.read()), style={'margin': '20px'})
        
def render_datasets(data_path=DATA_PATH):
    files = glob.glob(os.path.join(data_path, '*', '*', '*.me'))
    
    df = pd.DataFrame()
    
    for f in files:
        tmp = os.path.dirname(f).split(os.path.sep)
        aux = {}
        name = os.path.basename(f).split('.')[0]
        
        aux['Name'] = '<div class="dash-cell-value"><a href="/datasets/'+name+'" class="btn btn-link">'+name+'</a></div>'
        #html.A(name, href='/datasets/'+name) #'['+name+'](/datasets/'+name+')'
        
        aux['Category'] = getBadges(f, name)
        aux['File'] = f
#         print(aux)
        df = df.append(aux, ignore_index=True)
        
    return dash_table.DataTable(
        id='table-datasets',
#         columns=[{"name": i, "id": i, "presentation": "html"} for i in df.columns[:-1]],
        columns=[{
            'id': 'Name',
            'name': 'Dataset',
            'type': 'any',
            "presentation": "markdown",
        }, {
            'id': 'Category',
            'name': 'Category',
            'type': 'text',
            "presentation": "markdown",
#             'format': FormatTemplate.money(0)
        }],
        markdown_options={'link_target': '_self', 'html': True},
        data=df[df.columns[:-1]].to_dict('records'),
        css=[{'selector': 'td', 'rule': 'text-align: left !important;'},
             {'selector': 'th', 'rule': 'text-align: left !important; font-weight: bold'}
        ],
#             style_cell={
#                 'width': '{}%'.format(len(df_stats.columns)*2),
#                 'textOverflow': 'ellipsis',
#                 'overflow': 'hidden'
#             }
    )

def getBadges(file, name):
    colors={'multiple_trajectories':'bg-primary', 'raw_trajectories':'bg-success', 'semantic_trajectories':'bg-warning',
            'generic':'bg-danger', 'multivariate_ts':'bg-info', 'univariate_ts':'bg-secondary'}
    def translateBadge(s):
        return '<span class="badge rounded-pill '+colors[s]+'">'+s.split('_')[0].title()+'</span>'
    
    # Read the descriptors:
    badges = set()
    files = glob.glob(os.path.join(os.path.dirname(file), '..', 'descriptors', '*'+name+'*.json'))
    files = [os.path.basename(x).split('.')[0] for x in files]
    for x in files:
        if 'generic' in x:
            badges.add('generic')
        elif 'raw' in x:
            badges.add('raw_trajectories')
        elif 'semantic' in x:
            badges.add('semantic_trajectories')
        elif 'multivariate_ts' in x:
            badges.add('multivariate_ts')
        elif 'univariate_ts' in x:
            badges.add('univariate_ts')
            
    # Base Category
    tmp = os.path.dirname(file).split(os.path.sep)
    badges = list(badges)
    badges.sort()
    badges = [tmp[-2]] + badges
    
    return ' '.join([translateBadge(x) for x in badges])