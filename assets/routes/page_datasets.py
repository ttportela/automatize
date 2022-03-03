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
from automatize.assets.config import *
# from automatize.assets.helper.script_inc import DATASET_TYPES
from automatize.assets.helper.datasets_inc import *
# ------------------------------------------------------------
# DATA_PATH='../../datasets'

def render_page_datasets(pathname):
    if pathname == '/datasets':
        return html.Div(style = {'margin':10}, children=[
#             html.H3('Datasets'), 
            render_markdown_file(PAGES_ROUTE+'/pages/datasets.md'),
            html.Div(children=render_datasets()),
        ])
    else:
        return render_dataset(pathname)

def render_dataset(pathname):
    components = []
    
    file = glob.glob(os.path.join(DATA_PATH, '*', '*', os.path.basename(pathname)+'.md'))[0]
    with open(file, "r") as f:
        components.append(dcc.Markdown(f.read()))
        
    file = glob.glob(os.path.join(DATA_PATH, '*', '*', os.path.basename(pathname)+'-stats.md'))
    if len(file) > 0 and os.path.exists(file[0]):
        with open(file[0], "r") as f:
            components.append(html.Br())
            components.append(html.Hr())
            components.append(html.H6('Dataset Statistics:'))
            components.append(dcc.Markdown(f.read()))
    
    components.append(html.H6('Best Result:'))
    components.append(html.Br())
    components.append(html.Hr())
    components.append(html.H6('Related Publications:'))
    components.append(html.Br())
    components.append(html.Hr())
    components.append(html.H6('Download Files:'))
    components.append(html.Br())
    
    return html.Div(components, style={'margin': '20px'})
        
def render_datasets(data_path=DATA_PATH):
    lsds = list_datasets_dict(data_path)
    components = []
    
    for category, dslist in lsds.items():
#         for dataset, subsets in dslist.items():
        components.append(html.Br())
        components.append(html.H4(DATASET_TYPES[category] + ':'))
        components.append(render_datasets_category(category, dslist, data_path))
        components.append(html.Hr())
    
#     for category, name in DATASET_TYPES.items():
#         components.append(html.Br())
#         components.append(html.H4(name + ':'))
#         components.append(render_datasets_category(category, data_path))
#         components.append(html.Hr())
        
    return html.Div(components)
    
def render_datasets_category(category, dsdict, data_path=DATA_PATH):
    
    df = pd.DataFrame()
    for dataset, subsets in dsdict.items():
        aux = {}
        aux['Name'] = '<div class="dash-cell-value"><a href="/datasets/'+dataset+'" class="btn btn-link">'+dataset+'</a></div>'
        aux['Category'] = getBadges(category, dataset, subsets)
        aux['File'] = os.path.join(data_path, category, dataset, dataset+'.md')
        df = df.append(aux, ignore_index=True)
    
    
#     files = glob.glob(os.path.join(data_path, category, '*', '*.md'))
    
#     df = pd.DataFrame()
    
#     for f in files:
#         tmp = os.path.dirname(f).split(os.path.sep)
#         aux = {}
#         name = os.path.basename(f).split('.')[0]
        
#         aux['Name'] = '<div class="dash-cell-value"><a href="/datasets/'+name+'" class="btn btn-link">'+name+'</a></div>'
#         #html.A(name, href='/datasets/'+name) #'['+name+'](/datasets/'+name+')'
        
#         aux['Category'] = getBadges(f, name)
#         aux['File'] = f
# #         print(aux)
#         df = df.append(aux, ignore_index=True)
        
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

def render_datasets_all(data_path=DATA_PATH):
    files = glob.glob(os.path.join(data_path, '*', '*', '*.md'))
    
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

def getBadges(category, dataset, subsets):
#     colors={'multiple_trajectories':'bg-primary', 'raw_trajectories':'bg-success', 'semantic_trajectories':'bg-warning',
#             'generic':'bg-danger', 'multivariate_ts':'bg-info', 'univariate_ts':'bg-secondary'}
    def toBadge(cat, s):
        return '<span class="badge rounded-pill dataset-color-'+cat+'">'+ s +'</span>'
    
    # Read the descriptors:
#     subsets.sort()
    badges = [toBadge((category if x == 'specific' else x), translateCategory(dataset, category, x)) for x in subsets]

#     files = glob.glob(os.path.join(os.path.dirname(file), '..', 'descriptors', '*'+name+'*.json'))
#     files = [os.path.basename(x).split('.')[0] for x in files]
#     for x in files:
#         if 'generic' in x:
#             badges.add('generic')
#         elif 'raw' in x:
#             badges.add('raw_trajectories')
#         elif 'semantic' in x:
#             badges.add('semantic_trajectories')
#         elif 'multivariate_ts' in x:
#             badges.add('multivariate_ts')
#         elif 'univariate_ts' in x:
#             badges.add('univariate_ts')
            
    # Base Category
#     tmp = os.path.dirname(file).split(os.path.sep)
#     badges = list(badges)
#     badges.sort()
#     badges = [tmp[-2]] + badges
#     badges = [translateCategory(dataset, category)] + badges
    
    return ' '.join([x for x in badges])