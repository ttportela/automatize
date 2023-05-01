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
import pandas as pd
import glob2 as glob

import base64
import datetime
import io

import dash
from dash import dash_table
from dash import dcc
from dash import html, ALL
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State

from automatize.assets.app_base import app
from automatize.assets.config import *

from automatize.results import format_hour
from automatize.inc.datasets_def import *
from automatize.inc.script_def import METHODS_NAMES, CLASSIFIERS_NAMES
# ------------------------------------------------------------
# DATA_PATH='../../datasets'

def render_page_datasets(pathname):
    if pathname == '/datasets':
        return html.Div(style = {'margin':10}, children=[
#             html.H3('Datasets'), 
            render_markdown_file(PAGES_ROUTE+'assets/pages/datasets.md'),
            html.Div(children=render_datasets()),
        ])
    else:
        if os.environ.get('AUTOMATIZE_DATA_PATH') is not None:
            data_path = os.environ['AUTOMATIZE_DATA_PATH']
        else:
            data_path = DATA_PATH
        # ------------------------------------------------------------
        return render_dataset(pathname, data_path)


def render_datasets(data_path=DATA_PATH):
    if os.environ.get('AUTOMATIZE_DATA_PATH') is not None:
        data_path = os.environ['AUTOMATIZE_DATA_PATH']
    # ------------------------------------------------------------
    lsds = list_datasets_dict(data_path)
    components = []
    
    for category, dslist in lsds.items():
#         for dataset, subsets in dslist.items():
        components.append(html.Br())
        components.append(html.H4(DATASET_TYPES[category] + ':'))
        components.append(render_datasets_category(category, dslist, data_path))
        components.append(html.Hr())
        
    return html.Div(components)
    
def render_datasets_category(category, dsdict, data_path=DATA_PATH):    
    dsdict = dict(sorted(dsdict.items()))
    
    if len(dsdict) > 30:
        return render_datasets_category_ls(category, dsdict, data_path)
    else:
        return render_datasets_category_tb(category, dsdict, data_path)
    
def render_datasets_category_tb(category, dsdict, data_path=DATA_PATH):
    
    df = pd.DataFrame()
    for dataset, subsets in dsdict.items():
        aux = {}
        aux['Name'] = '<div class="dash-cell-value"><a href="/datasets/'+category+'/'+dataset+'" class="btn btn-link">'+dataset+'</a></div>'
        aux['Category'] = getBadges(category, dataset, subsets)
        aux['File'] = os.path.join(data_path, category, dataset, dataset+'.md')
        df = df.append(aux, ignore_index=True)
        
    return dash_table.DataTable(
        id='table-datasets',
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
        }],
        markdown_options={'link_target': '_self', 'html': True},
        data=df[df.columns[:-1]].to_dict('records'),
        css=[{'selector': 'td', 'rule': 'text-align: left !important;'},
             {'selector': 'th', 'rule': 'text-align: left !important; font-weight: bold'}
        ],
    )
    
def render_datasets_category_ls(category, dsdict, data_path=DATA_PATH):    
    return html.Div([
        html.Div(html.A(dataset,
                        href='/datasets/'+category+'/'+dataset, 
                        className="btn btn-link"), 
                 style={'display': 'inline-table'})
        for dataset, subsets in dsdict.items()
    ])

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

# ------------------------------------------------------------
def render_dataset(pathname, data_path=DATA_PATH):
    components = []    
    if len(pathname.split(os.path.sep)) < 4:
        return underDev(pathname)
    
    category, dataset = pathname.split(os.path.sep)[2:4]
    file = os.path.join(data_path, category, dataset, dataset+'.md')
    if not os.path.isfile(file):
        return underDev(pathname)
    
    file = glob.glob(os.path.join(data_path, category, dataset, dataset+'.md'))[0]
    with open(file, "r") as f:
        components.append(html.H3(dataset))
        components.append(dcc.Markdown(f.read(), className='markdown'))
        
    file = glob.glob(os.path.join(data_path, category, dataset, dataset+'-stats.md'))
    if len(file) > 0 and os.path.exists(file[0]):
        with open(file[0], "r") as f:
            components.append(html.Br())
            components.append(html.Hr())
#             components.append(html.H4('Dataset Statistics:'))
            components.append(dcc.Markdown(f.read(), className='markdown'))
    
    components.append(html.Br())
    components.append(html.Hr())
    components.append(html.H6('Best Result:'))
#     components.append(html.Br())
    components.append(render_results(dataset))
    components.append(html.Br())
    components.append(html.Hr())
    components.append(html.H6('Related Publications:'))
#     components.append(html.Br())
    components.append(render_related_publications(dataset))
    components.append(html.Br())
    components.append(html.Hr())
    components.append(html.H6('Download Files:'))
#     components.append(html.Br())
    components.append(render_downloads(category, dataset, data_path))
    components.append(dcc.Download(id="download-ds"))
    components.append(html.Br())
    
    return html.Div(components, style={'margin': '20px'})

def message(msg):
    return html.Span(msg, style={'font-style': 'italic'})

def render_results(dataset):
    
    if not os.path.exists(RESULTS_FILE):
        return message('Result file not set in application.')
    
    df = pd.read_csv(RESULTS_FILE, index_col=0)
    df = df[df['dataset'] == dataset]
    
    if len(df) <= 0:
        return message('Results not available for this dataset.')
    
    records = []
    def apline(i, key, value, df):
        line = {}
        line['Best'] = key.replace('_', ' ').title()
        
#         i = df[key].idxmin() if key != 'accuracy' else df[key].idxmax()
        line['Result'] = format_hour(df[key][i]) if key != 'metric:accuracy' else df[key][i]
        
        method = df['method'][i]
        mname = METHODS_NAMES[method] if method in METHODS_NAMES.keys() else method
        mlink = mname.split('-')[0].split(' ')[0].replace('Pivots', 'Movelets') # TODO: TEMP Fix, look for a better way
        line['Method'] = '['+mname+'](../../method/'+mlink+')'
        
        method = df['classifier'][i]
        method = CLASSIFIERS_NAMES[method] if method in CLASSIFIERS_NAMES.keys() else method
        line['Classifier'] = method
        records.append(line)
    
    i = df['metric:accuracy'].idxmax()
    apline(i, 'metric:accuracy', df['metric:accuracy'][i], df)
    i = df['runtime'].idxmin()
    apline(i, 'runtime', format_hour(df['runtime'][i]), df)
    i = df[df['cls_runtime'] > 0]['cls_runtime'].idxmin()
    apline(i, 'cls_runtime', format_hour(df['cls_runtime'][i]), df)
    i = df['totaltime'].idxmin()
    apline(i, 'totaltime', format_hour(df['totaltime'][i]), df)
        
    return dash_table.DataTable(data=records, columns=[
            {"name": ' ', "id":  'Best'},
            {"name": 'Result', "id":  'Result'},
            {"name": 'Method', "id":  'Method', 'type': 'text', "presentation": "markdown",},
            {"name": 'Classifier', "id":  'Classifier'},
        ], 
        style_cell={'padding-left': '5px', 'padding-right': '5px'}, css=[{
            'selector': 'table',
            'rule': '''
                width: auto !important;
            '''
        }],
        markdown_options={'link_target': '_self',},
    )
    
#     return html.Div([
#             html.Span(method+': '),
#             html.Span(str(acc)),
# #             html.Br(),
#         ]),

def render_related_publications(dataset):
    if not os.path.exists(RESULTS_FILE):
        return message('Result file not set in application.')
    
    df = pd.read_csv(RESULTS_FILE, index_col=0)
    df = df[df['dataset'] == dataset]
    
    if len(df) <= 0:
        return message('Related publications not available for this dataset.')
    
    txt = '| Title | Authors | Year | Venue | Links | Cite |\n|:------|:--------|------|:------|:------|:----:|\n'
    
    ls = [METHODS_NAMES[x].split('-')[0] if x in METHODS_NAMES.keys() else x for x in df['method'].unique()]
#     ls = list(df['method'].unique())
    for method in set(ls):
        file = os.path.join(PACKAGE_NAME, 'assets', 'method', method+'.md')
        if os.path.exists(file):
            with open(file, 'r') as f:
                line = f.read().splitlines()
                line = line[-1]
                if line.startswith('|'):
                    txt += line + '\n'

    return dcc.Markdown(txt, className='markdown')
#     return html.Div(
#         children=[html.A(dbc.Badge(
#             x,
#             color="white",
#             text_color="primary",
#             className="border me-1 btn-lg",
#         ), href='../method/'+x.split('-')[0]) for x in ls],
# #         style={'content: ', '; '},
#     )

def render_downloads(category, dataset, data_path=DATA_PATH):
    files = glob.glob(os.path.join(data_path, category, dataset, '*.*'))
    descs = list_subsets(dataset, category, os.path.join(data_path, category, dataset, dataset+'.md'), True)
    
    ls = []
    for f in files:
        if not f.endswith('.md'):
            ls.append(f)
    
    components = [dbc.ListGroupItem(
            html.A(os.path.basename(x), href="javascript:void(0);", #href=dataset+'/'+os.path.basename(ls[i]), 
                   id={
                        'type': 'download-ds-file',
                        'index': dataset+'/'+os.path.basename(x)
                   },
            )
        ) for x in ls]
    
    if len(descs) > 0:
        components = components + \
            [dbc.ListGroupItem(
                [
                    html.A(os.path.basename(x), href="javascript:void(0);", #href=dataset+'/'+os.path.basename(ls[i]), 
                       id={
                            'type': 'download-ds-file',
                            'index': x
                       },
                    ),
                    dccBadge(dataset, category, translateDesc(dataset, category, os.path.basename(x)), {'float':'right'}),
                ]) for x in descs]
    
    return dbc.ListGroup(components)

# ------------------------------------------------------------
def badgeClass(category, subset):
    return 'dataset-color-' + (category if subset in ['specific', '*'] else subset) + ('-default' if subset == category else '')

def dccBadge(dataset, category, subset, style={}):
    return dbc.Badge(translateCategory(dataset, category, subset), 
                     style=style,
                     color="primary", 
                     className="me-1 " + badgeClass(category, subset))

def toBadge(dataset, category, subset):
    return '<span class="badge rounded-pill '+badgeClass(category, subset)+'">'+translateCategory(dataset, category, subset) +'</span>'

def getBadges(category, dataset, subsets):
    # Read the descriptors:
    badges = [toBadge(dataset, category, x) for x in subsets]
    
    if category+'.*' in SUBSET_TYPES:
        badges = [toBadge(dataset, category, '*')] + badges
    
    return ' '.join([x for x in badges])

@app.callback(
    Output("download-ds", "data"),
#     Output({'type': 'download-ds-file', 'index': MATCH}, 'n_clicks'),
    Input({'type': 'download-ds-file', 'index': ALL}, 'n_clicks'),
    State({'type': 'download-ds-file', 'index': ALL}, 'id'),
    prevent_initial_call=True,
)
def download(n_clicks, id):
    triggered = [t["prop_id"] for t in dash.callback_context.triggered][0]
    triggered = eval(triggered.replace('.n_clicks', ''))
    href = triggered['index']
    href = glob.glob(os.path.join(DATA_PATH, '*', href))[0]
    return dcc.send_file(href)