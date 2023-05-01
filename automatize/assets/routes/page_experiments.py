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

from datetime import datetime
import zipfile
import tempfile

import dash
from dash import dash_table
from dash import dcc
from dash import html, callback_context, MATCH, ALL
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State

from automatize.assets.app_base import app
from automatize.assets.config import *

from automatize.script import gensh, trimsuffix
from automatize.inc.script_def import *
from automatize.inc.datasets_def import list_datasets
# ------------------------------------------------------------
# EXP_PATH='../../workdir/'
    
all_methods = BASE_METHODS
#[
#    'hiper+Log', 
#    
#    'SM+Log',
#    
#    'MM+Log',
#    
#    'MARC',
#    'poi',
#    'npoi',
#    'wpoi',
#    
#    'Movelets',
#    'Dodge',
#    'Xiao',
#    'Zheng',
#    
#    'TC-TRF',
#]

if os.environ.get('AUTOMATIZE_DATA_PATH') is not None:
    data_path = os.environ['AUTOMATIZE_DATA_PATH']
else:
    data_path = DATA_PATH
# ------------------------------------------------------------
all_datasets = list_datasets(data_path)

# all_executables = {
#     'hiper':           os.path.join(PACKAGE_NAME, 'assets', 'method', 'HIPERMovelets.jar'), 
#     'hiper-pivots':    os.path.join(PACKAGE_NAME, 'assets', 'method', 'HIPERMovelets.jar'), 
#     'SM':              os.path.join(PACKAGE_NAME, 'assets', 'method', 'SUPERMovelets.jar'),
#     'MM':              os.path.join(PACKAGE_NAME, 'assets', 'method', 'MASTERMovelets.jar'),
#     'MMp':             os.path.join(PACKAGE_NAME, 'assets', 'method', 'MASTERMovelets.jar'),
#     
#     'Movelets':        os.path.join(PACKAGE_NAME, 'assets', 'method', 'Movelets.jar'),
#     'Dodge':           os.path.join(PACKAGE_NAME, 'assets', 'method', 'Dodge.jar'),
#     'Xiao':            os.path.join(PACKAGE_NAME, 'assets', 'method', 'Xiao.jar'),
#     'Zheng':           os.path.join(PACKAGE_NAME, 'assets', 'method', 'Zheng.jar'),
# }

all_executables = {
    'hiper':           'HIPERMovelets.jar', 
    'hiper-pivots':    'HIPERMovelets.jar', 
    'SM':              'SUPERMovelets.jar',
    'MM':              'MASTERMovelets.jar',
    'MMp':             'MASTERMovelets.jar',
    
    'Movelets':        'Movelets.jar',
    'Dodge':           'Dodge.jar',
    'Xiao':            'Xiao.jar',
    'Zheng':           'Zheng.jar',
}
# ------------------------------------------------------------

def render_page_experiments(pathname):
    content = []
    
#     if pathname == '/experiments':
#         content = render_experiments()
#     else:
#         content = render_method(pathname)
        
    return html.Div(children=[
#         html.H3('Experimental Evaluations'),
        render_markdown_file(PAGES_ROUTE+'assets/pages/experiments.md'),
        
        html.Div([
            html.Div(children=[
                html.Strong('Installation Root Path:'),
                dbc.Input(id='input-exp-root', value='/home/user/experiment', type='text', style={'width': '100%'}),
                
                html.Strong('Experiment Group Prefix:'),
                dbc.Input(id='input-exp-folder', value='EXP01', type='text', style={'width': '100%'}),
                
                html.Br(),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(dbc.Checkbox(id='input-exp-use-dataset', value=False)), 
                        dbc.InputGroupText('Alternative Datasets Path:'),
                        dbc.Input(id='input-exp-datafolder', value='/home/user/experiment/data', type='text'),
                    ],
                    className="mb-3",# style={'display': 'inline-flex'}
                ),
                
#                 html.Br(),
#                 dbc.Input(type="number", min=0, step=1, style={'width': '100px'}),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(dbc.Checkbox(id='input-exp-use-tc', value=True)), 
                        dbc.InputGroupText('Set a Timeout: '),
                        dbc.Input(id='input-exp-tc', type="number", min=0, step=1, value=7),
                        dbc.InputGroupText(dbc.InputGroup(
                            [
                                dcc.RadioItems(
                                    id='input-exp-tccode',
                                    options=[
                                        {'label': ' '+y+' ', 'value': y[0].lower()} \
                                        for y in ['Minutes', 'Hours', 'Days', 'Weeks']
                                    ],
                                    value='d',
        #                             style={'display': 'inline-flex'},
                                    inputStyle={'margin-left': '10px'},
        #                             inline=True,
        #                             switch=True
                                ),

                            ],
        #                     className="mb-3"
        #                     style={'display': 'inline-flex', 'width': 'auto'},
                        ))
                    ],
                    className="mb-3", style={'width': '100%', 'display': 'inline-flex'}
                ),
                
                html.Br(),
                html.Strong('Select Datasets:'),
                dcc.Dropdown(
                        id='input-exp-datasets', 
                        options=[
                            {'label': x, 'value': y} for y,x in all_datasets.items()
                        ],
                        multi=True,
#                         options={'required':'true'},
                ),
                
#                 html.Br(),
#                 html.Strong('Select Methods:'),
#                 dcc.Dropdown(
#                         id='input-experiments-methods',
#                         options=[
#                             {'label': x, 'value': y} for y, x in all_methods.items()
#                         ],
#                         multi=True,
# #                         value=list(all_methods.keys()),
#                 ),
                
            ], style={'padding': 10, 'flex': 1}),

            html.Div(children=[
                html.Strong('Other Options:'),
#                 html.Br(),
                dbc.Checkbox(id='input-exp-use-exe', label='Include methods executable', value=True),
                
#                 html.Br(),                
                dbc.InputGroup(
                    [
                        dbc.InputGroupText('# of Threads:'), 
                        dbc.Input(id='input-exp-nt', type="number", min=0, step=1, value=4),
                        dbc.InputGroupText('Mem. Limit:'),
                        dbc.Input(id='input-exp-gb', type="number", min=0, step=1, value=600),
                        dbc.InputGroupText('GB'), 
                    ],
                    className="mb-3",# style={'display': 'inline-flex'}
                ),
                
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(dbc.Checkbox(value=True)), 
                        dbc.InputGroupText('K-Fold:'),
                        dbc.Input(id='input-exp-k', type="number", min=1, step=1, value=5),
                        dbc.InputGroupText('resamples'),
                    ],
                    className="mb-3",
                ),
                
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(dbc.Checkbox(value=True)), 
                        dbc.InputGroupText('Python command:'),
                        dbc.Input(id='input-exp-pyname', value='python3', type='text'),
                    ],
                    className="mb-3",
                ),
                
                html.Br(),
#                 html.H6('For HiPerMovelets: '),
#                 dbc.InputGroup(
#                     [
#                         dbc.InputGroupText(dbc.Checkbox(value=False)), 
#                         dbc.InputGroupText('τ'),
#                         dbc.InputGroupText('TAU (%):'),
#                         dbc.Input(type="number", min=0, step=0.01, value=0.9),
#                     ],
#                     className="mb-3",
#                 ),
                
#                 html.H6('For POI-F: '),
#                 dbc.InputGroup(
#                     [
#                         dbc.InputGroupText(dbc.Checkbox(value=False)),
#                         dbc.InputGroupText('Sequence Sizes:'), 
#                     ],
#                     className="mb-3", style={'width': '100%'}
#                 ),
#                 dcc.Slider(value=3, id='my-slider',
#                     min=1, max=10, step=1,
#                     marks={i: '{}'.format(i) for i in range(1, 11)},
#                     updatemode='drag', #style={'width': '100%', 'display': 'inline-flex'}
#                 ),
                
                
                html.Strong('Select Methods:'),
                dbc.InputGroup(
                    [
                        dbc.Select(
                            id='input-experiments-methods',
                            options=[
                                {'label': METHODS_NAMES[y], 'value': y} for y in all_methods
                            ],
                            value=all_methods[0]
#                             multi=False,
    #                         value=list(all_methods.keys()),
#                             style={'witdh': '60%'},
#                             className='form-control',
                        ),
                        dbc.Button('Add Method', outline=True, color="success", id='experiments-method-add'), 
                    ],
#                     className="mb-3", style={'width': '100%'}
                ),
#                 dcc.Dropdown(
#                         id='input-experiments-methods',
#                         options=[
#                             {'label': x, 'value': y} for y, x in all_methods.items()
#                         ],
#                         multi=False,
# #                         value=list(all_methods.keys()),
#                         style={'witdh': '60%', 'float': 'right'},
#                 ),
#                 dbc.Button('Add Method', id='experiments-method-add'),
                
                
                html.Br(),
                html.Br(),
                
                dbc.Button('Reset', id='experiments-reset', style={'float': 'left'}, outline=True, color="warning"),
                dbc.Button('Download Environment', id='experiments-download', style={'float': 'right'}),
                
            ], style={'padding': 10, 'flex': 1})
        ], style={'display': 'flex', 'flex-direction': 'row'}),
        
        dcc.Download(id="download-experiments"),
        dbc.Alert('Configure the environment, select datasets, and add methods to your experiments. Then, click Download Environment to generate files and folders.', id="experiments-err", color='secondary', style = {'margin':10}),
        html.Hr(),
        dbc.Accordion(id='experiments-methods-container', children=[]),
        html.Div(id='output-experiments', children=content),
        html.Br(),
        html.Br(),
    ], style={'margin':10})

# --------------------------------------------------------------------------------
# Methods add:
TO_GEN = []

@app.callback(
    Output('experiments-methods-container', 'children'),
    Input('experiments-method-add', 'n_clicks'),
    State('input-experiments-methods', 'value'),
    State('experiments-methods-container', 'children'),
    Input('experiments-reset', 'n_clicks'),
)
def display_methods(idx, method, children, reset):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
#     print(changed_id)
    if 'experiments-reset' in changed_id:
        return reset_methods()
    
    idx = len(TO_GEN) + 1
    
    if method in ['poi', 'npoi', 'wpoi']:
        item = dbc.AccordionItem(
            [
                dbc.Label('Sequence Sizes:'),
                dcc.Slider(value=3, #id='exp-poi-'+str(idx),
                    id={
                        'type': 'exp-poi-sequences',
                        'index': idx
                    },
                    min=1, max=10, step=1,
                    marks={i: '{}'.format(i) for i in range(1, 11)},
                    updatemode='drag', #style={'width': '100%', 'display': 'inline-flex'}
                ),
            ],
            title=getTitle(method, idx), # METHODS_NAMES[method]+' (#'+str(idx)+')',
            id={
                'type': 'exp-poi',
                'index': idx
            },
        )
        TO_GEN.append([method, {'sequences': [1,2,3]}])
    elif 'MM' in method:
        isLog = True if '+Log' in method else False
        isPivots = True if 'MMp' in method else False
        item = dbc.AccordionItem(
            [
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(dbc.Checkbox(value=isLog, id={'type': 'exp-log','index': idx})), 
                        dbc.InputGroupText('Use Log (limit the subtrajectory size to the natural log of trajectory size)'),
                    ],
                    className="mb-3",
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(dbc.Checkbox(value=isPivots, id={'type': 'exp-pivots','index': idx})), 
                        dbc.InputGroupText('Use Pivots'),
                    ],
                    className="mb-3",
                ),
            ],
            title=getTitle(method, idx), # METHODS_NAMES[method]+' (#'+str(idx)+')',
            id={
                'type': 'exp-mm',
                'index': idx
            },
        )
        TO_GEN.append([method, {}])
    elif 'SM' in method:
        isLog = True if '+Log' in method else False
        isLambda = True if '-2' in method else False
        item = dbc.AccordionItem(
            [
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(dbc.Checkbox(value=isLog, id={'type': 'exp-log','index': idx})), 
                        dbc.InputGroupText('Use Log (limit the subtrajectory size to the natural log of trajectory size)'),
                    ],
                    className="mb-3",
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(dbc.Checkbox(value=isLambda, id={'type': 'exp-lambda','index': idx})), 
                        dbc.InputGroupText('Use λ (discover a limit number for dimension combination)'),
                    ],
                    className="mb-3",
                ),
            ],
            title=getTitle(method, idx), # METHODS_NAMES[method]+' (#'+str(idx)+')',
            id={
                'type': 'exp-sm',
                'index': idx
            },
        )
        TO_GEN.append([method, {}])
    elif 'hiper' in method:
        isLog = True if '+Log' in method else False
        isPivots = True if 'pivots' in method else False
        item = dbc.AccordionItem(
            [
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(dbc.Checkbox(value=isLog, id={'type': 'exp-log','index': idx})), 
                        dbc.InputGroupText('Use Log (limit the subtrajectory size to the natural log of trajectory size)'),
                    ],
                    className="mb-3",
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(dbc.Checkbox(value=isPivots, id={'type': 'exp-pivots','index': idx})), 
                        dbc.InputGroupText('Use Pivots (HiPerMovelets-Pivots)'),
                    ],
                    className="mb-3",
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(dbc.Checkbox(value=False, id={'type': 'exp-use-tau','index': idx})), 
                        dbc.InputGroupText('τ'),
                        dbc.InputGroupText('TAU (%):'),
                        dbc.Input(type="number", min=0.01, max=1, step=0.01, value=0.9, id={'type': 'exp-tau','index': idx}),
                        dbc.InputGroupText('Scale: 0.01 to 1.00'),
                    ],
                    className="mb-3",
                ),
            ],
            title=getTitle(method, idx), # METHODS_NAMES[method]+' (#'+str(idx)+')',
            id={
                'type': 'exp-hiper',
                'index': idx
            },
        )
        TO_GEN.append([method, {}])
    elif 'random' in method:
        isLog = True if '+Log' in method else False
        item = dbc.AccordionItem(
            [
                dbc.InputGroup(
                    [
                        dbc.InputGroupText(dbc.Checkbox(value=isLog, id={'type': 'exp-log','index': idx})), 
                        dbc.InputGroupText('Use Log (limit the subtrajectory size to the natural log of trajectory size)'),
                    ],
                    className="mb-3",
                ),
            ],
            title=getTitle(method, idx), # METHODS_NAMES[method]+' (#'+str(idx)+')',
            id={
                'type': 'exp-rm',
                'index': idx
            },
        )
        TO_GEN.append([method, {}])
    else:
        item = dbc.AccordionItem(
            [
                html.I(html.P("Method "+METHODS_NAMES[method]+" has default configuration.")),
            ],
            title=getTitle(method, idx), # METHODS_NAMES[method]+' (#'+str(idx)+')',
            id={
                'type': 'exp-other',
                'index': idx
            },
        )
        TO_GEN.append([method, {}])
    
#     print(TO_GEN)
    
    children.append(item)
    return children

def reset_methods():
    global TO_GEN
    TO_GEN = []
    return []

# --------------------------------------------------------------------------------
@app.callback(
    Output({'type': 'exp-poi', 'index': MATCH}, 'title'),
    Input({'type': 'exp-poi-sequences', 'index': MATCH}, 'value'),
    State({'type': 'exp-poi-sequences', 'index': MATCH}, 'id'),
)
def update_poi(value, id):
    global TO_GEN
    idx = id['index']-1
    
    TO_GEN[idx][1]['sequences'] = [x for x in range(1, value+1)]
#     print(TO_GEN)
    return getTitle(TO_GEN[idx][0], idx+1)# METHODS_NAMES[TO_GEN[idx][0]]+' (#'+str(idx+1)+')'

@app.callback(
    Output({'type': 'exp-sm', 'index': MATCH}, 'title'),
    State({'type': 'exp-log', 'index': MATCH}, 'id'),
    Input({'type': 'exp-log', 'index': MATCH}, 'value'),
    
#     State({'type': 'exp-lambda', 'index': MATCH}, 'id'),
    Input({'type': 'exp-lambda', 'index': MATCH}, 'value'),
)
def update_sm(id, log, lamb):
    global TO_GEN
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
#     print(changed_id)
        
    idx = id['index']-1
    
    if 'exp-log' in changed_id:
        if log:
            TO_GEN[idx][0] = TO_GEN[idx][0] + '+Log'
        else:
            TO_GEN[idx][0] = TO_GEN[idx][0].replace('+Log', '')
        
    if 'exp-lambda' in changed_id:
        if lamb:
            TO_GEN[idx][0] = TO_GEN[idx][0] + '-2'
        else:
            TO_GEN[idx][0] = TO_GEN[idx][0].replace('-2', '')
        
#     print(TO_GEN)
    return getTitle(TO_GEN[idx][0], idx+1)# METHODS_NAMES[TO_GEN[idx][0]]+' (#'+str(idx+1)+')'

@app.callback(
    Output({'type': 'exp-mm', 'index': MATCH}, 'title'),
    
    State({'type': 'exp-log', 'index': MATCH}, 'id'),
    Input({'type': 'exp-log', 'index': MATCH}, 'value'),
    
#     State({'type': 'exp-pivots', 'index': MATCH}, 'id'),
    Input({'type': 'exp-pivots', 'index': MATCH}, 'value'),
)
def update_mm(id, log, pivots):
    global TO_GEN
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
#     print(changed_id)
        
    idx = id['index']-1
    
    if 'exp-log' in changed_id:
        if log:
            TO_GEN[idx][0] = TO_GEN[idx][0] + '+Log'
        else:
            TO_GEN[idx][0] = TO_GEN[idx][0].replace('+Log', '')
        
    if 'exp-pivots' in changed_id:
        TO_GEN[idx][0] = TO_GEN[idx][0].replace('MM', 'MMp') if pivots else TO_GEN[idx][0].replace('MMp', 'MM')
        
#     print(TO_GEN)
    return getTitle(TO_GEN[idx][0], idx+1)# METHODS_NAMES[TO_GEN[idx][0]]+' (#'+str(idx+1)+')'


@app.callback(
    Output({'type': 'exp-hiper', 'index': MATCH}, 'title'),
    
    State({'type': 'exp-log', 'index': MATCH}, 'id'),
    Input({'type': 'exp-log', 'index': MATCH}, 'value'),
    
#     State({'type': 'exp-pivots', 'index': MATCH}, 'id'),
    Input({'type': 'exp-pivots', 'index': MATCH}, 'value'),
    
    Input({'type': 'exp-use-tau', 'index': MATCH}, 'value'),
    Input({'type': 'exp-tau', 'index': MATCH}, 'value'),
)
def update_hiper(id, log, pivots, isTau, tau):
    global TO_GEN
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
#     print(changed_id)
        
    idx = id['index']-1
    
    if 'exp-log' in changed_id:
        if log:
            TO_GEN[idx][0] = TO_GEN[idx][0] + '+Log'
        else:
            TO_GEN[idx][0] = TO_GEN[idx][0].replace('+Log', '')
        
    if 'exp-pivots' in changed_id:
        TO_GEN[idx][0] = TO_GEN[idx][0].replace('hiper', 'hiper-pivots') if pivots else TO_GEN[idx][0].replace('hiper-pivots', 'hiper')
        
    if ('exp-tau' in changed_id and isTau) or ('exp-use-tau' in changed_id and isTau):
        TO_GEN[idx][1]['tau'] = tau
    
    if 'exp-use-tau' in changed_id and not isTau:
        TO_GEN[idx][1]['tau'] = 0.9
        
#     print(TO_GEN)
    return getTitle(TO_GEN[idx][0], idx+1)# METHODS_NAMES[TO_GEN[idx][0]]+' (#'+str(idx+1)+')'

@app.callback(
    Output({'type': 'exp-rm', 'index': MATCH}, 'title'),
    State({'type': 'exp-log', 'index': MATCH}, 'id'),
    Input({'type': 'exp-log', 'index': MATCH}, 'value'),
)
def update_rm(id, log):
    global TO_GEN
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
#     print(changed_id)
        
    idx = id['index']-1
    
    if 'exp-log' in changed_id:
        if log:
            TO_GEN[idx][0] = TO_GEN[idx][0] + '+Log'
        else:
            TO_GEN[idx][0] = TO_GEN[idx][0].replace('+Log', '')
        
#     print(TO_GEN)
    return getTitle(TO_GEN[idx][0], idx+1)# METHODS_NAMES[TO_GEN[idx][0]]+' (#'+str(idx+1)+')'

def getTitle(method, id):
    return str(id)+') ' + METHODS_NAMES[method]

# --------------------------------------------------------------------------------
# DOWNLOAD:
@app.callback(
    Output('download-experiments', 'data'),
    Output('experiments-err', 'children'),
    Input('experiments-download', 'n_clicks'),
    State('input-exp-root', 'value'),
    State('input-exp-folder', 'value'),
    State('input-exp-use-dataset', 'value'),
    State('input-exp-datafolder', 'value'),
    State('input-exp-use-tc', 'value'),
    State('input-exp-tc', 'value'),
    State('input-exp-tccode', 'value'),
    State('input-exp-datasets', 'value'),
    State('input-exp-use-exe', 'value'),
    State('input-exp-nt', 'value'),
    State('input-exp-gb', 'value'),
    State('input-exp-k', 'value'),
    State('input-exp-pyname', 'value'),
    prevent_initial_call=True,
)
def download(value, basedir, folderpref, isDs, datapath, isTC, TC, TCD, datasets, isExe, nt, gb, k, pyname):
    global TO_GEN
    
    if not datasets:
        return dash.no_update, 'You must specify at least one dataset.'
    if len(TO_GEN) <= 0:
        return dash.no_update, 'You must add methods to generate scripts.'

    root = os.path.dirname(basedir)
    base = os.path.basename(basedir)
    
    params_base = {
        'sh_folder': 'scripts', \
        'folder':    folderpref,
        'k':         k, \
        'call_exit': False, \

        'root':      root, \
        'threads':   nt, \
        'gig':       gb, \
        'pyname':    pyname,
        
#         'sequences': [1,2,3],
#         'ensemble_methods': [['MML', 'HL', 'HpL', 'RL', 'U'], ['npoi']],

#         'timeout': '7d', \
    }
    
    if isTC:
        params_base['timeout'] = str(TC) + TCD
        
#     if not isDs: 
#         datapath = os.path.join('${BASE}', 'data')
        
    shdir = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(shdir.name, base))
    os.mkdir(os.path.join(shdir.name, base, 'results'))
    
    params_base['sh_folder'] = os.path.join(shdir.name, base, 'programs')
    os.mkdir(params_base['sh_folder'])
        
    if not isDs: 
        datapath = os.path.join('${BASE}', base, 'data')
        os.mkdir(os.path.join(shdir.name, base, 'data'))
            
    torun = '#!/bin/bash\n'
    for ds in datasets:
        dsType, dsn = ds.split('.')
#         for dsn in ds:
        dsFeat = 'generic' if '?' in dsn else 'specific'
        dsName = dsn.replace('?', '')
        for GM in TO_GEN:
            method = GM[0]
            if 'tau' in GM[1].keys():
                method += '+TR' + str(int(GM[1]['tau'] * 100))
            
            params = params_base.copy()
            
            params.update(GM[1])
            
            params['data_folder'] = os.path.join(datapath, dsType)
#             params['res_path']    = os.path.join(params['root'], 'results', prefix)
            params['features']    = getFeature(dsType, dsn)
            # A>
            torun += gensh(method, {dsName+'.'+getDescName(dsType, dsName): [dsFeat]}, params)
        torun += '\n'
#         torun += '#-------------------------------------------------------------------------------------\n'

    f = open(os.path.join(params['sh_folder'], 'run-all.sh'),'w')
    print(torun, file=f)
    f.close()
    
#     try:
    return prepare_zip(shdir, params, base, isExe, TO_GEN), 'Files generated to "automatize_scripts.zip"'
#     except BaseException as e:
#         print(e)
#         return dash.no_update, 'Sorry, an error ocurred. We are going to revised it.'
        
    
def prepare_zip(shdir, params, base, isExe, TO_GEN):
    zf_tf = tempfile.NamedTemporaryFile(delete=True, suffix='.zip')
    
    zf = zipfile.ZipFile(zf_tf, mode='w', compression=zipfile.ZIP_DEFLATED)
    
    def addFolderToZip(zip_file, folder, basename):
        for file in os.listdir(folder):
            full_path = os.path.join(folder, file)
            if os.path.isfile(full_path):
    #             print 'File added: ' + str(full_path)
                zip_file.write(full_path, full_path.replace(basename, ''))
            elif os.path.isdir(full_path):
    #             print 'Entering folder: ' + str(full_path)
                zip_file.write(full_path, full_path.replace(basename, ''))
                addFolderToZip(zip_file, full_path, basename)
    def close_tmp_file(tf):
        try:
            os.unlink(tf.name)
            tf.close()
        except:
            pass
        
    addFolderToZip(zf, shdir.name, shdir.name)
    
    if isExe:
        import requests
        for GM in TO_GEN:
            if trimsuffix(GM[0]) in all_executables.keys():
                #ff = all_executables[trimsuffix(GM[0])]
                #ft = ff.replace(os.path.join(PACKAGE_NAME, 'assets', 'method'), 
                #                os.path.join(base, 'programs'))
                ff = os.path.join(PACKAGE_NAME, 'assets', 'method', all_executables[trimsuffix(GM[0])])
                if not os.path.exists(ff):
                    jar = 'https://github.com/ttportela/automatize/blob/main/jarfiles/' \
                        + all_executables[trimsuffix(GM[0])] + '?raw=true'
                    print(ff)
                    jar = requests.get(jar, allow_redirects=True)
                    open(ff, 'wb').write(jar.content)
                ft = os.path.join(base, 'programs', all_executables[trimsuffix(GM[0])])
                if ft not in zf.namelist():
                    zf.write(ff, ft)
                    #zf.writestr(ft, data=io.BytesIO(jar.read()))
            
    zf.close()
    
    close_tmp_file(shdir)
#     close_tmp_file(zf_tf)
    
    return dcc.send_file(zf_tf.name, filename="automatize_scripts.zip")
