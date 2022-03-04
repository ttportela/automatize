import sys, os 
import pandas as pd
import glob2 as glob
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join('.')))

import io
import base64
from datetime import datetime

import dash
from dash import dash_table
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State

from automatize.helper.CDDiagram import draw_cd_diagram 
from automatize.results import format_hour

from automatize.app_base import app
from automatize.assets.config import *
from automatize.assets.helper.script_inc import METHODS_NAMES, CLASSIFIERS_NAMES
# ------------------------------------------------------------
# EXP_PATH='../../workdir/'

def render_page_results(pathname):
    content = []
    
    if pathname == '/results':
        content = render_experiments()
    else:
        content = render_method(pathname)
        
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
)
def render_experiments_call(sel_datasets=None, sel_methods=None, sel_classifiers=None):
    return render_experiments(sel_datasets, sel_methods, sel_classifiers)
    
def render_experiments(sel_datasets=None, sel_methods=None, sel_classifiers=None, path=EXP_PATH):
#     hsf = os.path.join('automatize', 'assets', 'experiments_history.csv')
#     from automatize.results import history
#     df = history(path)
#     df.to_csv(hsf)

    time = datetime.fromtimestamp(os.path.getmtime(RESULTS_FILE))
    df = pd.read_csv(RESULTS_FILE, index_col=0)
    
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
    
    return [
        html.Div([
            html.Div([
                html.Div([
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
                ], style={'width': '50%', 'padding': 10, 'flex': 1}),
                html.Div([
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
        html.Br(),
    ]  

def render_experiments_panels(df):
    return dcc.Tabs(id="results-tabs", value='tab-1', children=[
        dcc.Tab(label='Critical Difference', value='tab-1', children=[render_expe_graph(df.copy())]),
        dcc.Tab(label='Average Ranking', value='tab-2', children=[render_avg_rank(df.copy())]),
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

def render_avg_rank(df): 
    cls_name = 'method'
    ds_key = 'key'
    rank_col = 'accuracy'
    
#     m = len(df[cls_name].unique())
#     nb_datasets = df.groupby([ds_key]).size()
    
#     sorted_df = df.sort_values([cls_name, ds_key])
    
#     print('sorted_df[rank_col]', sorted_df)
    
#     rank_data = np.array(sorted_df[rank_col]).reshape(m, nb_datasets)
#     sorted_df['rank'] = sorted_df[rank_col].rank()

    sorted_df = df.sort_values([cls_name, ds_key])
    sorted_df = sorted_df.rank(ascending=True).groupby(cls_name).mean()

    for row in sorted_df.items():
        print(row)
    
#     print('sorted_df', sorted_df)
    return underDev('Methods Rank')

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