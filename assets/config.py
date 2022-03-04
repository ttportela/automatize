DATA_PATH = '../../datasets'
EXP_PATH  = '../../workdir/results/Ensemble1'
README    = 'automatize/README.md'

PAGES_ROUTE =  'automatize/assets'

RESULTS_FILE    = 'automatize/assets/experimental_history.csv'

# page_title = 'Tarlis\'s Multiple Aspect Trajectory Analysis'
page_title = 'Automatize'

def underDev(pathname):
    import dash_bootstrap_components as dbc
    from dash import html
    return html.Div([
            dbc.Alert('Content in development. You are on page {}'.format(pathname), color="info", style = {'margin':10})
        ])

def alert(msg, mtype="info"):
    import dash_bootstrap_components as dbc
    return dbc.Alert(msg, color=mtype, style = {'margin':10})

def render_markdown_file(file, div=False):
    from dash import html
    from dash import dcc
    f = open(file, "r")
    if div:
        return html.Div(dcc.Markdown(f.read()), style={'margin': '20px'}, className='markdown')
    else:
        return dcc.Markdown(f.read())