DATA_PATH='../../datasets'
EXP_PATH='../../workdir/results/Ensemble1'

def underDev(pathname):
    import dash_bootstrap_components as dbc
    from dash import html
    return html.Div([
            dbc.Alert('Content in development. You are on page {}'.format(pathname), color="info", style = {'margin':10})
        ])