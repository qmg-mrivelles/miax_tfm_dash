import os
import re
import pandas as pd
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from views import layout_model_selection, layout_model_metrics, create_line_figure, create_underwater_figure, \
    create_graphical_analisis
from lib import get_model_id
from io import StringIO
from dotenv import load_dotenv

load_dotenv()  # This loads the variables from .env

service_account_key_path = './special_key.json'
# Set the environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_path

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(
        html.Img(src='/assets/logo.jpeg', style={'width': '400px'}),
        style={'text-align': 'center'}
    ),
    html.Div(id='page-content')
], className='container')

app.clientside_callback(
    """
    function(active_cell, data) {
        if(active_cell) {
            var row = data[active_cell.row_id];
            // Assuming you have a column that contains the URL or identifier
            var url = '/' + row.url;
            console.log(row);
            window.location.href = url;
        }
    }
    """,
    Output('hidden-div', 'children'),  # This output is not used, but necessary for the callback
    [Input('table', 'active_cell')],
    [State('table', 'data')]
)


@app.callback(
    Output('zoom-state', 'data'),
    [Input('line-chart', 'relayoutData')],
    [State('zoom-state', 'data')],
    prevent_initial_call=True
)
def update_zoom_state(relayoutData, current_zoom_state):
    if relayoutData and 'xaxis.range[0]' in relayoutData:
        return {
            'xaxis.range[0]': relayoutData['xaxis.range[0]'],
            'xaxis.range[1]': relayoutData['xaxis.range[1]']
        }
    return current_zoom_state


@app.callback(
    Output('line-chart', 'figure'),
    [Input('zoom-state', 'data'),
     Input('df_ec', 'data')],
    prevent_initial_call=True
)
def update_line_chart(zoom_state, json_data):
    str_io = StringIO(json_data)
    df = pd.read_json(str_io, orient='split')
    return create_line_figure(df, zoom_state)


@app.callback(
    Output('underwater-chart', 'figure'),
    [Input('zoom-state', 'data'),
     Input('df_ec', 'data')],
    prevent_initial_call=True
)
def update_underwater_chart(zoom_state, json_data):
    str_io = StringIO(json_data)
    df = pd.read_json(str_io, orient='split')
    return create_underwater_figure(df, zoom_state)


@app.callback(
    Output('analisis', 'children'),
    [Input('size-dropdown', 'value'),
     Input('df_data', 'data')]
)
def update_analisis_graph(size, json_data):
    str_io = StringIO(json_data)
    df = pd.read_json(str_io, orient='split')
    return create_graphical_analisis(df, size)


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    model_id = get_model_id(pathname)
    model_metrics_by_id = r'/model/(?P<id>[\w_]+)/metrics'

    # Start asking by the most children to least, HTML Router
    if re.match(model_metrics_by_id, pathname):
        return layout_model_metrics(model_id)
    elif pathname == '/' or pathname == '':
        return layout_model_selection()
    else:
        return '404'


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get("PORT", 8080)),
        debug=False,
        dev_tools_ui=False,
        dev_tools_props_check=False)
