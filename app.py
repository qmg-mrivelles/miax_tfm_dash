import os
import re
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from views import layout_model_selection, layout_model_metrics, layout_insert
from lib import get_model_id
import os

# Replace with the path to your service account key file
service_account_key_path = './special_key.json'

# Set the environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_path

app = Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css'])
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

app.clientside_callback(
    """
    function(active_cell, data) {
        if(active_cell) {
            var row = data[active_cell.row];
            // Assuming you have a column that contains the URL or identifier
            var url = '/' + row.url;
            window.location.href = url;
        }
    }
    """,
    Output('hidden-div', 'children'),  # This output is not used, but necessary for the callback
    [Input('table', 'active_cell')],
    [State('table', 'data')]
)

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    model_id = get_model_id(pathname)
    model_metrics_by_id = r'/model/(?P<id>[\w_]+)/metrics'

    # Start asking by the most children to least, HTML Router
    if re.match(model_metrics_by_id, pathname):
        return layout_model_metrics(model_id)
    #elif pathname == '/insert':
    #    return layout_insert();
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
