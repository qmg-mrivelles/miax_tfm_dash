from lib import fetch_data
from dash import dash_table, dcc, html

def layout_main():
    return html.Div([
        html.H1("Main Page"),
        html.Div("Hello World"),
        dcc.Link('Go to Data Models', href='/models')
    ])

def layout_model_selection():
    query = "SELECT model_id, model_type, end_os FROM metrics"
    # Fetch data from database
    data = fetch_data(query)
    # Adding navigation urls
    data['url'] = 'model/' + data['model_id'] + '/metrics'

    # Create a Dash DataTable
    table = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in data.columns],
        data=data.to_dict('records'),
        row_selectable='single',
    )

    return html.Div([
        html.H1("Models"),
        table,
        html.Div(id='hidden-div', style={'display':'none'}),
        dcc.Link('Go to Main Page', href='/')
    ])

def layout_model_metrics(model_id):
    query = f"SELECT model_type, end_os FROM metrics WHERE model_id = '{model_id}'"
    # Fetch data from database
    data = fetch_data(query)

    # Create a Dash DataTable
    table = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in data.columns],
        data=data.to_dict('records'),
        row_selectable='single'
    )

    return html.Div([
        html.H1(f"Model Metrics of {model_id}"),
        table,
        html.Div(id='hidden-div', style={'display':'none'}),
        dcc.Link('Go to Models Page', href=f'/models')
    ])
