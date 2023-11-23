import os
#import mysql.connector
import pandas as pd
from sqlalchemy import create_engine
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output

app = Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css'])
server = app.server

# Connect to the database
#cnx = mysql.connector.connect(**config)

def create_engine_mysql():
    user = 'root'
    password = 'comtrend'
    host = '35.202.233.213'
    database = 'miax-data'
    return create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')

# Function to fetch data from the database
def fetch_data(query):
    """Fetch data from MySQL database."""
    #cnx = mysql.connector.connect(**config)
    #df = pd.read_sql(query, cnx)
    #cnx.close()
    #return df
    engine = create_engine_mysql()
    with engine.connect() as connection:
        return pd.read_sql(query, connection)

def layout_main():
    return html.Div([
        html.H1("Main Page"),
        html.Div("Hello World"),
        dcc.Link('Go to Data Table', href='/data-table')
    ])


def layout_data_table():
    # Fetch data from database
    data = fetch_data("SELECT * FROM people")

    # Create a Dash DataTable
    table = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in data.columns],
        data=data.to_dict('records'),
    )

    return html.Div([
        html.H1("Data Table"),
        table,
        dcc.Link('Go to Main Page', href='/')
    ])


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/data-table':
        return layout_data_table()
    else:
        return layout_main()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)
