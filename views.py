from lib import fetch_data_db, save_data_db
from dash import dash_table, dcc, html
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px

def layout_main():
    return html.Div([
        html.H1("Main Page"),
        html.Div("Hello World"),
        dcc.Link('Go to Data Models', href='/models')
    ])

def layout_model_selection():
    query = ("SELECT id, model_id, model_type, annual_return_os_pct, sharpe_os, calmar_os, max_dd_os_pct, end_os, trades_os, n_neurons, num_train_params, horizon, alpha, beta, period FROM metrics")
    # Fetch data from database
    data = fetch_data_db(query)
    # Adding navigation urls
    data['url'] = f'model/' + data['id'].astype('string') + '/metrics'
    col_names = ['id', 'Model', 'Type', 'Annual return[%]', 'Sharpe', 'Calmar', 'Max DD[%]', 'Date', 'Num trades']
    hidden_cols = ['url', 'id', 'n_neurons', 'num_train_params', 'horizon', 'alpha', 'beta', 'period']
    # Create a Dash DataTable
    table = dash_table.DataTable(
        id='table',
        columns=[{"name": col_names[step], "id": data.columns[step]} for step in range(len(data.columns) - len(hidden_cols))],
        data=data.to_dict('records'),
        hidden_columns=hidden_cols,
        sort_action='native',
        cell_selectable=True
    )

    df = data[[
        'n_neurons', 'num_train_params', 'horizon', 'alpha', 'beta', 'period',
    ]]

    n_samples = df.shape[0]
    perplexity_value = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=0)
    X_2d = tsne.fit_transform(df)
    # Convert the t-SNE results into a DataFrame
    tsne_df = pd.DataFrame(X_2d, columns=['TSNE1', 'TSNE2'])

    # If you have labels for each model, add them to the DataFrame
    tsne_df['label'] = data['model_id']  # Uncomment and set labels if available

    # Create the Plotly figure
    fig = px.scatter(tsne_df, x='TSNE1', y='TSNE2',
                 color='label',  # Remove or modify if labels not available
                 title='t-SNE visualization of Models')

    # Hide the legend
    fig.update_layout(showlegend=False)

    return html.Div([
        html.H1("Models"),
        table,
        html.Div(id='hidden-div', style={'display':'none'}),
        dcc.Graph(figure=fig)
    ])

def layout_insert():
    data_insert = [{
        'model_id':'PUM','model_type': 'RandomForest', 'end_os': '2023-08-24 9:50:00',
        'annual_return_os_pct': 27.8, 'sharpe_os': 1.5, 'calmar_os': 2.3, 'max_dd_os_pct': 30.9, 'trades_os': 3000
    }]
    df = pd.DataFrame(data_insert)
    id = save_data_db(df, 'metrics')

    return html.Div([
        html.H1(f'Ok{id}'),
        ])

def layout_model_metrics(id):
    query = (f"SELECT model_id, model_type, end_os FROM metrics WHERE id = '{id}'")
    # Fetch data from database
    data = fetch_data_db(query)

    # Create a Dash DataTable
    table = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in data.columns],
        data=data.to_dict('records'),
        hidden_columns=['model_id'],
    )

    return html.Div([
        html.H1(f"Model Metrics of {data.iloc[0]['model_id']}"),
        table,
        html.Div(id='hidden-div', style={'display':'none'}),
        dcc.Link('Go to Models Page', href='/')
    ])
