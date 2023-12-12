from lib import fetch_data_db, read_equity_curve
from dash import dash_table, dcc, html
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px


def layout_model_selection():
    query = (
        "SELECT id, model_id, model_type, annual_return_os_pct, sharpe_os, calmar_os, max_dd_os_pct, end_os, trades_os, bar_kind, bar_size, signal_type, alpha, beta, sortino_os, test_balanced_accuracy FROM metrics")
    # Fetch data from database
    data = fetch_data_db(query)
    # Adding navigation urls
    data['url'] = f'model/' + data['id'].astype('string') + '/metrics'
    col_names = ['id', 'Model', 'Type', 'Annual return[%]', 'Sharpe', 'Calmar', 'Max DD[%]', 'Date', 'Num trades']
    hidden_cols = ['url', 'id', 'bar_kind', 'bar_size', 'signal_type', 'alpha', 'beta', 'sortino_os',
                   'test_balanced_accuracy']
    # Create a Dash DataTable
    table = dash_table.DataTable(
        id='table',
        columns=[{"name": col_names[step], "id": data.columns[step]} for step in
                 range(len(data.columns) - len(hidden_cols))],
        data=data.to_dict('records'),
        hidden_columns=hidden_cols,
        sort_action='native',
        cell_selectable=True,
        page_size=20
    )

    categorical_cols = ['bar_kind', 'bar_size', 'signal_type', 'model_type', 'alpha', 'beta']
    numerical_cols = ['sharpe_os', 'calmar_os', 'sortino_os', 'test_balanced_accuracy', 'max_dd_os_pct', 'trades_os']
    columns = categorical_cols + numerical_cols
    df = data[columns]
    df = df.dropna()

    # Pre procesado de los datos para clustering
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())  # Standardize numerical features
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding categorical
    ])

    # Combine transformers into a preprocessor with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Apply transformations to the data
    df_prepared = preprocessor.fit_transform(df)

    n_samples = df_prepared.shape[0]
    perplexity_value = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=0)
    X_2d = tsne.fit_transform(df_prepared)
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
        html.H1("Modelos"),
        table,
        html.Div(id='hidden-div', style={'display': 'none'}),
        dcc.Graph(figure=fig)
    ])


def layout_model_metrics(id):
    query = (f"SELECT * FROM metrics WHERE id = '{id}'")
    # Fetch data from database
    data = fetch_data_db(query)
    metrics = data.iloc[0]

    tab1_content = dbc.Card(
        dbc.CardBody([
            html.P("A continuacion se muestran las caracteristicas del modelo", className="card-text"),
            html.Div([
                dbc.Row(
                    [
                        dbc.Col(html.Div([
                            html.Div(f"Modelo: {metrics['model_type']}"),
                            html.Div(f"Neuronas: {metrics['n_neurons']}"),
                            html.Div(f"Dropout: {metrics['dropout_value']}"),
                            html.Div(f"Num. Parametros: {metrics['num_train_params']}"),
                            html.Div(f"Horizonte: {metrics['horizon']}"),
                            html.Div(f"Periodo: {metrics['period']}"),
                            html.Div(f"Alfa: {metrics['alpha']}"),
                            html.Div(f"Beta: {metrics['beta']}"),
                            html.Div(f"Barrido: {metrics['barrier_method']}"),
                        ]), width=4),
                        dbc.Col(html.Div([
                            html.Div(f"Loss: {metrics['loss']}"),
                            html.Div(f"Accuracy: {metrics['accuracy']}"),
                            html.Div(f"Balanced Acc.: {metrics['balanced_accuracy']}"),
                            html.Div(f"Val. Loss: {metrics['val_loss']}"),
                            html.Div(f"Val. Acc.: {metrics['val_accuracy']}"),
                            html.Div(f"Val. Balanced Acc.: {metrics['val_balanced_accuracy']}"),
                            html.Div(f"Test Loss: {metrics['test_loss']}"),
                            html.Div(f"Test Acc.: {metrics['test_accuracy']}"),
                            html.Div(f"Test Balanced Acc.: {metrics['test_balanced_accuracy']}"),
                            html.Div(f"Training Time: {metrics['training_time']}"),
                        ])),
                        dbc.Col(html.Div([
                            html.Div(f"Symbol: {metrics['symbol']}"),
                            html.Div(f"Bar Type: {metrics['bar_kind']}"),
                            html.Div(f"Bar Size Acc.: {metrics['bar_size']}"),
                            html.Div(f"Signal: {metrics['signal_type']}"),
                            html.Div(f"Name: {metrics['model_name']}"),
                        ]), width=4)
                    ])
            ])
        ]),
        className="mt-3",
    )

    # Equity curve chart
    df_ec = read_equity_curve(metrics['model_id'])
    df_ec['datetime'] = pd.to_datetime(df_ec['datetime'])
    df_ec = df_ec.set_index('datetime')
    fig_ec = px.line(df_ec['Equity'])
    fig_ec.update_layout(
        xaxis_title="",
        yaxis_title="",
        title="Equity Curve",
        legend_title=None
    )

    tab2_content = dbc.Card(
        dbc.CardBody([
            html.P("A continuacion se muestran los resultados de backtest in sample", className="card-text"),
            html.Div([
                dbc.Row(
                    [
                        dbc.Col(html.Div([
                            html.Div(f"Annual Return[%]: {metrics['annual_return_is_pct']}"),
                            html.Div(f"Annual Volatility[%]: {metrics['annual_vol_is_pct']}"),
                            html.Div(f"Sharpe[%]: {metrics['sharpe_is']}"),
                            html.Div(f"Sortino[%]: {metrics['sortino_is']}"),
                            html.Div(f"Calmar[%]: {metrics['calmar_is']}"),
                            html.Div(f"First Trade: {metrics['start_is']}"),
                            html.Div(f"Last Trade: {metrics['end_is']}"),
                            html.Div(f"Duration Trading: {metrics['duration_is']}"),
                        ])),
                        dbc.Col(html.Div([
                            html.Div(f"Max DD[%]: {metrics['max_dd_is_pct']}"),
                            html.Div(f"Avg DD[%]: {metrics['avg_dd_is_pct']}"),
                            html.Div(f"Num Trades: {metrics['trades_is']}"),
                            html.Div(f"Num Bars: {metrics['n_bars_is']}"),
                        ])),
                        dbc.Col(html.Div([
                            html.Div(f"Win Rate[%]: {metrics['win_rate_is_pct']}"),
                            html.Div(f"Best Trade[%]: {metrics['best_trade_is_pct']}"),
                            html.Div(f"Worst Trade[%]: {metrics['worst_trade_is_pct']}"),
                            html.Div(f"Avg Trade[%]: {metrics['avg_trade_is_pct']}"),
                            html.Div(f"Profit Factor[%]: {metrics['profit_factor_is']}"),
                        ]))
                    ]),
                html.Br(),
                html.P("A continuacion se muestran los resultados de backtest out sample", className="card-text"),
                dbc.Row(
                    [
                        dbc.Col(html.Div([
                            html.Div(f"Annual Return[%]: {metrics['annual_return_os_pct']}"),
                            html.Div(f"Annual Volatility[%]: {metrics['annual_vol_os_pct']}"),
                            html.Div(f"Sharpe[%]: {metrics['sharpe_os']}"),
                            html.Div(f"Sortino[%]: {metrics['sortino_os']}"),
                            html.Div(f"Calmar[%]: {metrics['calmar_os']}"),
                            html.Div(f"First Trade: {metrics['start_os']}"),
                            html.Div(f"Last Trade: {metrics['end_os']}"),
                            html.Div(f"Duration Trading: {metrics['duration_os']}"),
                        ])),
                        dbc.Col(html.Div([
                            html.Div(f"Max DD[%]: {metrics['max_dd_os_pct']}"),
                            html.Div(f"Avg DD[%]: {metrics['avg_dd_os_pct']}"),
                            html.Div(f"Num Trades: {metrics['trades_os']}"),
                            html.Div(f"Num Bars: {metrics['n_bars_os']}"),
                        ])),
                        dbc.Col(html.Div([
                            html.Div(f"Win Rate[%]: {metrics['win_rate_os_pct']}"),
                            html.Div(f"Best Trade[%]: {metrics['best_trade_os_pct']}"),
                            html.Div(f"Worst Trade[%]: {metrics['worst_trade_os_pct']}"),
                            html.Div(f"Avg Trade[%]: {metrics['avg_trade_os_pct']}"),
                            html.Div(f"Profit Factor[%]: {metrics['profit_factor_os']}"),
                        ]))
                    ]),
            ]),
            html.Br(),
            dcc.Graph(figure=fig_ec)
        ]),
        className="mt-3",
    )

    tabs = dbc.Tabs(
        [
            dbc.Tab(tab1_content, label="Model"),
            dbc.Tab(tab2_content, label="Backtest"),
        ]
    )

    return html.Div([
        html.H1(f"Model metrics of:"),
        html.H5(f"{metrics['model_id']}"),
        html.Br(),
        tabs,
        html.Br(),
        dcc.Link('Go to Models Page', href='/')
    ])
