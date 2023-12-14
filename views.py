from lib import fetch_data_db, read_equity_curve
from dash import dash_table, dcc, html
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from scipy.cluster.hierarchy import linkage, ward, fcluster


def prepare_cluster_data(data):
    # SELECCION VARIABLES para aplicar CLUSTERING
    categorical_cols = ['bar_kind', 'bar_size', 'signal_type', 'model_type', 'alpha', 'beta']
    numerical_cols = ['sharpe_os', 'calmar_os', 'sortino_os', 'test_balanced_accuracy', 'max_dd_os_pct', 'trades_os']

    # dataframe
    columns = categorical_cols + numerical_cols
    df = data[columns]
    df = df.dropna()  # en los algoritmos muy malos, el backtest rellena de nan en las metricas, me los puedo quitar

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
    # Get feature names for categorical variables after one-hot encoding
    cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(
        input_features=categorical_cols)

    # Combine with numerical column names
    all_features = numerical_cols + cat_features.tolist()

    # Convert the array to a DataFrame
    df_prepared = pd.DataFrame(df_prepared, columns=all_features)

    return df_prepared


def create_dendogram(df_prepared):
    # Create the dendrogram
    labels = ['' for _ in range(df_prepared.shape[0])]
    fig = ff.create_dendrogram(df_prepared, linkagefun=lambda x: linkage(x, "ward"), labels=labels)
    fig.update_layout(
        autosize=True,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        xaxis_title="",
        yaxis_title="Distance",
        title="Hierarchical Clustering Dendrogram",
        legend_title=None
    )
    return fig


def create_graphical_analisis(data, nivel=20):
    numerical_cols = ['sharpe_os', 'calmar_os', 'sortino_os', 'test_balanced_accuracy', 'max_dd_os_pct', 'trades_os']

    df_prepared = prepare_cluster_data(data)
    df_numerical = data[numerical_cols]
    df_numerical = df_numerical.dropna()

    Z = linkage(df_prepared, method='ward')
    clusters = fcluster(Z, t=nivel, criterion='distance')  # "corto" en clusters
    # Cantidad de muestras por cluester
    df_prepared['cluster'] = clusters

    # Muestras
    fig_muestras = px.bar(df_prepared.groupby('cluster').size())
    fig_muestras.update_layout(
        xaxis_title="Cluster size",
        yaxis_title="Muestras",
        title="Cantidad de muestras por cluster",
        legend_title=None,
        plot_bgcolor="white",
        height=300
    )

    df_report = df_prepared.copy()
    df_report.index = df_numerical.index
    df_report[numerical_cols] = df_numerical

    # Sharpe
    fig_sharpe = px.box(df_report, x='cluster', y='sharpe_os', color='cluster')
    fig_sharpe.update_layout(
        xaxis_title="",
        yaxis_title="Sharpe Ratio",
        title="Sharpe Ratio by Cluster",
        legend_title=None
    )

    # Test Balanced Accuracy
    fig_balanced = px.box(df_report, x='cluster', y='test_balanced_accuracy', color='cluster')
    fig_balanced.update_layout(
        xaxis_title="",
        yaxis_title="Test Balanced Accuracy",
        title="Test Balanced Accuracy by Cluster",
        legend_title=None
    )

    # Test Balanced Accuracy
    fig_calmar = px.box(df_report, x='cluster', y='calmar_os', color='cluster')
    fig_calmar.update_layout(
        xaxis_title="",
        yaxis_title="Calmar Ratio",
        title="Calmar Ratio by Cluster",
        legend_title=None
    )

    # Calmar Ratio
    fig_calmar = px.box(df_report, x='cluster', y='calmar_os', color='cluster')
    fig_calmar.update_layout(
        xaxis_title="",
        yaxis_title="Calmar Ratio",
        title="Calmar Ratio by Cluster",
        legend_title=None
    )

    # Max. Drawdown [%]
    fig_max_dd = px.box(df_report, x='cluster', y='max_dd_os_pct', color='cluster')
    fig_max_dd.update_layout(
        xaxis_title="",
        yaxis_title="Max. Drawdown [%]",
        title="Max Drawdown by Cluster",
        legend_title=None
    )

    # Sortino Ratio
    fig_sortino = px.box(df_report, x='cluster', y='sortino_os', color='cluster')
    fig_sortino.update_layout(
        xaxis_title="",
        yaxis_title="Sortino Ratio",
        title="Sortino Ratio by Cluster",
        legend_title=None
    )

    # Trades_OS
    fig_trades = px.box(df_report, x='cluster', y='trades_os', color='cluster')
    fig_trades.update_layout(
        xaxis_title="",
        yaxis_title="Trades",
        title="Trades by Cluster",
        legend_title=None
    )

    return html.Div([
        dbc.Row(
            [
                dbc.Col(html.Div([dcc.Graph(figure=fig_muestras)])),
            ]),
        dbc.Row(
            [
                dbc.Col(html.Div([dcc.Graph(figure=fig_sharpe)])),
                dbc.Col(html.Div([dcc.Graph(figure=fig_balanced)]))
            ]),
        dbc.Row(
            [
                dbc.Col(html.Div([dcc.Graph(figure=fig_calmar)])),
                dbc.Col(html.Div([dcc.Graph(figure=fig_max_dd)]))
            ]),
        dbc.Row(
            [
                dbc.Col(html.Div([dcc.Graph(figure=fig_sortino)])),
                dbc.Col(html.Div([dcc.Graph(figure=fig_trades)]))
            ])
    ])


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
    df_prepared = prepare_cluster_data(data)
    fig_dendogram = create_dendogram(df_prepared.iloc[:1000])
    analisis_html = create_graphical_analisis(data)

    return html.Div([
        html.H1("Modelos"),
        table,
        html.Div(id='hidden-div', style={'display': 'none'}),
        dcc.Graph(figure=fig_dendogram, style={'height': '600px', 'width': '100%'}),
        html.Div([dbc.Row(
            [
                dbc.Col(html.Div([
                    html.H3("Analisis grafico del clustering")
                ])),
                dbc.Col(html.Div([
                    html.Label("Distance"),
                    dcc.Dropdown(
                        id='size-dropdown',
                        options=[{'label': i, 'value': i} for i in np.arange(1, 51, 1)],
                        value=20  # Default value
                    ),
                ])),
            ])
        ]),
        html.Div(id='analisis', children=analisis_html),
        dcc.Store(id='df_data', data=data.to_json(date_format='iso', orient='split'))
    ])


def create_line_figure(df, zoom_state=None):
    fig = px.line(df['Equity'])
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Equity value",
        title="Equity Curve",
        legend_title=None
    )
    if zoom_state:
        fig.update_layout(xaxis_range=[zoom_state['xaxis.range[0]'], zoom_state['xaxis.range[1]']])

    return fig


def create_underwater_figure(df, zoom_state=None):
    fig = px.bar(df['DrawdownPct'] * -1 * 100)
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Drawdown %",
        title="",
        legend_title=None,
        plot_bgcolor="white",
        height=250
    )

    if zoom_state:
        fig.update_layout(xaxis_range=[zoom_state['xaxis.range[0]'], zoom_state['xaxis.range[1]']])

    return fig


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

    fig_ec = create_line_figure(df_ec)

    # Drawdown Bars
    fig_dd = create_underwater_figure(df_ec)

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
            dcc.Graph(figure=fig_ec, id='line-chart'),
            dcc.Graph(figure=fig_dd, id='underwater-chart'),
            dcc.Store(id='zoom-state'),
            dcc.Store(id='df_ec', data=df_ec.to_json(date_format='iso', orient='split'))
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
