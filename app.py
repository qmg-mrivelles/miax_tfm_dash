from dash import Dash, html

app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div(children='Hello World')
])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=True)
