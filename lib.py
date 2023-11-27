import pandas as pd
import re
from sqlalchemy import create_engine

# Connect to the database
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

# Function to get model id from url, like params
def get_model_id(pathname):
    model_id_pattern = r'/model/(?P<id>[\w_]+)/'
    model_id = ''
    match_model_id = re.match(model_id_pattern, pathname)
    if match_model_id:
        model_id = match_model_id.groups()[0]

    return model_id
