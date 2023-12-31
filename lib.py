import os
import re
import pandas as pd
from sqlalchemy import create_engine
from google.cloud import storage
from io import StringIO, BytesIO


# Connect to the database
def create_engine_mysql():
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('DB_HOST')
    database = os.getenv('DB_NAME')
    return create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')


# Function to fetch data from the database
def fetch_data_db(query):
    """Fetch data from MySQL database."""
    engine = create_engine_mysql()
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


# Function to insert data to the database
def save_data_db(df, table_name):
    """Insert data to MySQL database."""
    engine = create_engine_mysql()
    with engine.connect() as connection:
        return df.to_sql(name=table_name, con=engine, index=False, if_exists='append')


# Function to insert data to the bucket
def save_data_bucket(df, bucket_name, destination_blob_name):
    """Insert data to Bucket GCP."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    # Convert DataFrame to CSV in memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Rewind the buffer to the beginning of the file
    # Create a new blob and upload the file's content.
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')


# Function to read data to the bucket
def read_data_bucket(bucket_name, source_blob_name):
    """Read data from Bucket GCP."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    # Get blob
    blob = bucket.blob(source_blob_name)
    # Download the blob to a BytesIO object
    byte_stream = BytesIO()
    blob.download_to_file(byte_stream)
    byte_stream.seek(0)

    # Read this stream into a DataFrame
    df = pd.read_csv(byte_stream)
    return df


# Function to get model id from url, like params
def get_model_id(pathname):
    model_id_pattern = r'/model/(?P<id>[\w_]+)/'
    model_id = ''
    match_model_id = re.match(model_id_pattern, pathname)
    if match_model_id:
        model_id = match_model_id.groups()[0]

    return model_id


# Function to read equity curve csv
def read_equity_curve(model_id):
    bucket_name = 'miax-data'
    source_blob_name = f'equity_curves/{model_id}.csv'
    df = read_data_bucket(bucket_name, source_blob_name)
    return df
