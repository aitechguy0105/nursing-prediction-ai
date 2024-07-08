"""
Run the script to copy facility_model_config & client_metadata from DEV environment to STAGING or PROD
Commands:
python copy_model_metadata.py --to_env prod --mlflow_experiment_id 178
"""

import json

import boto3
import fire
import pandas as pd
from eliot import log_message
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

REGION_NAME = "us-east-1"


class ModelMetadata(object):
    def __init__(self, region_name=REGION_NAME):
        self.session = boto3.session.Session()
        self.secrets_client = self.session.client(
            service_name='secretsmanager',
            region_name=region_name
        )

    def get_secrets(self, secret_name):
        """
        :return: Based on the environment get secrets for
        Client SQL db & Postgres Saivadb
        """
        log_message(message_type='info', action_type='get_secrets', secret_name=secret_name)
        db_info = json.loads(
            self.secrets_client.get_secret_value(SecretId=secret_name)[
                'SecretString'
            ]
        )
        return db_info

    def get_postgresdb_engine(self, env):
        """
        Based on the environment connects to the respective database
        :param client: client name
        :return: Saivadb Postgres engine
        """
        log_message(message_type='info', action_type='connect_to_postgresdb', client='SaivaDB')
        # Fetch credentials from AWS Secrets Manager
        log_message(message_type='info', action_type='connect_to_postgresdb', client='SaivaDB')
        # Fetch credentials from AWS Secrets Manager
        postgresdb_info = self.get_secrets(secret_name=f'{env}-saivadb')
        # Create DB URL
        saivadb_url = URL(
            drivername='postgresql',
            username=postgresdb_info['username'],
            password=postgresdb_info['password'],
            host=postgresdb_info['host'],
            port=postgresdb_info['port'],
            database=postgresdb_info['dbname'],
        )
        # Return Postgres Engine
        return create_engine(saivadb_url, echo=False)

    def fetch_data(self, dev_dbengine, mlflow_experiment_id):
        query = f""" select * from model_metadata where model_s3_folder='{mlflow_experiment_id}' """
        return pd.read_sql(query, con=dev_dbengine)

    def insert_data(self, destination_dbengine, mlflow_experiment_id, df):
        destination_dbengine.execute(
            f"""delete from model_metadata where model_s3_folder='{mlflow_experiment_id}' """
        )

        df.to_sql(
            'model_metadata',
            destination_dbengine,
            if_exists="append",
            index=False,
        )


def run(to_env, mlflow_experiment_id):
    metadata = ModelMetadata()
    # Open a connection to DEV environment
    dev_dbengine = metadata.get_postgresdb_engine(env='dev')
    # Open a connection to destination environment
    destination_dbengine = metadata.get_postgresdb_engine(env=to_env)

    source_df = metadata.fetch_data(dev_dbengine, mlflow_experiment_id)
    metadata.insert_data(destination_dbengine, mlflow_experiment_id, source_df)


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes.
    fire.Fire(run)
