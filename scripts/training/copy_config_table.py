"""
Run the script to copy facility_model_config & client_metadata from DEV environment to STAGING or PROD
Commands:
python copy_config_table.py --to_env staging --client coxsunshine --table_name client_metadata
python copy_config_table.py --to_env staging --client coxsunshine --table_name facility_model_config
"""

import json

import boto3
import fire
import pandas as pd
from eliot import log_message
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

REGION_NAME = "us-east-1"


class ClientMetadata(object):
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

    def fetch_data(self, client, dev_dbengine, table_name):
        """
        - We select only the latest facility-model configuration from dev server.
        - As part of dev we might run multiple experiments and we may insert multiple configurations
        for testing purpose but its not neccessary we add the same to prod. Hence we choose only the
        latest rows having deleted_at = NULL and insert them to staging or prod environment.
        """
        query = f""" select * from {table_name} where client='{client}' and deleted_at is null """
        return pd.read_sql(query, con=dev_dbengine)

    def insert_data(self, client, destination_dbengine, df, table_name):
        destination_dbengine.execute(
            f"""update {table_name} set deleted_at=now() where client = '{client}' and deleted_at is null """
        )

        df.to_sql(
            table_name,
            destination_dbengine,
            if_exists="append",
            index=False,
        )

    def duplicate_exists(self, client, destination_dbengine, table_name, source_df):
        query = f""" select * from {table_name} where client='{client}' and deleted_at is null """
        df = pd.read_sql(query, con=destination_dbengine)

        if table_name == 'client_metadata':
            merge_cols = ['client', 'ingestion_method', 'ingestion_version',
                          'model_version', 'report_version']
        elif table_name == 'facility_model_config':
            df['modelid'] = df['modelid'].str.strip()
            merge_cols = ['client', 'facilityid', 'modelid',
                          'facility_golive_date', 'rank_cutoff',
                          'model_golive_date', 'active_facility']
        else:
            raise Exception('Invalid Table..')

        _df = pd.concat([df, source_df])
        _df = _df.drop_duplicates(merge_cols)

        # If the dataframe is empty after dropping duplicates it means there were duplicate records
        if _df.empty:
            return True
        else:
            return False


def run(to_env, client, table_name):
    metadata = ClientMetadata()
    # Open a connection to DEV environment
    dev_dbengine = metadata.get_postgresdb_engine(env='dev')
    # Open a connection to destination environment
    destination_dbengine = metadata.get_postgresdb_engine(env=to_env)

    source_df = metadata.fetch_data(client, dev_dbengine, table_name)

    is_duplicate = metadata.duplicate_exists(client, destination_dbengine, table_name, source_df)

    if not is_duplicate:
        metadata.insert_data(client, destination_dbengine, source_df, table_name)
    else:
        print('#############################################'
              '# DUPLICATE RECORDS - did not insert any records to DB.'
              '#############################################')


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes.
    fire.Fire(run)
