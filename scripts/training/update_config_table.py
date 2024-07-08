"""
Run the script to copy facility_model_config & client_metadata from DEV environment to STAGING or PROD
Commands:
python update_config_table.py --to_env dev --client trio --table_name facility_model_config --para model_golive_date --val 2021-09-06
python update_config_table.py --to_env dev --client trio --table_name facility_model_config --para facility_golive_date --val 2021-08-21
python update_config_table.py --to_env dev --client trio --table_name facility_model_config --para deleted_at --val 2021-08-21
python update_config_table.py --to_env dev --client trio --table_name facility_model_config --facilityids 1,2,3 --para group_level --val facility
python update_config_table.py --to_env dev --client trio --table_name facility_model_config --facilityids 1,2,3 --para active_facility --val true
python update_config_table.py --to_env dev --client trio --table_name client_metadata --para deleted_at --val 2021-08-21

NOTE: 'active_facility' & 'group_level' must always be updated via this script since it creates a history
record on updating the respective values.
"""

import json
from datetime import datetime

import boto3
import fire
import pandas as pd
from eliot import log_message
from pytz import timezone
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

REGION_NAME = "us-east-1"
TZ = timezone('US/Eastern')


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

    def update_row(self, client, destination_dbengine, table_name, para, val, facilityids):
        if para == 'deleted_at':
            val = datetime.now(TZ)

        facilityids_str = ','.join(str(s) for s in facilityids)
        if facilityids:
            query_str = f"""update {table_name} set {para}='{val}' where client='{client}' and facilityid in ({facilityids_str}) and deleted_at is null """
        else:
            query_str = f"""update {table_name} set {para}='{val}' where client='{client}' and deleted_at is null """

        print("Query: " + query_str)
        destination_dbengine.execute(query_str)

    def update_and_insert(self, client, destination_dbengine, table_name, para, val, facilityids):
        """
        Whenever group_level or active_facility paramter is updated we mark the old row
        as deleted and insert a new row.
        """

        facilityids_str = ','.join(str(s) for s in facilityids)
        query = f""" select * from {table_name} where client='{client}' and facilityid in ({facilityids_str}) and deleted_at is null """
        df = pd.read_sql(query, con=destination_dbengine)
        # set the new parameter and make sure the created_at timestamp is set correctly too!
        df[para] = val
        df['created_at'] = datetime.now(TZ)

        destination_dbengine.execute(
            f"""update {table_name} set deleted_at=now() where client='{client}' and facilityid in ({facilityids_str}) and deleted_at is null """
        )
        df.to_sql(
            table_name,
            destination_dbengine,
            if_exists="append",
            index=False,
        )


def run(to_env, client, table_name, para=None, val=None, facilityids=''):
    facilityid_tuple = ()
    if para in ['active_facility', 'group_level'] and facilityids == '':
        raise Exception('"facilityids" parameter is required.')
    else:
        if facilityids != '':
            if type(facilityids) == int:
                # single integer passed! So, force it to be a tuple
                facilityid_tuple = (facilityids, )
            elif type(facilityids) == str:
                # actually split becomes a list, but that is ok
                facilityid_tuple = facilityids.split(',')
            else:
                # Fire passes in a tuple if it has commas in it and no quotes
                facilityid_tuple = facilityids
    if para is None or val is None:
        raise Exception('"para/val" needs to be specified.')
    if para not in ['deleted_at', 'active_facility', 'model_golive_date', 'facility_golive_date', 'group_level']:
        raise Exception('Invalid parameter specified.')

    metadata = ClientMetadata()
    # Open a connection to destination environment
    destination_dbengine = metadata.get_postgresdb_engine(env=to_env)
    if para in ['active_facility', 'group_level']:
        metadata.update_and_insert(client, destination_dbengine, table_name, para, val, facilityid_tuple)
    else:
        metadata.update_row(client, destination_dbengine, table_name, para, val, facilityid_tuple)
    print("Done!")


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes.
    fire.Fire(run)
