import json
import typing

import boto3
import sqlalchemy.engine

from eliot import log_message
from shared.constants import ENV, REGION_NAME, SAIVADB_CREDENTIALS_SECRET_ID
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL


class DbEngine(object):
    """
    Fetch the credentials from AWS Secrets Manager.
    :return: DB connection to the respective database
    """

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

    def get_postgresdb_engine(self, db_name=None) -> sqlalchemy.engine.Engine:
        """
        Based on the environment connects to the respective database
        :param client: client name
        :return: Saivadb Postgres engine
        """
        log_message(message_type='info', action_type='connect_to_postgresdb', client='SaivaDB')
        # Fetch credentials from AWS Secrets Manager
        postgresdb_info = self.get_secrets(secret_name=SAIVADB_CREDENTIALS_SECRET_ID)
        # Create DB URL
        saivadb_url = URL(
            drivername='postgresql',
            username=postgresdb_info['username'],
            password=postgresdb_info['password'],
            host=postgresdb_info['host'],
            port=postgresdb_info['port'],
            database=db_name or postgresdb_info['dbname'],
        )
        # Return Postgres Engine
        return create_engine(saivadb_url, echo=False, hide_parameters=True)

    def get_sqldb_engine(
        self,
        *,
        db_name: str,
        query: typing.Dict,
        credentials_secret_id: str
    ) -> sqlalchemy.engine.Engine:
        log_message(message_type='info', action_type='connect_to_sqldb', client=db_name)

        # Fetch credentials from AWS Secrets Manager
        sqldb_info = self.get_secrets(secret_name=credentials_secret_id)
        sqldb_info['dbname'] = db_name

        # Create DB URL
        client_sqldb_url = URL(
            drivername='mssql+pyodbc',
            username=sqldb_info['username'],
            password=sqldb_info['password'],
            host=sqldb_info['host'],
            port=sqldb_info['port'],
            database=sqldb_info['dbname'],
            query=query,
        )
        # Return Sql Engine
        return create_engine(client_sqldb_url, connect_args={"TrustServerCertificate": "yes"})

    def verify_connectivity(self, engine):
        assert engine.execute('select 1').fetchall() is not None  # verify connectivity
