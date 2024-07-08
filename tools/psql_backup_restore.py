"""
Configured to take a backup and restore of any database inside saivadb RDS for different env
Examples:
python psql_backup_restore.py --from_env staging --to_env dev --db_name webapp
"""

import json
import os
import subprocess

import boto3
import fire
import psycopg2
import psycopg2.extras
from eliot import log_message
from psycopg2 import sql

REGION_NAME = "us-east-1"


class DbEngine(object):
    """
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

    def get_postgresdb_info(self, env, db_name):
        """
        Based on the environment connects to the respective database
        :param client: client name
        :return: Saivadb Postgres engine
        """
        log_message(message_type='info', action_type='connect_to_postgresdb', client='SaivaDB')
        # Fetch credentials from AWS Secrets Manager
        postgresdb_info = self.get_secrets(secret_name=f'{env}-saivadb')

        postgresdb_info['database'] = db_name

        # Return Postgres credentials
        return postgresdb_info

    def db_dump(self, dbinfo, filename):
        command = f"pg_dump postgresql://{dbinfo['username']}:{dbinfo['password']}@{dbinfo['host']}:{dbinfo['port']}/{dbinfo['database']} > ./{filename}"
        print('******************EXECUTING******************')
        print(command)
        subprocess.run(
            command,
            shell=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )

    def db_restore(self, dbinfo, filename):
        print('******************RESTORE STARTED******************')
        connection = self.get_db_connection(dbinfo, 'saivadb')
        connection.autocommit = True
        cursor = connection.cursor()
        print('******************Clear all existing sessions******************')
        cursor.execute(sql.SQL(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid <> pg_backend_pid() AND datname = 'webapp' ")
        )
        print('******************Drop webapp database******************')
        cursor.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier('webapp')))
        print('******************Create webapp database******************')
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier('webapp')))

        connection.commit()
        cursor.close()

        command = f"psql postgresql://{dbinfo['username']}:{dbinfo['password']}@{dbinfo['host']}:{dbinfo['port']}/{dbinfo['database']} < ./{filename}"
        print('******************Restoring the DUMP******************')
        print(command)
        subprocess.run(
            command,
            shell=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )

    def get_db_connection(self, postgresdb_info, dbname):
        # Create db connection
        connection = psycopg2.connect(
            user=postgresdb_info['username'],
            password=postgresdb_info['password'],
            host=postgresdb_info['host'],
            port=postgresdb_info['port'],
            database=dbname
        )
        # Return Postgres Engine
        return connection


def run(from_env, to_env, db_name='webapp'):
    db = DbEngine()
    dbinfo = db.get_postgresdb_info(env=from_env, db_name=db_name)
    db.db_dump(dbinfo, 'db.dump')
    dbinfo = db.get_postgresdb_info(env=to_env, db_name=db_name)
    db.db_restore(dbinfo, 'db.dump')
    os.remove("db.dump")


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes.
    fire.Fire(run)
