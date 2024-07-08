"""
Creates an index on patient_id for the specified table
eg. python create_index.py --env dev --client marquis --table view_ods_progress_note
"""
import os
import json
import fire
import boto3
import pandas as pd
from eliot import start_action
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.sql import text


region_name = "us-east-1"

def main(env, client, table):

    with start_action(action_type="get_secrets"):
        # Get database Passwords from AWS SecretsManager
        session = boto3.session.Session()
        secrets_client = session.client(
            service_name="secretsmanager", region_name=region_name
        )
        client_db_info = json.loads(
            secrets_client.get_secret_value(SecretId=f"{env}-sqlserver")[
                "SecretString"
            ]
        )

    with start_action(action_type="generate_db_urls"):
        client_url = URL(
            drivername="mssql+pyodbc",
            username=client_db_info["username"],
            password=client_db_info["password"],
            host=client_db_info["host"],
            port=client_db_info["port"],
            database=client,
            query={"driver": "ODBC Driver 17 for SQL Server"},
        )

    # Establish connection with client sqlserver
    with start_action(
        action_type="connect_to_databases",
        client_url=repr(client_url)
    ):
        engine = create_engine(client_url, echo=False)

    with engine.connect().execution_options(autocommit=True) as con:
        # Check if index already exists, if not create the index
        statement = text(f"""
        IF NOT EXISTS (
        	SELECT
        		*
        	FROM
        		sys.indexes
        	WHERE
        		name = 'idx_patientid' AND object_id = OBJECT_ID ('{table}'))
        BEGIN
        	CREATE INDEX idx_patientid ON {client}.dbo.{table} (patientid)
        END;""")
        con.execute(statement)


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes.
    fire.Fire(main)