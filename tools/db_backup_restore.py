"""
# examples
# python db_backup_restore.py --from_env staging --to_env dev --client dycora
# python db_backup_restore.py --from_env staging --to_env dev --client dycora --filename client_backup_20200709.bak
# python db_backup_restore.py --from_env staging --to_env dev --client dycora --backup

"""
import json
import sys
import time
from datetime import datetime

import boto3
import fire
from eliot import log_message, to_file
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

to_file(sys.stdout)


class DbConnect():
    def __init__(self):
        self.region_name = "us-east-1"

    def make_db_engines(self, env, client):
        """
        Due to the limitation of AWS RDS SQL Server (100 DBs per instance), we need to distribute clients between servers
        We will use AWS Secrets to configure the client database connection string, and in the future, we will move this into PG DB
        Note: Each client will have secret param, which will point to the correct db connection string
            DB-<client>
                prod.db
                dev.db
                staging.db

        Default: We will default to {self.env}-sqlserver

        Based on the environment connects to the respective database.
        :param client: client name
        :return: Client SQL engine
        """
        session = boto3.session.Session()
        secrets_client = session.client(
            service_name="secretsmanager", region_name=self.region_name
        )

        try:
            sqldb_secret = json.loads(secrets_client.get_secret_value(SecretId=f'DB-{client}')["SecretString"])[f'{env}.db']
        except:
            sqldb_secret = f'{env}-sqlserver'

        # Get database Passwords from AWS SecretsManager
        sql_server_info = json.loads(
            secrets_client.get_secret_value(SecretId=sqldb_secret)["SecretString"]
        )

        client_url = URL(
            drivername="mssql+pyodbc",
            username=sql_server_info["username"],
            password=sql_server_info["password"],
            host=sql_server_info["host"],
            port=sql_server_info["port"],
            database="master",
            query={"driver": "ODBC Driver 17 for SQL Server", "autocommit": True},
        )

        return create_engine(client_url)

    def verify_connectivity(self, engine):
        """
        function to verify connection with db.
        :param engine:
        :return:
        """
        assert engine.execute('select 1').fetchall() is not None  # verify connectivity


class DbDump():

    def date_in_string(self):
        """
        function converts today's date into string format.
        :return:
        """
        now = datetime.now()
        date_time = now.strftime("%Y%m%d")
        return date_time

    def monitor_status(self, client):
        """
        function to monitor status of restoring/dumping  db.
        :param client:
        :return:
        """
        return f"""exec msdb.dbo.rds_task_status @db_name='{client}'"""

    def backup_db_to_S3(self, env, client, filename):
        """
        function returns command to dump db from respective env to respective client s3 location.
        :param env:
        :param client:
        :param filename:
        :return:
        """
        file_arn = f"arn:aws:s3:::saiva-{env}-data-bucket/dumps/{filename}"
        dump_command = f"""exec msdb.dbo.rds_backup_database
                          @source_db_name='{client}',
                          @s3_arn_to_backup_to='{file_arn}',
                          @overwrite_S3_backup_file=1;"""
        return dump_command

    def restore_db_from_s3(self, env, client, filename):
        """
        function returns command to dump db from respective env to respective client s3 location.
        :param db_engine:
        :param env:
        :param client:
        :param filename:
        :return:
        """
        file_arn = f"arn:aws:s3:::saiva-{env}-data-bucket/dumps/{filename}"
        restore_command = f"""exec msdb.dbo.rds_restore_database
                            @restore_db_name='{client}',
                            @s3_arn_to_restore_from='{file_arn}'"""
        return restore_command

    def move_dump_within_s3(self, from_env, to_env, filename):
        """
        copying filename.bak file from source s3 path to destination s3 path.
        :param from_env:
        :param to_env:
        :param filename:
        :return:
        """
        s3 = boto3.resource('s3')
        copy_source = {
            'Bucket': f'saiva-{from_env}-data-bucket',
            'Key': f'dumps/{filename}'
        }
        bucket = s3.Bucket(f'saiva-{to_env}-data-bucket')
        bucket.copy(copy_source, f'dumps/{filename}')


def status_checker(db_engine, object, env, client, filename):
    """
    function continously checks the status of the exeuted db command.
    :param db_engine:
    :param object:
    :param env:
    :param client:
    :param filename:
    :return:
    """
    status_command = db_engine.execute(object.monitor_status(client))
    z = dict(zip(status_command.keys(), status_command.first()))
    status = z["lifecycle"]
    while status in (["CREATED", "IN_PROGRESS"]):
        time.sleep(15)
        status_command = db_engine.execute(object.monitor_status(client))
        z = dict(zip(status_command.keys(), status_command.first()))
        status = z["lifecycle"]
        percentage_completion = z["% complete"]
        log_message(
            message_type="info",
            result=f"Process status - {status}. Completed - {percentage_completion}%",
        )
    if status == "SUCCESS":
        log_message(
            message_type="info",
            result=f"Process completed successfully with status {status}. Env -> {env}. Filename -> {filename}",
        )
        db_engine.dispose()
    else:
        raise Exception(
            f"Process failed with status {status}. Env -> {env}. Filename -> {filename}"
        )


def execute(from_env, to_env, client, filename='', action='all', to_db=None):
    """
    1. Connect with db and Execute backup command from_env db -> s3 from_env data bucket dump.
    2. If status is SUCCESS, then dispose the connection.
    3. Copy the .bak file from s3 from_env to s3 to_env dump.
    4. Connect with db and Execute restoration command s3 to_env data bucket dump -> to_env db.

    :param from_env: db-env from where backup is made.
    :param to_env: db-env where restoration of file is done from s3.
    :param client: client for which the db restoration is done.
    :param filename: add custom file name. by default - client_backup_todays date.bak
    :param action: if action = 'backup' - run only backup process
                   if action = 'copy' - run only copy command
                   if action = 'restore' - run only restore command
                   if action = 'all' - run the whole script
    :return:
    """
    obj = DbDump()
    if filename == '':
        filename = client + '_backup_' + obj.date_in_string() + '.bak'
    # =========================loading and verifying the connection with from_env db ==============================
    if action in ['all', 'backup']:
        conn = DbConnect()
        db_engine = conn.make_db_engines(from_env, client)
        conn.verify_connectivity(db_engine)
        log_message(
            message_type="info",
            result=f"Connection with {client}-{from_env} db established.",
        )
        # =========================executing the backup command from_env db -> s3 from_env dump ========================
        result = db_engine.execute(obj.backup_db_to_S3(from_env, client, filename))
        log_message(
            message_type="info",
            result=f"Backup process from db to S3 started...",
        )
        status_checker(db_engine, obj, from_env, client, filename)
        if action in ['backup']:
            exit(0)
    # ================Backup completed. Copying .bak file from  'from_env' S3 to  'to_env S3' dump location============
    if action in ['all', 'copy']:
        log_message(
            message_type="info",
            result=f"Starting {filename} copy from source S3 location to Destination S3 location.",
        )
        obj.move_dump_within_s3(from_env, to_env, filename)
        log_message(
            message_type="info",
            result=f"File copy completed.",
        )
        if action in ['copy']:
            exit(0)
    # ================File copy completed.Starting Restoring process.===================================================
    if action in ['all', 'restore']:
        log_message(
            message_type="info",
            result=f"Starting db restoration process",
        )
        # =========================loading and verifying the connection with to_env db ==============================
        conn = DbConnect()
        db_engine = conn.make_db_engines(to_env, client)
        conn.verify_connectivity(db_engine)
        log_message(
            message_type="info",
            result=f"Connection with {client}-{to_env} db established.",
        )
        to_db_name = to_db if to_db is not None else client
        # =========================executing the restoration command s3 to_env dump -> to_env db ===========================
        db_engine.execute(f"DROP DATABASE IF EXISTS [{to_db_name}];")
        result = db_engine.execute(obj.restore_db_from_s3(to_env, to_db_name, filename))
        status_checker(db_engine, obj, to_env, to_db_name, filename)
        log_message(
            message_type="info",
            result=f"Db restoration process completed successfully",
        )


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes.
    fire.Fire(execute)
