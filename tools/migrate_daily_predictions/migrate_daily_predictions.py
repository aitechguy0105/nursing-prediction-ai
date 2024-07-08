"""
daily_predictions contains rows for only the residents who were ranked.
For analytics purposes, it is required to have the rows in the daily_census table who were not ranked.
daily_predictions doesn't contain info related to payer and admission
This script will insert unranked residents and fill in the columns for payer and admission in the daily_predictions table

To execute:
python code/migrate_daily_predictions.py --start-date '2021-01-01'

This is the approach taken for migration:
1) For each client, starting from the min prediction date for that client:
    a) Get the unranked census from sqlserver and insert it into a table unranked_census
    b) Get the additional columns for all census dates and insert it into a table census_payer_admission_data
2) Dedup unranked_census census using daily_predictions, removing all the rows which are already part of daily_predictions
3) Insert into unranked_census all the rows from daily_predictions so that unranked_census now has all the ranked & unranked residents for all census dates including
   all the prediction info
4) Join unranked census with census_payer_admission_data to get the additional columns and insert into temp_daily_predictions
5) After validating temp_daily_predictions, add all constraints and triggers to temp_daily_predictions table from daily_predictions
6) Deprecate daily_predictions & rename temp_daily_predictions as daily_predictions 
"""
import json
import os
import sys
import time
import timeit

import boto3
import fire
from eliot import start_action, start_task, log_message, to_file
import pandas as pd

from database import DbEngine
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from settings import get_client_class, ENV

to_file(sys.stdout)

region_name = "us-east-1"
env = os.environ.get("SAIVA_ENV", "dev")

# This dict contains the patient_id col name and excluded census codes for each client
client_config = {
    'avante': {
        'censusactions': "censusactioncode in ('DE', 'DRA', 'DRNA', 'H', 'HI', 'HL', 'HMU', 'L', 'PBH', 'RDD', 'RDE', 'TL', 'TLU', 'TO', 'TP')",
         'patiendid_col': 'clientid'
    },
    'champion': {
        'censusactions': """censusactioncode in ('DE', 'RDE', 'RDH', 'DH', 'DD', 'RDD', 'DAMA', 'HP', 'HUP', 
                        'H30', 'H31', 'H30', 'DP', 'EP', 'TO', 'L', 'TOH', 'LOA', 'TOT', 'TOS', 'TTS', 'TI', 'RL', 
                        'TFH', 'RFL', 'TFT', 'TFR', 'TIS', 'TP', 'TUP', 'T', 'T30', 'T31', 'TN')""",
        'patiendid_col': 'clientid'
    },
    'marquis': {
        'censusactions': """censusactioncode in ('DD', 'DE', 'TO', 'RDD', 'RDE', 'RDH', 'DH', 'HL',
                        'TL', 'L', 'HUP', 'TUP', 'BH', 'HP', 'UHL', 'UTL', 'DAMA', 'DF', 'E', 'BE', 'BHE')""",
        'patiendid_col': 'clientid'
    },
    'midwest': {
        'censusactions': """censusactioncode in ('DAMA', 'DD', 'DE', 'DH', 'HP', 'HUP', 'L', 
                        'LOA', 'LV', 'MHHP', 'MHHU', 'MHTP', 'MHTU', 'MO', 'RDD', 'RDE', 'RDH', 'TO', 'TP', 'TUP')""",
        'patiendid_col': 'clientid'
    },
    'mmh': {
        'censusactions':"""censusactioncode in ('DD', 'DE', 'TO', 'RDD', 'RDE', 'RDH', 'DH', 'HP', 'TP', 'L', 
                        'DAMA', 'LSNF', 'TOSNF', 'HUP', 'TUP', 'HL', 'HUM')""",
        'patiendid_col': 'clientid'
    },
    'trio': {
        'censusactions':"""censusactioncode in ('DAMA', 'DD', 'DE', 'DH', 'E', 'HU', 'L', 'LV', 'MO', 'TO', 'TP', 'TU')""",
        'patiendid_col': 'patientid'
    },
    'uch': {
        'censusactions':"""censusactioncode in ('AD', 'DD', 'DE', 'TO', 'RDD', 'RDE', 'RDH', 'DH', 
                        'H', 'L', 'T', 'H8', 'T8', 'DP', 'HB', 'TP', 'EP', 'DTH30', 'TTH30', 'HLC', 
                        'CL', 'HHU', 'TLU', 'H>8N', 'BHRC')""",
        'patiendid_col': 'clientid'
    },
    'vintage': {
        'censusactions':"""censusactioncode in ('BH', 'BHR', 'DA', 'DC', 'DD', 'DE', 'DH', 'DL', 'HL', 'L', 
                        'LOA', 'RDD', 'RDE', 'RDH', 'TL', 'TLU', 'TO', 'TUP')""",
        'patiendid_col': 'clientid'
    },
    'hsm': {
        'censusactions':"""censusactioncode in ('DD', 'DE', 'TO', 'RDD', 'RDE', 'RDH', 'DH', 'HP', 'TP', 'LOA', 
                    'DP', 'EP', 'HUP', 'TUP', 'RR', 'RR ', 'DAMA', 'DNV', 'UHL', 'UTL', 'HD', 'IS', 'RRl', 'RRU')""",
        'patiendid_col': 'clientid'
    }
}


def get_raw_sql(file_name):
    """Returns a raw-sql statement with the given file_name from the same directory"""
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, 'sql', file_name)
    data_file = open(file_path, 'r')
    return data_file.read()


def migrate_daily_predictions():

    with start_task(action_type="migrate_daily_predictions"):
        start_time = timeit.default_timer()
        session = boto3.session.Session()
        secrets_client = session.client(
            service_name="secretsmanager", region_name=region_name
        )
        with start_action(action_type="get_saiva_engine"):
            saiva_db_info = json.loads(
                secrets_client.get_secret_value(SecretId=f"{env}-saivadb")[
                    "SecretString"
                ]
            )
            saiva_url = URL(
                drivername="postgresql",
                username=saiva_db_info["username"],
                password=saiva_db_info["password"],
                host=saiva_db_info["host"],
                port=saiva_db_info["port"],
                database=saiva_db_info["dbname"],
            )
            saiva_engine = create_engine(saiva_url, echo=False)
        log_message(
            message_type='info', task='create_unranked_census'
        )
        with saiva_engine.connect().execution_options(autocommit=True) as conn:
            query = """
            CREATE TABLE "public"."unranked_census" (
            "masterpatientid" int4 NOT NULL,
            "facilityid" int4 NOT NULL,
            "bedid" int4,
            "censusdate" timestamp NOT NULL,
            "predictionvalue" numeric(10,8),
            "predictionrank" int8,
            "modelid" bpchar(300),
            "createdat" timestamptz NOT NULL DEFAULT now(),
            "updatedat" timestamptz NOT NULL DEFAULT now(),
            "client" varchar(200) NOT NULL DEFAULT 'unspecified'::character varying,
            "published" bool DEFAULT true,
            "experiment_group" bool NOT NULL DEFAULT true,
            "group_rank" int8,
            "show_in_report" bool NOT NULL DEFAULT false,
            "group_level" varchar(60) DEFAULT 'facility'::character varying,
            "group_id" varchar(60),
            "censusactioncode" varchar(10),
            "payername" varchar(50),
            "payercode" varchar(10),
            "admissionstatus" varchar(100),
            "to_from_type" varchar(250)
            );"""
            conn.execute(query)
        
        with start_action(action_type="get_unranked_census_and_census_payer_admission_data"):  
            log_message(
                message_type='info', task='load_prediction_start_dates'
            )
            min_dates_df = pd.read_sql("SELECT client, min(censusdate) as date from daily_predictions GROUP by client", saiva_engine)
            # Iterate through each client an insert the missing rows and fetch the missing columns 
            for key in client_config:
                _start_date = min_dates_df[min_dates_df.client == key]['date'].iloc[0].strftime('%Y-%m-%d')
                if key == "avante":
                    client_db_info = json.loads(
                        secrets_client.get_secret_value(SecretId="avantedb")["SecretString"]
                    )
                else:    
                    client_db_info = json.loads(
                        secrets_client.get_secret_value(SecretId=f"{env}-sqlserver")[
                            "SecretString"
                        ]
                    )
                client_url = URL(
                    drivername="mssql+pyodbc",
                    username=client_db_info["username"],
                    password=client_db_info["password"],
                    host=client_db_info["host"],
                    port=client_db_info["port"],
                    database=key if key != 'avante' else client_db_info['dbname'],
                    query={"driver": "ODBC Driver 17 for SQL Server"},
                )
                client_engine = create_engine(client_url, echo=False)
                query = get_raw_sql('unranked_census.sql').format(
                    client=key,
                    censusactioncode_filter=client_config[key]['censusactions'],
                    patiendid_col=client_config[key]['patiendid_col'],
                    start_date=_start_date
                )
                log_message(
                    message_type='info', task='get_unranked_census', client=key, query=query
                )
                df = pd.read_sql(query, client_engine)
                log_message(
                        message_type='info', task='load_unranked_census', client=key, shape=df.shape
                )
                # Retrieves unranked census for each client and load it into daily predictions table
                df.to_sql(
                    'unranked_census',
                    saiva_engine,
                    method='multi', 
                    if_exists='append',
                    index=False,
                    chunksize=1000
                )
                # Query sql server to get the missing columns from admission and daily_census table
                query = get_raw_sql('census_payer_admission.sql').format(
                    client=key,
                    censusactioncode_filter=client_config[key]['censusactions'],
                    patiendid_col=client_config[key]['patiendid_col'],
                    start_date=_start_date
                )
                log_message(
                    message_type='info', task='get_census_payer_admission_data', client=key, query=query
                )
                df = pd.read_sql(query, client_engine)
                log_message(
                    message_type='info', task='load_census_payer_admission_data', client=key, shape=df.shape
                ) 
                # Load missing columns into a temp table temp_prediction_census_payer_admission_data
                df.to_sql(
                    'temp_prediction_census_payer_admission_data',
                    saiva_engine,
                    method='multi', 
                    if_exists='append',
                    index=False,
                    chunksize=1000
                )
        
        with saiva_engine.connect().execution_options(autocommit=True) as conn:
            # Dedup unranked_census using daily_predictions and merge daily_predictions into unranked_census
            with start_action(action_type="dedup_unranked_census"):
                query = """
                DELETE
                FROM
                	unranked_census uc
                WHERE
                	EXISTS (
                		SELECT
                			censusdate
                		FROM
                			daily_predictions dp
                		WHERE
                			dp.client = uc.client
                			AND dp.facilityid = uc.facilityid
                			AND dp.masterpatientid = uc.masterpatientid
                			AND dp.censusdate = uc.censusdate)
                """
                conn.execute(query)

            # Delete rows from unranked census if there were no predications for the client/facilityid/censusdate combination
            with start_action(action_type="delete_extra_rows"):
                query = """
                DELETE
                FROM
                	unranked_census uc
                WHERE
                	NOT EXISTS (
                		SELECT
                			censusdate
                		FROM
                			daily_predictions dp
                		WHERE
                			dp.client = uc.client
                			AND dp.facilityid = uc.facilityid
                			AND dp.censusdate = uc.censusdate)
                """
                conn.execute(query)    
            
            with start_action(action_type="merge_daily_predictions_to_unranked_census"):
                query = """
                INSERT INTO unranked_census (
                    masterpatientid, facilityid, bedid, censusdate, predictionvalue, predictionrank, modelid, createdat, updatedat, client, published, experiment_group, 
                    group_rank, show_in_report, group_level, group_id, censusactioncode, payername, payercode, admissionstatus, to_from_type
                )
                SELECT masterpatientid, facilityid, bedid, censusdate, predictionvalue, predictionrank, modelid, createdat, updatedat, client, published, experiment_group, 
                    group_rank, show_in_report, group_level, group_id, censusactioncode, payername, payercode, admissionstatus, to_from_type from daily_predictions
                """
                conn.execute(query)    


        with saiva_engine.connect().execution_options(autocommit=True) as conn:
            # Join daily_predictions with the temp_prediction_census_payer_admission_data created above and insert
            # into a temp table which should become a copy of the daily_predictions with the extra columns
            # TODO: After creating the temp_daily_predictions table, verify that the data is correct, apply the
            # triggers and indices to the new table and deprecate the old daily predictions table
            with start_action(action_type="join_daily_predictions_with_payer_admission"):
                query = """
                SELECT
                	dp.masterpatientid,
                	dp.facilityid,
                	dp.bedid,
                	dp.censusdate,
                	dp.predictionvalue,
                	dp.predictionrank,
                	dp.modelid,
                	dp.createdat,
                	dp.updatedat,
                	dp.client,
                	dp.published,
                	dp.experiment_group,
                	dp.group_rank,
                	dp.show_in_report,
                	dp.group_level,
                	dp.group_id,
                	tp.censusactioncode,
                	tp.payername,
                	tp.payercode,
                	tp.admissionstatus,
                	tp.to_from_type INTO temp_daily_predictions
                FROM
                	unranked_census dp
                	LEFT JOIN temp_prediction_census_payer_admission_data tp ON dp.client = tp.client
                		AND dp.facilityid = tp.facilityid
                		AND dp.masterpatientid = tp.masterpatientid
                		AND dp.censusdate = tp.censusdate
                """
                conn.execute(query)
            
            with start_action(action_type="create_indices_and_trigger"):    
                idx1 = """
                CREATE UNIQUE INDEX temp_daily_predictions_client_modelid_facid_maspatid_date_key ON public.temp_daily_predictions 
                USING btree (client, modelid, facilityid, masterpatientid, censusdate);
                """
                conn.execute(idx1)

                idx2 = """CREATE UNIQUE INDEX temp_daily_predictions_client_facid_maspatid_date_idx ON public.temp_daily_predictions 
                USING btree (client, facilityid, masterpatientid, censusdate) WHERE (modelid IS NULL);"""
                conn.execute(idx2)

                idx3 = """CREATE INDEX fki_temp_daily_predictions_facilityid_fkey ON public.temp_daily_predictions USING btree (facilityid);"""
                conn.execute(idx3)

                idx4 = "CREATE INDEX idx_temp_daily_predictions_censusdate ON public.temp_daily_predictions USING btree (censusdate);"
                conn.execute(idx4)

                idx5 = """CREATE INDEX idx_temp_daily_predictions_facid_censusdate_modelid_predictionv ON public.temp_daily_predictions
                USING btree (facilityid, censusdate, modelid, predictionvalue DESC);"""
                conn.execute(idx5)

                trigger = """create trigger set_timestamp before update on
                            public.temp_daily_predictions for each row execute procedure trigger_set_timestamp();"""
                conn.execute(trigger)                            

                
            with start_action(action_type="drop_temp_table"):    
                conn.execute('drop table temp_prediction_census_payer_admission_data')

        elapsed = timeit.default_timer() - start_time
        log_message(message_type="info", message=f'TOTAL TIME TAKEN FOR THE RUN {elapsed}')  
                  
            
        


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes. --client passed as command line argument
    fire.Fire(migrate_daily_predictions)
