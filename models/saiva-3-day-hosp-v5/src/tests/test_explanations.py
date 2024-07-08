"""
aws s3 sync s3://saiva-models/99/d0c497c8b9b04f4d9e1e1e0c9297cc1f /data/models/d0c497c8b9b04f4d9e1e1e0c9297cc1f
cd src
python tests/test_explanations.py
"""

import sys

sys.path.insert(0, '/src')
import pickle
import pandas as pd
from shared.database import DbEngine
from run_explanation import generate_explanations
from eliot import to_file, log_message
from data_models import BaseModel
import subprocess


to_file(open("eliot.log", "w"))


prediction_date = '2021-02-02'
client = 'avante'
facilityid = '1'
s3_path = f's3://saiva-dev-data-bucket/unit_test_data/explanations/{client}/{prediction_date}'
modelid = 'd0c497c8b9b04f4d9e1e1e0c9297cc1f'

table_list = ['master_patient_lookup', 'patient_census', 'patient_rehosps','patient_room_details',
              'patient_progress_notes', 'patient_diagnosis', 'patient_vitals', 'patient_lab_results',
              'patient_meds', 'patient_orders', 'patient_alerts', 'patient_demographics']
raw_data_dict = {}
for table in table_list:
    print(f"reading {table}")
    raw_data_dict[table] = pd.read_parquet(
        f"{s3_path}/{table}.parquet"
    )


# log_message(message_type='info', message='Downloading Prediction Models...')    
# subprocess.run(
#     f'aws s3 sync s3://saiva-models/99/{modelid} /data/models/{modelid}',
#     shell=True,
#     stderr=subprocess.DEVNULL,
#     stdout=subprocess.DEVNULL,
# )  

with open(f"/data/models/{modelid}/artifacts/{modelid}.pickle", "rb") as f:
    model = pickle.load(f)

pd_final_df = pd.read_parquet(f"{s3_path}/pd_final_df.parquet")
pd_final_idens = pd.read_parquet(f"{s3_path}/pd_final_idens.parquet")

engine = DbEngine()
saiva_engine = engine.get_postgresdb_engine()

generate_explanations(
    prediction_date,
    model,
    pd_final_df,
    pd_final_idens,
    raw_data_dict,
    client,
    facilityid,
    saiva_engine,
)
