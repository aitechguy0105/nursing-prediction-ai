import typing
import logging
from dataclasses import dataclass
import datetime
from pathlib import Path


# STEP 1

import sys
import pandas as pd
from saiva.shared.load_raw_data import fetch_training_data, fetch_training_cache_data, validate_dataset, remove_invalid_datevalues, unroll_patient_census
from saiva.shared.database import DbEngine
from saiva.shared.utils import get_client_class
from saiva.shared.utils import get_memory_usage
from saiva.shared.constants import CLIENT
from saiva.clients.base import BaseClient
from eliot import to_file

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class TrainingArguments:
    client: str
    train_start_date: datetime.date
    test_end_date: datetime.date


args = TrainingArguments(
    client="avante",
    train_start_date=datetime.date(2020, 7, 1),
    test_end_date=datetime.date(2022, 12, 31)
)
S3_BUCKET = 'saiva-dev-data-bucket'

ClientClass = get_client_class(client=args.client)

# Connect to DB and fetch data
engine = DbEngine()
saiva_engine = engine.get_postgresdb_engine()
client_sql_engine = engine.get_sqldb_engine(clientdb_name=args.client)

# verify connectivity
engine.verify_connectivity(client_sql_engine)

### ======================== Fetch Data ============================
clientInstance: BaseClient = ClientClass()
training_queries = clientInstance.get_training_queries(args.test_end_date, args.train_start_date)

def save_to_disk(*, df: pd.DataFrame, dataset_name: str) -> str:
    filename = f'data/{dataset_name}.parquet'
    df.to_parquet(filename)
    return filename

def load_from_disk(*, dataset_name: str) -> pd.DataFrame:
    return pd.read_parquet(f'data/{dataset_name}.parquet')


dataset_to_file_map = {}

# fetch data for training and dump it to disk
for dataset_name, query in training_queries.items():
    log.info(f"Fetching data for dataset {dataset_name}")
    df = pd.read_sql(query, con=client_sql_engine)

    log.info(f"Total records fetched for dataset {dataset_name} is {len(df)}")

    validate_dataset(dataset_name, df)
    df = remove_invalid_datevalues(df)

    if dataset_name == 'patient_census':
        df = unroll_patient_census(df, args.train_start_date, args.test_end_date)

    # TODO: should do client transformation here, but no client has it implemented yet, so skipping for now

    filename = save_to_disk(df=df, dataset_name=dataset_name)
    dataset_to_file_map[dataset_name] = filename


# Join some datasets with master_patient_lookup
@dataclass
class JoinMasterPatientLookupConfig:
    merge_on: typing.List[str]
    column_subset: typing.Optional[typing.List[str]] = None

join_config = {
    'patient_census': JoinMasterPatientLookupConfig(merge_on=["patientid", "facilityid"]),
    'patient_rehosps': JoinMasterPatientLookupConfig(merge_on=["patientid", "facilityid"], column_subset=['facilityid', 'patientid', 'masterpatientid']),
    'patient_admissions': JoinMasterPatientLookupConfig(merge_on=["patientid", "facilityid"], column_subset=['facilityid', 'patientid', 'masterpatientid']),
    'patient_diagnosis': JoinMasterPatientLookupConfig(merge_on=["patientid", "facilityid"], column_subset=['facilityid', 'patientid', 'masterpatientid']),
    'patient_vitals': JoinMasterPatientLookupConfig(merge_on=["patientid", "facilityid"], column_subset=['facilityid', 'patientid', 'masterpatientid']),
    'patient_meds': JoinMasterPatientLookupConfig(merge_on=["patientid", "facilityid"], column_subset=['facilityid', 'patientid', 'masterpatientid']),
    'patient_orders': JoinMasterPatientLookupConfig(merge_on=["patientid", "facilityid"], column_subset=['facilityid', 'patientid', 'masterpatientid']),
    'patient_alerts': JoinMasterPatientLookupConfig(merge_on=["patientid", "facilityid"], column_subset=['facilityid', 'patientid', 'masterpatientid']),
    'patient_immunizations': JoinMasterPatientLookupConfig(merge_on=["patientid", "facilityid"], column_subset=['facilityid', 'patientid', 'masterpatientid']),
    'patient_risks': JoinMasterPatientLookupConfig(merge_on=["patientid", "facilityid"], column_subset=['facilityid', 'patientid', 'masterpatientid']),
    'patient_assessments': JoinMasterPatientLookupConfig(merge_on=["patientid", "facilityid"], column_subset=['facilityid', 'patientid', 'masterpatientid']),
    'patient_adt': JoinMasterPatientLookupConfig(merge_on=["patientid", "facilityid"], column_subset=['facilityid', 'patientid', 'masterpatientid']),
    'patient_progress_notes': JoinMasterPatientLookupConfig(merge_on=["patientid", "facilityid"], column_subset=['facilityid', 'patientid', 'masterpatientid']),
    'patient_lab_results': JoinMasterPatientLookupConfig(merge_on=["patientid", "facilityid"], column_subset=['facilityid', 'patientid', 'masterpatientid']),
}

master_patient_lookup = load_from_disk(dataset_name='master_patient_lookup')
for dataset_name, filename in dataset_to_file_map.items():
    if dataset_name == 'master_patient_lookup':
        continue

    log.info(f"Joining dataset {dataset_name} with master_patient_lookup")
    df = load_from_disk(dataset_name=dataset_name)
    join_config = join_config[dataset_name]
    if df.empty:
        log.info(f"Skipping join for dataset {dataset_name} as it is empty")
        continue

    if join_config.column_subset:
        df = df.merge(master_patient_lookup[join_config.column_subset], on=join_config.merge_on)
    else:
        df = df.merge(master_patient_lookup, on=join_config.merge_on)

    if dataset_name == 'patient_demographics':
        df['dateofbirth'] = pd.to_datetime(df['dateofbirth'], errors='coerce')

    # TODO: we should tweak start and end dates for queries so get less than 15k rows for master_patient_lookup
    # have a max of 15042 master_patient_lookup rows ie. Infinity-Infinity

    filename = save_to_disk(df=df, dataset_name=dataset_name)
    dataset_to_file_map[dataset_name] = filename

master_patient_lookup = None


for dataset_name, filename in dataset_to_file_map.items():
    df = load_from_disk(dataset_name=dataset_name)
    log.info(f"Dataset {dataset_name}, shape: {df.shape}")

# Note: merging multiple clients is dropped from automatic training


### test + validation set is 25%
## Note: Obtained training, test, validation dates needs to be added in client respective file under `get_experiment_dates` function!!

def get_prior_date_as_str(date_as_str):
    prior_date = pd.to_datetime(date_as_str) - timedelta(days=1)
    prior_date_as_str = prior_date.date().strftime('%Y-%m-%d')
    return prior_date_as_str

# do cleaning
patient_census_df = load_from_disk(dataset_name='patient_census')

patient_census_df.drop_duplicates(
    subset=['masterpatientid', 'censusdate'],
    keep='last',
    inplace=True
)
patient_census_df.sort_values(by=['censusdate'], inplace=True)
# done with cleaning

# get stats
total_count = patient_census_df.shape[0]
test_count = int((total_count * 25) / 100)
test_split_count = int((test_count * 50) / 100) # split between validation & test set

# build test dataframe
test_patient_census_df = patient_census_df.tail(test_count) # cut last n rows
validation_patient_census_df = test_patient_census_df.head(test_split_count)
test_patient_census_df = test_patient_census_df.tail(test_split_count)

train_start_date = test_patient_census_df.censusdate.min().date().strftime('%Y-%m-%d')
validation_start_date = validation_patient_census_df.censusdate.min().date().strftime('%Y-%m-%d')
test_start_date = test_patient_census_df.censusdate.min().date().strftime('%Y-%m-%d')
test_end_date = test_patient_census_df.censusdate.max().date().strftime('%Y-%m-%d')

test_patient_census_df = None
validation_patient_census_df = None
patient_census_df = None

train_end_date = get_prior_date_as_str(validation_start_date)
validation_end_date = get_prior_date_as_str(test_start_date)

@dataclass
class ExperimentDates:
    train_start_date: datetime.date
    train_end_date: datetime.date
    validation_start_date: datetime.date
    validation_end_date: datetime.date
    test_start_date: datetime.date
    test_end_date: datetime.date

log.info(f'train_start_date: {train_start_date}')
log.info(f'train_end_date: {train_end_date}')
log.info(f'validation_start_date: {validation_start_date}')
log.info(f'validation_end_date: {validation_end_date}')
log.info(f'test_start_date: {test_start_date}')
log.info(f'test_end_date: {test_end_date}')

experiment_dates = ExperimentDates(
    train_start_date=train_start_date,
    train_end_date=train_end_date,
    validation_start_date=validation_start_date,
    validation_end_date=validation_end_date,
    test_start_date=test_start_date,
    test_end_date=test_end_date
)

log.info(f'Experiment dates: {experiment_dates}')

# TODO: not sure why next block is needed, is it just for visual check? Do we actually need to have this?

### ======================== TESTING ================================

# Load generic named Training data which is cached in local folders
# from shared.load_raw_data import fetch_training_cache_data

# result_dict = fetch_training_cache_data(client=CLIENT, generic=True)
# for key, value in result_dict.items():
#     print(f'{key} : {result_dict[key].info()}')

# Remove all newly generated parquet files

# for ft in client_file_types:
#     os.remove(data_path/f'{ft}.parquet')


# feature engineering

from saiva.shared.demographics import DemographicFeatures
from saiva.shared.labs import LabFeatures
from saiva.shared.meds import MedFeatures
from saiva.shared.orders import OrderFeatures
from saiva.shared.vitals import VitalFeatures
from saiva.shared.alerts import AlertFeatures
from saiva.shared.rehosp import RehospFeatures
from saiva.shared.notes import NoteFeatures
from saiva.shared.diagnosis import DiagnosisFeatures
from saiva.shared.patient_census import PatientCensus
from saiva.shared.admissions import AdmissionFeatures
from saiva.shared.immunizations import ImmunizationFeatures
from saiva.shared.risks import RiskFeatures
from saiva.shared.assessments import AssessmentFeatures
from saiva.shared.adt import AdtFeatures

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


def featurize_patient_census():
    patient_census_df = load_from_disk(dataset_name='patient_census')
    featurizer = PatientCensus(
        census_df=patient_census_df,
        train_start_date=experiment_dates.train_start_date,
        train_end_date=experiment_dates.train_end_date,
    )
    featurized_patient_census_df = featurizer.generate_features()
    save_to_disk(featurized_patient_census_df, dataset_name='featurized_patient_census')


def featurize_demographics():
    featurized_patient_census_df = load_from_disk(dataset_name='featurized_patient_census')
    patient_demographics_df = load_from_disk(dataset_name='patient_demographics')

    featurizer = DemographicFeatures(
        census_df=featurized_patient_census_df,
        demo_df=patient_demographics_df,
        training=True,
    )
    featurized_patient_demographics = featurizer.generate_features()
    save_to_disk(featurized_patient_demographics, dataset_name='featurized_patient_demographics')


def featurize_vitals():
    featurized_patient_census_df = load_from_disk(dataset_name='featurized_patient_census')
    patient_vitals_df = load_from_disk(dataset_name='patient_vitals')

    featurizer = VitalFeatures(
        census_df=featurized_patient_census_df,
        vitals=patient_vitals_df,
        training=True
    )
    featurized_patient_vitals_df = featurizer.generate_features()
    save_to_disk(featurized_patient_vitals_df, dataset_name='featurized_patient_vitals')


def featurize_orders():
    featurized_patient_census_df = load_from_disk(dataset_name='featurized_patient_census')
    patient_orders_df = load_from_disk(dataset_name='patient_orders')

    featurizer = OrderFeatures(
        census_df=featurized_patient_census_df,
        orders=patient_orders_df,
        training=True
    )
    featurized_patient_orders_df = featurizer.generate_features()
    save_to_disk(featurized_patient_orders_df, dataset_name='featurized_patient_orders')


def featurize_meds():
    featurized_patient_census_df = load_from_disk(dataset_name='featurized_patient_census')
    patient_meds_df = load_from_disk(dataset_name='patient_meds')

    featurizer = MedFeatures(
        census_df=featurized_patient_census_df,
        meds=patient_meds_df,
        training=True
    )

    featurized_patient_meds_df, updated_patient_meds_df = featurizer.generate_features()
    save_to_disk(featurized_patient_meds_df, dataset_name='featurized_patient_meds')
    save_to_disk(updated_patient_meds_df, dataset_name='updated_patient_meds')


def featurize_alerts():
    featurized_patient_census_df = load_from_disk(dataset_name='featurized_patient_census')
    patient_alerts_df = load_from_disk(dataset_name='patient_alerts')

    featurizer = AlertFeatures(
        census_df=featurized_patient_census_df,
        alerts=patient_alerts_df,
        training=True
    )
    featurized_patient_alerts_df = featurizer.generate_features()
    save_to_disk(featurized_patient_alerts_df, dataset_name='featurized_patient_alerts')


def featurize_lab_results():
    # TODO: patient_lab_results might be empty
    featurized_patient_census_df = load_from_disk(dataset_name='featurized_patient_census')
    patient_lab_results_df = load_from_disk(dataset_name='patient_lab_results')

    featurizer = LabFeatures(
        census_df=featurized_patient_census_df,
        labs=patient_lab_results_df,
        training=True
    )
    featurized_patient_lab_results_df = featurizer.generate_features()
    save_to_disk(featurized_patient_lab_results_df, dataset_name='featurized_patient_lab_results')


def featurize_patient_rehosps():
    featurized_patient_census_df = load_from_disk(dataset_name='featurized_patient_census')
    patient_rehosps_df = load_from_disk(dataset_name='patient_rehosps')
    patient_adt_df = load_from_disk(dataset_name='patient_adt')

    featurizer = RehospFeatures(
        census_df=featurized_patient_census_df,
        rehosps=patient_rehosps_df,
        adt_df=patient_adt_df,
        training=True
    )
    featurized_patient_rehosps_df = featurizer.generate_features()
    save_to_disk(featurized_patient_rehosps_df, dataset_name='featurized_patient_rehosps')


def featurize_patient_admissions():
    featurized_patient_census_df = load_from_disk(dataset_name='featurized_patient_census')
    patient_admissions_df = load_from_disk(dataset_name='patient_admissions')

    featurizer = AdmissionFeatures(
        census_df=featurized_patient_census_df,
        admissions=patient_admissions_df,
        training=True
    )
    featurized_patient_admissions_df = featurizer.generate_features()
    save_to_disk(featurized_patient_admissions_df, dataset_name='featurized_patient_admissions')


def featurize_patient_diagnosis(*, s3_bucket: str):
    # Note: requires file: data/lookup/DXCCSR_v2022-1.csv to be in s3 bucket
    featurized_patient_census_df = load_from_disk(dataset_name='featurized_patient_census')
    patient_diagnosis_df = load_from_disk(dataset_name='patient_diagnosis')

    featurizer = DiagnosisFeatures(
        census_df=featurized_patient_census_df,
        diagnosis=patient_diagnosis_df,
        s3_bucket=s3_bucket,
        training=True
    )
    featurized_patient_diagnosis_df, updated_patient_diagnosis = featurizer.generate_features()
    save_to_disk(featurized_patient_diagnosis_df, dataset_name='featurized_patient_diagnosis')
    save_to_disk(updated_patient_diagnosis, dataset_name='updated_patient_diagnosis')


def featurize_patient_progress_notes(*, client: str):
    # TODO: patient_progress_notes might be empty
    featurized_patient_census_df = load_from_disk(dataset_name='featurized_patient_census')
    patient_progress_notes_df = load_from_disk(dataset_name='patient_progress_notes')

    featurizer = NoteFeatures(
        census_df=featurized_patient_census_df,
        notes=patient_progress_notes_df,
        client=client,
        training=True
    )

    featurized_patient_progress_notes_df = featurizer.generate_features()
    save_to_disk(featurized_patient_progress_notes_df, dataset_name='featurized_patient_progress_notes')


def featurize_patient_immunizations():
    featurized_patient_census_df = load_from_disk(dataset_name='featurized_patient_census')
    patient_immunizations_df = load_from_disk(dataset_name='patient_immunizations')

    featurizer = ImmunizationFeatures(
        census_df=featurized_patient_census_df,
        immuns_df=patient_immunizations_df,
        training=True
    )
    featurized_patient_immunizations_df = featurizer.generate_features()
    save_to_disk(featurized_patient_immunizations_df, dataset_name='featurized_patient_immunizations')


def featurize_patient_risks():
    featurized_patient_census_df = load_from_disk(dataset_name='featurized_patient_census')
    patient_risks_df = load_from_disk(dataset_name='patient_risks')

    featurizer = RiskFeatures(
        census_df=featurized_patient_census_df,
        risks_df=patient_risks_df,
        training=True
    )
    featurized_patient_risks_df = featurizer.generate_features()
    save_to_disk(featurized_patient_risks_df, dataset_name='featurized_patient_risks')


def featurize_patient_assessments():
    featurized_patient_census_df = load_from_disk(dataset_name='featurized_patient_census')
    patient_assessments_df = load_from_disk(dataset_name='patient_assessments')

    featurizer = AssessmentFeatures(
        census_df=featurized_patient_census_df,
        assessments_df=patient_assessments_df,
        training=True
    )
    featurized_patient_assessments_df = featurizer.generate_features()
    save_to_disk(featurized_patient_assessments_df, dataset_name='featurized_patient_assessments')


def featurize_patient_adt():
    featurized_patient_census_df = load_from_disk(dataset_name='featurized_patient_census')
    patient_adt_df = load_from_disk(dataset_name='patient_adt')

    featurizer = AdtFeatures(
        census_df=featurized_patient_census_df,
        adt_df=patient_adt_df,
        training=True
    )
    featurized_patient_adt_df = featurizer.generate_features()
    save_to_disk(featurized_patient_adt_df, dataset_name='featurized_patient_adt')


featurize_patient_census()
featurize_demographics()
featurize_vitals()
featurize_orders()
featurize_meds()
featurize_alerts()
featurize_lab_results()
featurize_patient_rehosps()
featurize_patient_admissions()
featurize_patient_diagnosis(s3_bucket=S3_BUCKET)
featurize_patient_progress_notes(client=args.client)
featurize_patient_immunizations()
featurize_patient_risks()
featurize_patient_assessments()
featurize_patient_adt()

# end feature engineering

# STEP 4

import json
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, '/src')
from eliot import start_action, start_task, to_file, log_message
from shared.utils import get_client_class, get_memory_usage
to_file(sys.stdout)

processed_path = Path('/data/processed')

%%time
demo_df = pd.read_parquet(processed_path/'demo_df.parquet')
vitals_df = pd.read_parquet(processed_path/'vitals_df.parquet')
orders_df = pd.read_parquet(processed_path/'orders_df.parquet')
alerts_df = pd.read_parquet(processed_path/'alerts_df.parquet')
meds_df = pd.read_parquet(processed_path/'meds_df.parquet')
rehosp_df = pd.read_parquet(processed_path/'rehosp_df.parquet')
admissions_df = pd.read_parquet(processed_path/'admissions_df.parquet')
diagnosis_df = pd.read_parquet(processed_path/'diagnosis_df.parquet')

%%time

final_df = demo_df.merge(
    vitals_df,
    how='left',
    left_on=['masterpatientid', 'facilityid', 'censusdate'],
    right_on=['masterpatientid', 'facilityid', 'censusdate']
)
final_df = final_df.merge(
    orders_df,
    how='left',
    left_on=['masterpatientid', 'facilityid', 'censusdate'],
    right_on=['masterpatientid', 'facilityid', 'censusdate']
)
final_df = final_df.merge(
    rehosp_df,
    how='left',
    left_on=['masterpatientid', 'facilityid', 'censusdate'],
    right_on=['masterpatientid', 'facilityid', 'censusdate']
)
final_df = final_df.merge(
    admissions_df,
    how='left',
    left_on=['masterpatientid', 'facilityid', 'censusdate'],
    right_on=['masterpatientid', 'facilityid', 'censusdate']
)
final_df = final_df.merge(
    meds_df,
    how='left',
    left_on=['masterpatientid', 'facilityid', 'censusdate'],
    right_on=['masterpatientid', 'facilityid', 'censusdate']
)
final_df = final_df.merge(
    alerts_df,
    how='left',
    left_on=['masterpatientid', 'facilityid', 'censusdate'],
    right_on=['masterpatientid', 'facilityid', 'censusdate']
)
final_df = final_df.merge(
    diagnosis_df,
    how='left',
    left_on=['masterpatientid', 'facilityid', 'censusdate'],
    right_on=['masterpatientid', 'facilityid', 'censusdate']
)

%%time
if Path.exists(processed_path/'labs_df.parquet'):
    labs_df = pd.read_parquet(processed_path/'labs_df.parquet')
    final_df = final_df.merge(
        labs_df,
        how='left',
        left_on=['masterpatientid', 'facilityid', 'censusdate'],
        right_on=['masterpatientid', 'facilityid', 'censusdate']
    )

%%time
if Path.exists(processed_path/'notes_df.parquet'):
    notes_df = pd.read_parquet(processed_path/'notes_df.parquet')
    final_df = final_df.merge(
        notes_df,
        how='left',
        left_on=['masterpatientid', 'facilityid', 'censusdate'],
        right_on=['masterpatientid', 'facilityid', 'censusdate']
    )

%%time
if Path.exists(processed_path/'immuns_df.parquet'):
    immuns_df = pd.read_parquet(processed_path/'immuns_df.parquet')
    final_df = final_df.merge(
        immuns_df,
        how='left',
        left_on=['masterpatientid', 'facilityid', 'censusdate'],
        right_on=['masterpatientid', 'facilityid', 'censusdate']
    )


%%time
if Path.exists(processed_path/'risks_df.parquet'):
    risks_df = pd.read_parquet(processed_path/'risks_df.parquet')
    final_df = final_df.merge(
        risks_df,
        how='left',
        left_on=['masterpatientid', 'facilityid', 'censusdate'],
        right_on=['masterpatientid', 'facilityid', 'censusdate']
    )

%%time
if Path.exists(processed_path/'assessments_df.parquet'):
    assessments_df = pd.read_parquet(processed_path/'assessments_df.parquet')
    final_df = final_df.merge(
        assessments_df,
        how='left',
        left_on=['masterpatientid', 'facilityid', 'censusdate'],
        right_on=['masterpatientid', 'facilityid', 'censusdate']
    )

%%time
if Path.exists(processed_path/'adt_df.parquet'):
    adt_df = pd.read_parquet(processed_path/'adt_df.parquet')    
    final_df = final_df.merge(
        adt_df,
        how='left',
        left_on=['masterpatientid', 'facilityid', 'censusdate'],
        right_on=['masterpatientid', 'facilityid', 'censusdate']
    )

%%time
rename_cols = {'count_allergy_x':'count_allergy',
               'day_since_bed_change_x':'day_since_bed_change',
               'client_x':'client',
               'date_of_transfer_x': 'date_of_transfer',
               'payername_x':'payername'
              }

final_df = final_df.rename(columns=rename_cols)

%%time
# drop unwanted columns
final_df.drop(
    final_df.columns[final_df.columns.str.contains('_masterpatientid|_facilityid|_x$|_y$')].tolist()
, axis=1, inplace = True)

if 'patientid' in final_df.columns:
    # If patientid included in the above regex pattern it drops masterpatientid column even
    final_df.drop(
        ['patientid'],
        axis=1,
        inplace=True
    )
#drop duplicated columns
print('Number of columns in the dataframe:', final_df.shape[1])
print('Number of duplicated columns:', sum(final_df.columns.duplicated()))
final_df = final_df.loc[:,~final_df.columns.duplicated()]
print('Number of columns in the dataframe after dropping duplcated columns:', final_df.shape[1])

%%time
# Write to new parquet file
final_df.to_parquet(processed_path/'final_df.parquet')

print(get_memory_usage(final_df))

final_df.shape

nan_cols = [i for i in final_df.columns if final_df[i].isna().any() and 'e_' not in i]
nan_cols

len(nan_cols)

## Generate Feature Group dictionary

exclude_columns = ['masterpatientid', 'facilityid', 'censusdate', 'client', 'date_of_transfer', 'na_indictator_date_of_transfer',]

vitals_features = [x for x in vitals_df.columns if x not in exclude_columns] 
orders_features = [x for x in orders_df.columns if x not in exclude_columns]
rehosp_features = [x for x in rehosp_df.columns if x not in exclude_columns]
admissions_features = [x for x in admissions_df.columns if x not in exclude_columns]
meds_features = [x for x in meds_df.columns if x not in exclude_columns]
alerts_features = [x for x in alerts_df.columns if x not in exclude_columns]
diagnosis_features = [x for x in diagnosis_df.columns if x not in exclude_columns]
demo_features = [x for x in demo_df.columns if x not in exclude_columns]

feature_groups = {
    'Vitals': vitals_features,
    'Orders': orders_features,
    'Transfers': rehosp_features,
    'Admissions': admissions_features,
    'Medications': meds_features,
    'Alerts': alerts_features,
    'Diagnoses': diagnosis_features,
    'Demographics': demo_features,
}
if Path.exists(processed_path/'labs_df.parquet'):
    feature_groups['Labs'] = [x for x in labs_df.columns if x not in exclude_columns]

if Path.exists(processed_path/'notes_df.parquet'):
    feature_groups['ProgressNotes'] = [x for x in notes_df.columns if x not in exclude_columns]

if Path.exists(processed_path/'immuns_df.parquet'):
    feature_groups['Immunizations'] = [x for x in immuns_df.columns if x not in exclude_columns]

if Path.exists(processed_path/'risks_df.parquet'):
    feature_groups['Risks'] = [x for x in risks_df.columns if x not in exclude_columns]
    
if Path.exists(processed_path/'assessments_df.parquet'):
    feature_groups['Assessments'] = [x for x in assessments_df.columns if x not in exclude_columns]

with open('./feature_groups.json', 'w') as outfile: json.dump(feature_groups, outfile)  


# STEP 5

import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from prettytable import PrettyTable
sys.path.insert(0, '/src')
from eliot import to_file
to_file(sys.stdout)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

processed_path = Path('/data/processed')
raw_path = Path('/data/raw')

base_df = pd.read_parquet(processed_path/'final_df.parquet')

CUMULATIVE_GROUP_MAPPING = {
    r'^cumsum_2_day_alert_.*': 'cumsum_2_day_alert',
    r'^cumsum_7_day_alert_.*': 'cumsum_7_day_alert',
    r'^cumsum_14_day_alert_.*': 'cumsum_14_day_alert',
    r'^cumsum_30_day_alert_.*': 'cumsum_30_day_alert',
    r'^cumsum_all_alert_.*': 'cumsum_all_alert',
    r'^cumsum_2_day_dx_.*': 'cumsum_2_day_dx',
    r'^cumsum_7_day_dx_.*': 'cumsum_7_day_dx',
    r'^cumsum_14_day_dx_.*': 'cumsum_14_day_dx',
    r'^cumsum_30_day_dx_.*': 'cumsum_30_day_dx',
    r'^cumsum_all_dx_.*': 'cumsum_all_dx',
     r'^cumsum_2_day_med_.*': 'cumsum_2_day_med',
    r'^cumsum_7_day_med_.*': 'cumsum_7_day_med',
    r'^cumsum_14_day_med_.*': 'cumsum_14_day_med',
    r'^cumsum_30_day_med_.*': 'cumsum_30_day_med',
    r'^cumsum_all_med_.*': 'cumsum_all_med',
    r'^cumsum_2_day_order_.*': 'cumsum_2_day_order',
    r'^cumsum_7_day_order_.*': 'cumsum_7_day_order',
    r'^cumsum_14_day_order_.*': 'cumsum_14_day_order',
    r'^cumsum_30_day_order_.*': 'cumsum_30_day_order',
    r'^cumsum_all_order_.*': 'cumsum_all_order',
    r'^cumsum_2_day_labs_.*': 'cumsum_2_day_labs',
    r'^cumsum_7_day_labs_.*': 'cumsum_7_day_labs',
    r'^cumsum_14_day_labs_.*': 'cumsum_14_day_labs',
    r'^cumsum_30_day_labs_.*': 'cumsum_30_day_labs',
    r'^cumsum_all_labs_.*': 'cumsum_all_labs',
    
    r'^cumidx_2_day_alert_.*': 'cumidx_2_day_alert',
    r'^cumidx_7_day_alert_.*': 'cumidx_7_day_alert',
    r'^cumidx_14_day_alert_.*': 'cumidx_14_day_alert',
    r'^cumidx_30_day_alert_.*': 'cumidx_30_day_alert',
    r'^cumidx_all_alert_.*': 'cumidx_all_alert',
    r'^cumidx_2_day_dx_.*': 'cumidx_2_day_dx',
    r'^cumidx_7_day_dx_.*': 'cumidx_7_day_dx',
    r'^cumidx_14_day_dx_.*': 'cumidx_14_day_dx',
    r'^cumidx_30_day_dx_.*': 'cumidx_30_day_dx',
    r'^cumidx_all_dx_.*': 'cumidx_all_dx',
    r'^cumidx_2_day_med_.*': 'cumidx_2_day_med',
    r'^cumidx_7_day_med_.*': 'cumidx_7_day_med',
    r'^cumidx_14_day_med_.*': 'cumidx_14_day_med',
    r'^cumidx_30_day_med_.*': 'cumidx_30_day_med',
    r'^cumidx_all_med_.*': 'cumidx_all_med',
    r'^cumidx_2_day_order_.*': 'cumidx_2_day_order',
    r'^cumidx_7_day_order_.*': 'cumidx_7_day_order',
    r'^cumidx_14_day_order_.*': 'cumidx_14_day_order',
    r'^cumidx_30_day_order_.*': 'cumidx_30_day_order',
    r'^cumidx_all_order_.*': 'cumidx_all_order',
    r'^cumidx_2_day_labs_.*': 'cumidx_2_day_labs',
    r'^cumidx_7_day_labs_.*': 'cumidx_7_day_labs',
    r'^cumidx_14_day_labs_.*': 'cumidx_14_day_labs',
    r'^cumidx_30_day_labs_.*': 'cumidx_30_day_labs',
    r'^cumidx_all_labs_.*': 'cumidx_all_labs',
}

f = open ('./feature_groups.json', "r")
feature_groups = json.loads(f.read())
# Not the most efficient code but not optimizing since the cell runs pretty fast
def get_feature_group_counts():
    training_feats = base_df.columns
    features = {}
    for grp in feature_groups:
        features[grp] = len([x for x in training_feats if x in feature_groups[grp]])
    return features

def get_cumulative_group_counts():
    training_feats = pd.DataFrame({'feature': list(base_df.columns)})
    training_feats['feature_group'] = training_feats.feature.replace(
            CUMULATIVE_GROUP_MAPPING,
            regex=True
        )
    features = training_feats.groupby('feature_group')['feature_group'].count().to_dict()
    cumulative_cols = CUMULATIVE_GROUP_MAPPING.values()
    features = {k: features[k] for k in cumulative_cols}

    return features

feature_drop_stats = {}
cumulative_feature_drop_stats = {}

feature_group_count = get_feature_group_counts()
cumulative_group_count = get_cumulative_group_counts()
for grp in feature_groups:
    feature_drop_stats[grp] = {'before_drop_count': feature_group_count[grp]}
    
for grp in cumulative_group_count:
    cumulative_feature_drop_stats[grp] = {'before_drop_count': cumulative_group_count[grp]}
feature_drop_stats


### Remove features which have 100% 0 values
def na_analysis(df):
    lst = []
    cols = []
    total_rows = df.shape[0]
    cols = df.columns[df.columns.str.contains('cumidx|cumsum|days_since_last_event|na_indictator|vtl_|notes_')]
    for col in cols:
        # Sum of NaN values in a column
        na_values = max(df[col].eq(0).sum(), df[col].eq(9999).sum(), df[col].isnull().sum())
        lst.extend([[col,total_rows,na_values,(na_values/total_rows)*100]])
        if ((na_values/total_rows)*100) >= 99 and (col not in cols):
            cols.append(col)

    return lst


df_na = pd.DataFrame(
    na_analysis(base_df),
    columns=['column_name','total_count','null_values','%_null_values']
)

df_na.sort_values(['%_null_values'],ascending=False,inplace=True)

df_na.head(10)

print(base_df.shape)

drop_cols = df_na[
    (df_na['%_null_values'] >=99.9) & (~df_na['column_name'].str.startswith('hosp_target'))
]['column_name']
base_df.drop(drop_cols,
        axis=1,
        inplace=True
       )

len(drop_cols)

base_df.shape

get_cumulative_group_counts()

feature_group_count = get_feature_group_counts()
cumulative_group_count = get_cumulative_group_counts()

total_before_drop = 0
total_after_drop = 0
for grp in feature_groups:
    feature_drop_stats[grp]['after_drop_count'] = feature_group_count[grp]
    dropped_percentage = (feature_drop_stats[grp]['before_drop_count'] - feature_drop_stats[grp]['after_drop_count'])/feature_drop_stats[grp]['before_drop_count']
    feature_drop_stats[grp]['dropped_percentage'] = "{:.0%}".format(dropped_percentage)
    total_before_drop += feature_drop_stats[grp]['before_drop_count']
    total_after_drop += feature_drop_stats[grp]['after_drop_count']
dropped_percentage = (total_before_drop-total_after_drop)/total_before_drop
feature_drop_stats['Total'] = {'before_drop_count': total_before_drop, 'after_drop_count': total_after_drop, 'dropped_percentage': "{:.0%}".format(dropped_percentage)}

total_before_drop = 0
total_after_drop = 0
for grp in cumulative_group_count:
    cumulative_feature_drop_stats[grp]['after_drop_count'] = cumulative_group_count[grp]
    dropped_percentage = (cumulative_feature_drop_stats[grp]['before_drop_count'] - cumulative_feature_drop_stats[grp]['after_drop_count'])/cumulative_feature_drop_stats[grp]['before_drop_count']
    cumulative_feature_drop_stats[grp]['dropped_percentage'] = "{:.0%}".format(dropped_percentage)
    total_before_drop += cumulative_feature_drop_stats[grp]['before_drop_count']
    total_after_drop += cumulative_feature_drop_stats[grp]['after_drop_count']
dropped_percentage = (total_before_drop-total_after_drop)/total_before_drop
cumulative_feature_drop_stats['Total'] = {'before_drop_count': total_before_drop, 'after_drop_count': total_after_drop, 'dropped_percentage': "{:.0%}".format(dropped_percentage)}

print(cumulative_feature_drop_stats)

with open('./feature_drop_stats.json', 'w') as outfile: json.dump(feature_drop_stats, outfile)
with open('./cumulative_feature_drop_stats.json', 'w') as outfile: json.dump(cumulative_feature_drop_stats, outfile)

## Write feature_drop_stats and cumulative_feature_drop_stats as ascii tables

x = PrettyTable()
x.title = 'Feature Group Drop Stats'
x.field_names = ["Feature Group", "Before Feature Reduction", "After Feature Reduction", "% of Dropped Features"]
# To make sure the groups are in alphabetical order
grps = list(feature_drop_stats.keys())
total = grps.pop()
grps = sorted(grps) + [total]
for grp in grps:
        x.add_row([grp, feature_drop_stats[grp]['before_drop_count'], feature_drop_stats[grp]['after_drop_count'], feature_drop_stats[grp]['dropped_percentage']])

with open('./feature_group_drop_stats.txt', 'w') as w:
    w.write(str(x))

x = PrettyTable()
x.title = 'Feature Cumulative Group Drop Stats'
x.field_names = ["Feature Group", "Before Feature Reduction", "After Feature Reduction", "% of Dropped Features"]
for grp in cumulative_feature_drop_stats:
        x.add_row([grp, cumulative_feature_drop_stats[grp]['before_drop_count'], cumulative_feature_drop_stats[grp]['after_drop_count'], cumulative_feature_drop_stats[grp]['dropped_percentage']])
        
with open('./feature_cumulative_drop_stats.txt', 'w') as w:
    w.write(str(x))

print(base_df.shape)

base_df.to_parquet(processed_path/'final_cleaned_df.parquet')

## =======================END=====================

base_df = pd.read_parquet(processed_path/'05-result.parquet')

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, SelectPercentile
import pickle

output_cols = [col for col in base_df.columns if 'hosp_target' in col]
x_df = base_df[base_df.columns.difference(output_cols)]
y_df = base_df[output_cols]
print(x_df.shape)
print(y_df.shape)

exclude_cols = ['masterpatientid','censusdate', 'facilityid', 'bedid', 'client']

x_df = x_df[x_df.columns.difference(exclude_cols)]
x_df.shape

y_df = y_df.fillna(False)
y_df['hosp_target_3_day_hosp'] = y_df['hosp_target_3_day_hosp'].astype('float32')
target_3_day = y_df['hosp_target_3_day_hosp']

def fill_na_train(df):
    # Fill Median value for all NaN's in the respective columns
    has_na = df.isna().sum() > 0
    d = df.loc[:, has_na].median()
    df = df.fillna(d)
    
    return df, d

def fill_na_valid_or_test(df, na_filler):
    return df.fillna(na_filler)


x_df, na_filler = fill_na_train(x_df)
x_df = x_df.astype('float32')

print(x_df.shape)
print(y_df.shape)
print(len(target_3_day))

x_df.to_parquet(processed_path/'x_df.parquet')
with open(processed_path/'target_3_day.pickle','wb') as f: pickle.dump(target_3_day, f, protocol=4)

x_df = pd.read_parquet(processed_path/'x_df.parquet')
with open(processed_path/'target_3_day.pickle','rb') as f: target_3_day = pickle.load(f)

## Feature Selection 

%%time

# Correlation for all features with the target

corr_matrix = x_df.corrwith(y_df['hosp_target_3_day_hosp'])

_df = pd.DataFrame({'cols':corr_matrix.index, 'value':corr_matrix.values})
_df.sort_values(by='value',ascending=False).head(2000)

%%time
## Remove constant features

constant_features = []
for feat in x_df.columns:
    # convert all features to Float32
    
    if x_df[feat].std() == 0:
        constant_features.append(feat)

print(constant_features)

# x_df.drop(labels=constant_features, axis=1, inplace=True)


%%time

# Remove duplicated features

duplicated_features = []
for i in range(0, len(x_df.columns)):
    col_1 = x_df.columns[i]

    for col_2 in x_df.columns[i + 1:]:
        if x_df[col_1].equals(x_df[col_2]):
            duplicated_features.append(col_2)

print(duplicated_features)

# x_df.drop(labels=duplicated_features, axis=1, inplace=True)

%%time


# calculate the mutual information between the variables and the target
# this returns the mutual information value of each feature.
# the smaller the value the less information the feature has about the target


mi = mutual_info_classif(x_df.fillna(0), target_3_day)
print(mi)

# let's add the variable names and order the features
# according to the MI for clearer visualisation
mi = pd.Series(mi)
mi.index = x_df.columns
mi = mi.sort_values(ascending=False)
mi.to_csv('mi-date_cols.csv', header=True)
# and now let's plot the ordered MI values per feature
mi.sort_values(ascending=False).plot.bar(figsize=(20, 8))

%%time

# here I will select the top 10 features
# which are shown below
sel_ = SelectKBest(mutual_info_classif, k=10).fit(x_df.fillna(0), target_3_day)
x_df.columns[sel_.get_support()]

%%time

# calculate the chi2 p_value between each of the variables
# and the target
# it returns 2 arrays, one contains the F-Scores which are then
# evaluated against the chi2 distribution to obtain the pvalue
# the pvalues are in the second array, see below

f_score = chi2(x_df, target_3_day)
f_score


# Keep in mind, that contrarily to MI, where we were interested in the higher MI values,
# for Fisher score, the smaller the p_value, the more significant the feature is to predict the target.

# One thing to keep in mind when using Fisher score or univariate selection methods,
# is that in very big datasets, most of the features will show a small p_value,
# and therefore look like they are highly predictive.
# This is in fact an effect of the sample size. So care should be taken when selecting features
# using these procedures. An ultra tiny p_value does not highlight an ultra-important feature,
# it rather indicates that the dataset contains too many samples.

# If the dataset contained several categorical variables, we could then combine this procedure with
# SelectKBest or SelectPercentile, as I did in the previous lecture.

%%time

# let's add the variable names and order it for clearer visualisation

pvalues = pd.Series(f_score[1])
pvalues.index = x_df.columns
pvalues.sort_values(ascending=True)


%%time

# LASSO Regularization

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

logistic = LogisticRegression(C=1, penalty='l1',solver='liblinear',random_state=7).fit(x_df,target_3_day)
model = SelectFromModel(logistic, prefit=True)

# x_new_df = model.transform(x_df)

# this command let's me visualise those features that were kept
model.get_support()

%%time

# Now I make a list with the selected features
selected_feat = x_df.columns[(model.get_support())]

print('total features: {}'.format((x_df.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
    np.sum(model.estimator_.coef_ == 0)))


%%time

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# step forward feature selection
# I indicate that I want to select 10 features from
# the total, and that I want to select those features
# based on the optimal roc_auc

sfs1 = SFS(RandomForestRegressor(),
           k_features=20,
           forward=True,
           floating=False,
           verbose=2,
           scoring='r2',
           cv=3)

sfs1 = sfs1.fit(np.array(x_df), target_3_day)
selected_feat= x_df.columns[list(sfs1.k_feature_idx_)]
selected_feat

!pip install mlxtend


# STEP 6

import gc
import sys
import numpy as np
from pathlib import Path
from datetime import timedelta, datetime
import re

import pandas as pd

sys.path.insert(0, '/src')
from shared.constants import CLIENT, HYPER_PARAMETER_TUNING, MODEL_TYPE
from shared.utils import get_client_class, url_encode_cols


MODEL_TYPE = MODEL_TYPE.lower()
print('MODEL:', MODEL_TYPE)

 ## =========== Set HYPER_PARAMETER_TUNING in constants.py ===========

 clientClass = get_client_class(client=CLIENT)
EXPERIMENT_DATES = getattr(clientClass(), 'get_experiment_dates')()

# starting training from day 31 so that cumsum window 2,7,14,30 are all initial correct.
# One day will be added to `censusdate` later in the code, so that the first date in
# `train` will be `EXPERIMENT_DATES['train_start_date'] + 1 day`, that's why here we
# add 31 days but not 30
EXPERIMENT_DATES['train_start_date'] = str((pd.to_datetime(EXPERIMENT_DATES['train_start_date']) +  pd.DateOffset(days=31)).date())

if not HYPER_PARAMETER_TUNING:
    EXPERIMENT_DATES['train_end_date'] = (datetime.strptime(EXPERIMENT_DATES['validation_end_date'], '%Y-%m-%d') - timedelta(days=2)).strftime('%Y-%m-%d')
    EXPERIMENT_DATES['validation_start_date'] = (datetime.strptime(EXPERIMENT_DATES['validation_end_date'], '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    
print(CLIENT)
print(HYPER_PARAMETER_TUNING)
print(EXPERIMENT_DATES)


processed_path = Path('/data/processed')
processed_path.mkdir(parents=True, exist_ok=True)
filename = 'final_cleaned_df.parquet'

final = pd.read_parquet(processed_path/f'{filename}')

final = url_encode_cols(final)

IDEN_COLS = ['censusdate', 'facilityid', 'masterpatientid', 'LFS', 'primaryphysicianid',
         'payername', 'to_from_type', 'client', 'admissionstatus',
         f'positive_date_{MODEL_TYPE}']

# UPT model doesn't need the rows that with payername contains 'hospice'
if MODEL_TYPE=='upt':
    final = final[~(final['payername'].str.contains('hospice', case=False, regex=True, na=False))]

# column processing
final['client'] = final['masterpatientid'].apply(lambda z: z.split('_')[0])
final["facilityid"] = final["client"] + "_" + final["facilityid"].astype(str)

final['LFS'] = final['days_since_last_admission']

""" We increment the census date by 1, since the prediction day always includes data upto last night.
This means for every census date the data is upto previous night. 
"""
print(final.shape)

# Increment censusdate by 1
final['censusdate'] = (pd.to_datetime(final['censusdate']) + timedelta(days=1))

print(final.shape)

def drop_unwanted_columns(df):   
    positive_date = f'positive_date_{MODEL_TYPE}'
    target_3_day = f'target_3_day_{MODEL_TYPE}'
    drop_columns = ['dateofadmission']
    dates = list(df.columns[df.columns.str.contains('positive_date_')])
    targets = list(df.columns[df.columns.str.contains('target_3_day_')])
    for date in dates: 
        if date!=positive_date:
            drop_columns.append(date)
    for target in targets:
        if target!= target_3_day:
            drop_columns.append(target)
    df = df.drop(columns=drop_columns)
    return df

# drop unwanted columns for this model
final = drop_unwanted_columns(final)
    
final[f'target_3_day_{MODEL_TYPE}'] = final[f'target_3_day_{MODEL_TYPE}'].fillna(False)

# manual check to make sure we're not including any columns that could leak data
with open(f'/data/processed/columns_{MODEL_TYPE}.txt','w') as f:
    for col in final.columns:
        f.write(col + '\n')

train = final.loc[(final.censusdate >= EXPERIMENT_DATES['train_start_date']) & (final.censusdate <= EXPERIMENT_DATES['train_end_date'])]
valid = final.loc[(final.censusdate >= EXPERIMENT_DATES['validation_start_date']) & (final.censusdate <= EXPERIMENT_DATES['validation_end_date'])]
test = final.loc[final.censusdate >= EXPERIMENT_DATES['test_start_date']]

print(final.shape)
print(train.shape)
print(valid.shape)
print(test.shape)

del final
gc.collect()

for col in train.columns:
    if 'target_3_day' in col:
        print(col)

print(f'train - target_3_day_{MODEL_TYPE}', train[f'target_3_day_{MODEL_TYPE}'].value_counts())
print(f'valid - target_3_day_{MODEL_TYPE}', valid[f'target_3_day_{MODEL_TYPE}'].value_counts())
print(f'test - target_3_day_{MODEL_TYPE}', test[f'target_3_day_{MODEL_TYPE}'].value_counts())

# start of basic tests - assert we have disjoint sets over time
assert train.censusdate.max() < valid.censusdate.min()
assert valid.censusdate.max() < test.censusdate.min()
print('Success...')

print(f'Train set covers {train.censusdate.min()} to {train.censusdate.max()} with 3_day_{MODEL_TYPE} percentage {train[f"target_3_day_{MODEL_TYPE}"].mean()}')
print(f'Valid set covers {valid.censusdate.min()} to {valid.censusdate.max()} with 3_day_{MODEL_TYPE} percentage {valid[f"target_3_day_{MODEL_TYPE}"].mean()}')
print(f'Test set covers {test.censusdate.min()} to {test.censusdate.max()} with 3_day_{MODEL_TYPE} percentage {test[f"target_3_day_{MODEL_TYPE}"].mean()}')


for col in train.columns:
#     if col not in IDEN_COLS:
#         if 'int' not in train[col].dtypes or 'bool' not in train[col].dtypes:
    if train[col].dtypes=='datetime64[ns]':
        if col not in IDEN_COLS:
            print(col, train[col].dtypes)

# Remove the Target values & identification columns
# Keep facilityid in idens and add a duplicate field as facility for featurisation
def prep(df):
    drop_cols = IDEN_COLS + [col for col in df.columns if 'target' in col]

    target_3_day = df[f'target_3_day_{MODEL_TYPE}'].astype('float32').values
    
    df['facility'] = df['facilityid']  
    df['facility'] = df['facility'].astype('category')
    
    x = df.drop(columns=drop_cols).reset_index(drop=True)
    category_columns = x.dtypes[x.dtypes == 'category'].index.tolist()
    x = x.drop(columns=category_columns).reset_index(drop=True)
    
    categorical_data = df.loc[:,category_columns].reset_index(drop=True)
    x = x[x.columns].astype('float32', errors='ignore')
    
    # concatenating categorical data and float values
    x = pd.concat([x, categorical_data],axis=1)
    idens = df.loc[:,IDEN_COLS]

    return x, target_3_day, idens

%%time

# Seperate target, x-frame and identification columns
train_x, train_target_3_day, train_idens = prep(train)
del train
valid_x, valid_target_3_day, valid_idens = prep(valid)
del valid
test_x, test_target_3_day, test_idens = prep(test)
del test

gc.collect()

# get columns names of categorial variables
cate_columns = list(train_x.columns[train_x.dtypes=='category'])

# make sure for that x's, targets, an idens all have the same # of rows
assert train_x.shape[0] == train_target_3_day.shape[0] == train_idens.shape[0]
assert valid_x.shape[0] == valid_target_3_day.shape[0] == valid_idens.shape[0]
assert test_x.shape[0] == test_target_3_day.shape[0] == test_idens.shape[0]

# make sure that train, valid, and test have the same # of columns
assert train_x.shape[1] == valid_x.shape[1] == test_x.shape[1]

# make sure that the idens all have the same # of columns
assert train_idens.shape[1] == valid_idens.shape[1] == test_idens.shape[1]


%%time

# Save train, test and validation datasets in local folder

import pickle;
with open(processed_path/f'final-train_x_{MODEL_TYPE}.pickle','wb') as f: pickle.dump(train_x, f, protocol=4)
with open(processed_path/f'final-train_target_3_day_{MODEL_TYPE}.pickle','wb') as f: pickle.dump(train_target_3_day, f, protocol=4)
with open(processed_path/f'final-train_idens_{MODEL_TYPE}.pickle','wb') as f: pickle.dump(train_idens, f, protocol=4)

with open(processed_path/f'final-valid_x_{MODEL_TYPE}.pickle','wb') as f: pickle.dump(valid_x, f, protocol=4)
with open(processed_path/f'final-valid_target_3_day_{MODEL_TYPE}.pickle','wb') as f: pickle.dump(valid_target_3_day, f, protocol=4)
with open(processed_path/f'final-valid_idens_{MODEL_TYPE}.pickle','wb') as f: pickle.dump(valid_idens, f, protocol=4)

with open(processed_path/f'final-test_x_{MODEL_TYPE}.pickle','wb') as f: pickle.dump(test_x, f, protocol=4)
with open(processed_path/f'final-test_target_3_day_{MODEL_TYPE}.pickle','wb') as f: pickle.dump(test_target_3_day, f, protocol=4)
with open(processed_path/f'final-test_idens_{MODEL_TYPE}.pickle','wb') as f: pickle.dump(test_idens, f, protocol=4)

with open(processed_path/'cate_columns.pickle', 'wb') as f: pickle.dump(cate_columns, f, protocol=4)
    

print("--------------Completed--------------")

print(train_x.shape)
print(train_target_3_day.shape)
print(valid_x.shape)
print(valid_target_3_day.shape)
print(test_x.shape)
print(test_target_3_day.shape)


# STEP 7

import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
import sys

sys.path.insert(0, '/src')
from shared.constants import CLIENT, VECTOR_MODELS, HYPER_PARAMETER_TUNING, MODEL_TYPE, OPTUNA_TIME_BUDGET
MODEL_TYPE = MODEL_TYPE.lower()
from shared.utils import get_client_class
from training import train_optuna_integration, get_facilities_from_train_data, IdensDataset, load_x_y_idens

### ========= Set the CONFIG & HYPER_PARAMETER_TUNING in constants.py ==========
clientClass = get_client_class(client=CLIENT)
EXPERIMENT_DATES = getattr(clientClass(), 'get_experiment_dates')()

# starting training from day 31 so that cumsum window 2,7,14,30 are all initial correct.
EXPERIMENT_DATES['train_start_date'] = str((pd.to_datetime(EXPERIMENT_DATES['train_start_date']) +  pd.DateOffset(days=31)).date())


TRAINING_DATA='Caldera'   # trained on which data? e.g. avante + champion
SELECTED_MODEL_VERSION = 'saiva-3-day-upt_v6'    # e.g. v3, v4 or v6 model

# Name used to filter models in AWS quicksight & also used as ML Flow experiment name
MODEL_DESCRIPTION = 'test' # e.g. 'avante-upt-v6-model'

print('MODEL_TYPE:', MODEL_TYPE)
print('HYPER_PARAMETER_TUNING:', HYPER_PARAMETER_TUNING)  
print('CLIENT:', CLIENT)
EXPERIMENT_DATES

## ============ Initialise MLFlow Experiment =============

# Create an ML-flow experiment
mlflow.set_tracking_uri('http://mlflow.saiva-dev')

# Experiment name which appears in ML flow
mlflow.set_experiment(MODEL_DESCRIPTION)

EXPERIMENT = mlflow.get_experiment_by_name(MODEL_DESCRIPTION)
EXPERIMENT_ID = EXPERIMENT.experiment_id

print(f'Experiment ID: {EXPERIMENT_ID}')

## =================== Loading data ======================

processed_path = Path('/data/processed')
processed_path.mkdir(parents=True, exist_ok=True)

train_x, train_target_3_day, train_idens = load_x_y_idens(processed_path, MODEL_TYPE, 'train')
valid_x, valid_target_3_day, valid_idens = load_x_y_idens(processed_path, MODEL_TYPE, 'valid')
test_x, test_target_3_day, test_idens = load_x_y_idens(processed_path, MODEL_TYPE, 'test')

print(train_x.shape)
print(train_target_3_day.shape)
print(valid_x.shape)
print(valid_target_3_day.shape)
print(test_x.shape)
print(test_target_3_day.shape)

info_cols = ['facilityid', 'censusdate', 'masterpatientid', f'positive_date_{MODEL_TYPE}', 'LFS']
train_data = IdensDataset(train_x, label=train_target_3_day, idens=train_idens.loc[:,info_cols])
valid_data = IdensDataset(valid_x, label=valid_target_3_day, idens=valid_idens.loc[:,info_cols])
test_data = IdensDataset(test_x, label=test_target_3_day, idens=test_idens.loc[:,info_cols])

## =================== Model Training ===================

# We have a new training method. After calling it, wait for 5 minutes, make sure everything is working properly. If there are no issues, you can start doing something else. Typically, this process takes around 12-24 hours (depending on the size of the dataset), and you can track the results through mlflow.

params = {
    "seed": 1,
    "metric": "auc",
    "verbosity": 5,
    "boosting_type": "gbdt",
    }

model = train_optuna_integration(
    params,
    train_data,
    valid_data,
    test_data,
    processed_path,
    VECTOR_MODELS[CLIENT],
    MODEL_TYPE,
    EXPERIMENT_DATES,
    HYPER_PARAMETER_TUNING,
    TRAINING_DATA,
    SELECTED_MODEL_VERSION,
    MODEL_DESCRIPTION,
    EXPERIMENT_ID,
    OPTUNA_TIME_BUDGET
)

# STEP 8

import json
import sys

import pandas as pd

sys.path.insert(0, '/src')
from shared.database import DbEngine
from shared.constants import CLIENT, ENV


# These 2 variables needs to be updated after fine tuning
FACILITY_GOLIVE_DATE = '2022-01-08'
MODEL_GOLIVE_DATE = '2022-01-08'

DEFAULT_FACILITY_CONFIG = {
    'rank_cutoff': 15,
    'group_level': 'facility',
    'facility_golive_date': FACILITY_GOLIVE_DATE
}
# Configure the facilities which are required
ACTIVE_FACILITIES = {
#     '7': {
#         'rank_cutoff': 15,
#         'group_level': 'facility',
#         'facility_golive_date': FACILITY_GOLIVE_DATE
#     },
#     '42': {
#         'rank_cutoff': 15,
#         'group_level': 'facility',
#         'facility_golive_date': FACILITY_GOLIVE_DATE
#     },
#     '52': {
#         'rank_cutoff': 15,
#         'group_level': 'facility',
#         'facility_golive_date': FACILITY_GOLIVE_DATE
#     },
#     '278': {
#         'rank_cutoff': 15,
#         'group_level': 'facility',
#         'facility_golive_date': FACILITY_GOLIVE_DATE
#     },
#     '55': {
#         'rank_cutoff': 15,
#         'group_level': 'facility',
#         'facility_golive_date': '2020-07-05'
#     }
}

for i in ['29','33','17','30','27']:
    ACTIVE_FACILITIES[i] = {
        'rank_cutoff': 15,
        'group_level': 'facility',
        'facility_golive_date': FACILITY_GOLIVE_DATE
    }

engine = DbEngine()
saivadb_engine = engine.get_postgresdb_engine()

engine.verify_connectivity(saivadb_engine)

print(ENV)
print(CLIENT)
print(len(ACTIVE_FACILITIES))
ACTIVE_FACILITIES


with open('./model_config.json','rb') as f: model_list = json.load(f)

selected_modelid = model_list[0]['modelid']
model_metadata_df = pd.DataFrame(model_list)
facility_list = ACTIVE_FACILITIES.keys()
model_metadata_df.head()


# write all the models to the model_metadata table
model_metadata_df.to_sql(
    'model_metadata',
     saivadb_engine,
     if_exists='append',
     index=False
)


facility_model_config = []
for id in facility_list:
    obj = {
        'client': CLIENT,
        'facilityid': id,
        'modelid': selected_modelid,
        'group_level':ACTIVE_FACILITIES.get(id, DEFAULT_FACILITY_CONFIG).get('group_level'),
        'model_golive_date': MODEL_GOLIVE_DATE,
        'facility_golive_date': ACTIVE_FACILITIES.get(id, DEFAULT_FACILITY_CONFIG).get('facility_golive_date'),
        'rank_cutoff': ACTIVE_FACILITIES.get(id, DEFAULT_FACILITY_CONFIG).get('rank_cutoff'),
        'active_facility': True if id in ACTIVE_FACILITIES.keys() else False,
        # 'parent': None
    }
    facility_model_config.append(obj)

print(len(facility_model_config))
facility_model_config


# Update all old records as deleted rows
saivadb_engine.execute(
                    f"""update facility_model_config set deleted_at=now() where client = '{CLIENT}' and deleted_at is null """
                )

model_config_df = pd.DataFrame(facility_model_config)

# append the new rows to the table
model_config_df.to_sql(
    'facility_model_config',
    saivadb_engine,
    if_exists='append',
    index=False
)


# Change client_metadata configuration only when there is a change in report_version or sftp_ingestion_version or model_version


data = {
        'client': CLIENT,
        'ingestion_method': 'SCRAPING',    # DB_VPN, DB_SFTP, API, SCRAPING
        'ingestion_version': 'v1',   # v3, v4
        'model_version': 'saiva-3-day-hosp-v5',  # saiva-3-day-hosp-v3, saiva-3-day-hosp-v1
        'report_version': 'v2',  # v1, v2
        'ehr': 'MATRIXCARE',
        # 'parent': None
}

saivadb_engine.execute(
                    f"""update client_metadata set deleted_at=now() where client = '{CLIENT}' and deleted_at is null """
                )

client_config_df = pd.DataFrame(data, index=[0])

client_config_df.to_sql(
    'client_metadata',
    saivadb_engine,
    if_exists='append',
    index=False
)

# If there are any subclients insert the rows manually and set the parent field accordingly
