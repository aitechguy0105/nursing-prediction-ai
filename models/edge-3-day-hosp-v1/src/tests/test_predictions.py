import pytest
# from shared.load_raw_data import fetch_training_data
import os
import pandas as pd
import boto3
from urllib.parse import urlparse
import difflib
import time

# To run tests: 
#   "pytest -s tests" from parent dir to run all tests
#   "pytest -s" in tests dir to run all tests
#   "pytest -s test_predictions.py::test_queries_are_same" to run only the test test_queries_are_same
#   "pytest -s test_predictions.py::test_predictions_are_consistent" to run only the test test_predictions_are_consistent
#   "pytest -s --runslow" to include running the one long slow test that runs predictions twice and checks if results are same
#   "pytest -s --dryrun" to run tests without actually invoking the run_model (this will work if local files are already stored)
#   "pytest -s --runslow --dryrun" to include slow tests and dryrun

# To Save New Golden Copy, run the following Command on the command line once (remember to turn on dev client 
#                                                    VPN if cache does not already exist!)
# python /src/run_model.py --client avante --facilityids '[9]' 
#         --prediction_date 2020-05-08 --s3_bucket saiva-dev-data-bucket 
#         --trainset_start_date 2017-01-01 --replace_existing_predictions True 
#         --cache_s3_folder golden/run1 --cache_strategy only_use_cache --save_outputs_in_s3 True --test True


DEV_CLIENT_VPN_INSTANCE_ID = 'i-07197a52885c34d68'

def cmd_run_model(client, facility_id, prediction_date, cache_strategy, 
                  cache_s3_folder, local_folder_path):
    cmd = (f"python /src/run_model.py --client {client} " + 
           f"--facilityids '[{facility_id}]' --prediction_date {prediction_date} " +
           f"--s3_bucket saiva-dev-data-bucket --trainset_start_date 2017-01-01 " +
           f"--cache_s3_folder {cache_s3_folder} " +
           f"--cache_strategy {cache_strategy} --test True  --save_outputs_in_local True " +
           f"--local_folder {local_folder_path} ")
    ret_val = os.system(cmd)
    assert ret_val == 0, f'run_model.py failed'
    return ret_val


# will call run_model only once per parameter for the entire module for all tests
@pytest.fixture(scope="module", params=[('avante', '9', '2020-05-08', 'golden/run1', 
                                         '/data/test/2020-05-08'),
                                         ('avante', '10', '2020-05-06', 'golden/run1', 
                                         '/data/test/2020-05-06')
                                         ])
def run_model(request):
    """ setup any state specific to the execution of the given module.
        This is run once per module, fixture combination"""
    print("in setup_module")
    dryrun = request.config.getoption("--dryrun")
    client = request.param[0]
    facility_id = request.param[1]
    prediction_date = request.param[2]
    cache_s3_folder = request.param[3]
    local_folder_path = request.param[4]
    s3_bucket = 'saiva-dev-data-bucket'
    if not dryrun:
        cmd_run_model(client = client, 
                      facility_id = facility_id,
                      prediction_date = prediction_date,
                      cache_strategy = 'only_use_cache',
                      cache_s3_folder = cache_s3_folder, 
                      local_folder_path = local_folder_path)
    golden_copy_location = f's3://{s3_bucket}/data/{client}/{prediction_date}/{facility_id}/{cache_s3_folder}'
    return golden_copy_location, local_folder_path


def dfs_are_equal(df1, df2):
    df = pd.concat([df1, df2]).drop_duplicates(keep=False)
    print(df.shape)
    if len(df) != 0:
        print(f"two dataframes are different!!")
    return (len(df) == 0)


def read_txt_from_s3(s3_path):
    s3_client = boto3.client('s3')
    # get the bucket and key from the path
    o = urlparse(s3_path)
    bucket = o.netloc
    key = o.path.lstrip('/')
    # log_message(message_type="info", bucket=bucket, key=key)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    query_str = obj['Body'].read().decode('utf-8')
    return query_str


def read_txt_from_local(local_path):
    file = open(local_path,"r")
    query_str = file.read()
    return query_str


def normalize_str(input_str):
    # first delete all \n characters
    out_str = input_str.replace("\n", "")
    # next delete multiple spaces
    out_str = ' '.join(out_str.split())
    return out_str


def strs_are_equal(golden_str, local_str):
    golden = normalize_str(golden_str)
    local = normalize_str(local_str)
    # print(f"golden: {g1}")
    # print(f"local: {l1}")
    if golden != local:
        output_list = [li for li in difflib.ndiff(golden, local) if li[0] != ' ']
        print(f"DIFF: f{output_list}")
    return golden == local


def start_dev_client_vpn_instance():
    print("Starting dev client VPN Instance...")
    client = boto3.client('ec2', region_name='us-east-1')
    client.start_instances(InstanceIds=[DEV_CLIENT_VPN_INSTANCE_ID])


def stop_dev_client_vpn_instance():
    print("Stopping dev client VPN Instance...")
    client = boto3.client('ec2', region_name='us-east-1')
    client.stop_instances(InstanceIds=[DEV_CLIENT_VPN_INSTANCE_ID])


@pytest.mark.parametrize("query_name", ["patient_vitals", "master_patient_lookup", "patient_census", 
                                        "patient_rehosps",  "patient_demographics", "patient_diagnosis", 
                                        "patient_meds", "patient_orders", "patient_alerts", 
                                        "patient_lab_results", "patient_progress_notes"])
def test_queries_are_same(run_model, query_name):
    golden_copy_path, local_folder_path = run_model
    golden = golden_copy_path + '/' + query_name + '.txt'
    local = local_folder_path + '/' + query_name + '.txt'
    print(f"comparing {golden} vs {local}")
    golden_query = read_txt_from_s3(golden)
    local_query = read_txt_from_local(local)
    assert strs_are_equal(golden_query, local_query)


@pytest.mark.parametrize("dataset_name", ["base3", "base4", "base5", "base6", "base9", "base11", "base12", 
                                          "combined", "complex_features", "finalx", "predictions"])
def test_datasets(run_model, dataset_name):
    golden_copy_path, local_folder_path = run_model
    golden = golden_copy_path + '/' + dataset_name + '_output.parquet'
    local = local_folder_path + '/' + dataset_name + '_output.parquet'
    print(f"comparing {golden} vs {local}")
    golden_copy_df = pd.read_parquet(golden)
    local_df = pd.read_parquet(local)
    assert dfs_are_equal(golden_copy_df, local_df)


@pytest.mark.slow
def test_predictions_are_consistent(request):

    run1_local_path = '/data/test/2020-05-14/run1'
    run2_local_path = '/data/test/2020-05-14/run2'
    dryrun = request.config.getoption("--dryrun")
    if not dryrun:
        start_dev_client_vpn_instance()
        print("Sleeping for 30 seconds")
        time.sleep(30)
        
        print("Running Prediction First Time")
        cmd_run_model(client = 'avante', 
                    facility_id = '6',
                    prediction_date = '2020-05-14',
                    cache_strategy = 'ignore_cache',
                    cache_s3_folder = 'test/run1', 
                    local_folder_path = run1_local_path)
        
        print("Running Prediction Second Time")
        cmd_run_model(client = 'avante', 
                    facility_id = '6',
                    prediction_date = '2020-05-14',
                    cache_strategy = 'ignore_cache',
                    cache_s3_folder = 'test/run2', 
                    local_folder_path = run2_local_path)

        stop_dev_client_vpn_instance()

    run1_predictions = run1_local_path + '/'  + 'predictions_output.parquet'
    run2_predictions = run2_local_path + '/'  + 'predictions_output.parquet'
    print(f"comparing {run1_predictions} vs {run2_predictions}")
    run1_df = pd.read_parquet(run1_predictions)
    run2_df = pd.read_parquet(run2_predictions)
    assert dfs_are_equal(run1_df, run2_df)
