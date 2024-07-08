import json
import os
import timeit
from distutils.util import strtobool
from functools import wraps
from inspect import getcallargs

import boto3
import pandas as pd
import sentry_sdk
from sentry_sdk import set_tag

from saiva.model.base_model import Inference
from saiva.model.shared.database import DbEngine
from saiva.model.shared.constants import saiva_api

env = os.environ.get("SAIVA_ENV", "dev")
region_name = "us-east-1"

dbobj = DbEngine()
db_engine = dbobj.get_postgresdb_engine(db_name='backend')

# TODO: update for multimodel
def put_metrics(metric_name, dimensions, value, unit, disable_instrumentation):
    if not disable_instrumentation:
        dimensions = [{'Name': k, 'Value': v} for k, v in dimensions.items() if v]
        cloudwatch = boto3.client('cloudwatch', region_name=region_name)
        cloudwatch.put_metric_data(
            MetricData=[
                {
                    'MetricName': metric_name,
                    'Dimensions': dimensions,
                    'Unit': unit,
                    'Value': value
                },
            ],
            Namespace='ML'
        )

def setup_sentry(disable_instrumentation, client, facility_id):
    if not disable_instrumentation:
        session = boto3.session.Session()
        secrets_client = session.client(
            service_name="secretsmanager", region_name=region_name
        )
        sentry_info = json.loads(
            secrets_client.get_secret_value(SecretId="sentry")[
                "SecretString"
            ]
        )
        sentry_sdk.init(
            dsn=sentry_info['ml-dsn'],
            environment=env,
            traces_sample_rate=1.0
        )
        if client and facility_id:
            org = saiva_api.organizations.get(org_id=client)
            set_tag('client_id', org.id)
            set_tag('client_id', org.name)
        if facility_id:
            facility = saiva_api.facilities.get_by_customers_identifier(org_id=client, customers_identifier=facility_id)
            set_tag('facility_id', facility.customers_identifier)
            set_tag('facility_name', facility.name)
            

# disable_instrumentation is None by default and takes the value defined in SSM Parameter Store
# To override the configured value, pass disable_instrumentation as True/False from CLI
def should_disable_instrumentation(disable_instrumentation):
    if disable_instrumentation is None:
        ssm = boto3.client('ssm', region_name=region_name)
        parameter = ssm.get_parameter(Name=f'/Configurations/{env}/disable-instrumentation', WithDecryption=True)
        return strtobool(parameter['Parameter']['Value'])
    else:
        return disable_instrumentation


def instrumented(function):
    @wraps(function)
    def _instrumented(*args, **kwargs):
        task_name = function.__name__
        call_args = getcallargs(function, *args, **kwargs)

        inference: Inference = call_args.get('self')
        client = inference.client
        facility_id = str(inference.facility_id)
        disable_instrumentation = should_disable_instrumentation(call_args.get('disable_instrumentation')) 
        dimensions = {'client': client, 'facility_id': facility_id, 'env': env}

        setup_sentry(disable_instrumentation, client, facility_id)

        put_metrics(
            metric_name=f'{task_name}_invocation',
            dimensions=dimensions,
            value=1,
            unit='Count',
            disable_instrumentation=disable_instrumentation
        ) 
        start_time = timeit.default_timer()
        try:
            function(*args, **kwargs)
        except Exception:
            put_metrics(
                metric_name=f'{task_name}_error',
                dimensions=dimensions,
                value=1,
                unit='Count',
                disable_instrumentation=disable_instrumentation
            )
            raise
        elapsed = float(timeit.default_timer() - start_time)
        put_metrics(
            metric_name=f'{task_name}_success',
            dimensions=dimensions,
            value=1,
            unit='Count',
            disable_instrumentation=disable_instrumentation
        )
        put_metrics(
            metric_name=f'{task_name}_duration',
            dimensions=dimensions,
            value=elapsed,
            unit='Seconds',
            disable_instrumentation=disable_instrumentation
        )
    return _instrumented
