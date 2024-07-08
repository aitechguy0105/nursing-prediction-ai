import json
import os
from collections import defaultdict

import s3fs
import boto3
from saiva_internal_sdk import SaivaInternalAPI

# ========================================================================

LOCAL_TRAINING_CONFIG_PATH = '/src/saiva/conf/training/'

# ========================================================================
S3 = s3fs.S3FileSystem(anon=False)
REGION_NAME = 'us-east-1'
ENV = os.environ.get('SAIVA_ENV', 'dev')
SAIVADB_CREDENTIALS_SECRET_ID = os.environ.get('SAIVADB_CREDENTIALS_SECRET_ID', f'{ENV}-saivadb')
MODEL_TYPE = 'MODEL_UPT' # MODEL_FALL
INVALID_ACTIONTYPE = ['Leave', 'Discharge', 'Death', 'Room Reserve']
REQUIRED_DATAFRAMES = ['patient_census', 'patient_rehosps', 'patient_adt', 'patient_demographics', 'patient_admissions']
STOP_WATCH_STDALERTTYPEID = [2,3,101]
FALL_LABEL = 'fall|found\s+on\s+floor|observed\s+on\s+floor'
VECTOR_MODELS = defaultdict(lambda: 'SpacyModel')  # alternative model is: FastTextModel
# Example to configure different Vector model
# VECTOR_MODELS['trio'] = 'FastTextModel'
# ========================================================================
ONLY_USE_CACHE = 'only_use_cache'
IGNORE_CACHE = 'ignore_cache'
CHECK_CACHE_FIRST = 'check_cache_first'
CACHE_STRATEGY_OPTIONS = [ONLY_USE_CACHE, IGNORE_CACHE, CHECK_CACHE_FIRST]

# ========================================================================
CLIENT_NPI_CONFIG = {}
CLIENT_NPI_CONFIG['phcp'] = ('1568974046', '1093169849', '1366480311', '1407857089', '1588658348',
                             '1407159536')


# ========================Copy trio to saiva_demo=========================
# MODELS['saiva_demo'] = MODELS['trio']

# ========================================================================
saiva_internal_api_token = os.environ.get("SAIVA_INTERNAL_API_TOKEN", None)

if not saiva_internal_api_token:
    session = boto3.session.Session()
    secrets_client = session.client(
        service_name='secretsmanager',
        region_name=REGION_NAME
    )
    saiva_internal_api_token = json.loads(
        secrets_client.get_secret_value(SecretId=f'{ENV}-backend-service-account-api-token')[
            'SecretString'
        ]
    )['API_TOKEN']
saiva_api = SaivaInternalAPI(environment=ENV, api_token=saiva_internal_api_token)
