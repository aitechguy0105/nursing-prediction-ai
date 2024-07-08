import os
from collections import defaultdict

import s3fs

# ========================================================================
CLIENT = 'trio'

# ========================================================================
S3 = s3fs.S3FileSystem(anon=False)
REGION_NAME = 'us-east-1'
ENV = os.environ.get('SAIVA_ENV', 'dev')
SAIVADB_CREDENTIALS_SECRET_ID = os.environ.get('SAIVADB_CREDENTIALS_SECRET_ID', f'{ENV}-saivadb')

# ========================================================================
ONLY_USE_CACHE = 'only_use_cache'
IGNORE_CACHE = 'ignore_cache'
CHECK_CACHE_FIRST = 'check_cache_first'
CACHE_STRATEGY_OPTIONS = [ONLY_USE_CACHE, IGNORE_CACHE, CHECK_CACHE_FIRST]

# ========================================================================
MODELS = {
    'avante': defaultdict(lambda: 'd0c497c8b9b04f4d9e1e1e0c9297cc1f'),
    'trio': defaultdict(lambda: '86fbc7600dd54378bf7630f3ed60059d'),
    'infinity-infinity': defaultdict(lambda: '9bfb02fa32f644aeafba299380742580'),
    'infinity-benchmark': defaultdict(lambda: '0a4579942cb844a4b73332d1824fe143'),
    'dycora': defaultdict(lambda: '41238147855f4210ae98e067241f8dff'),
    'northshore': defaultdict(lambda: '03e63cb0f343476ea0f97991ee05e41b'),
    'meridian': defaultdict(lambda: '4356cba3e6554d38a6d1189dc6c84708'),
    'vintage': defaultdict(lambda: '5ae483a18048444ab735b10f1c9c33a4'),
    'palmgarden': defaultdict(lambda: '1f6306449c41489d8c2cf75df6457bc9'),
    'mmh': defaultdict(lambda: 'b9b897bc1d444222b5b38892e6533cc5'),
    'marquis': defaultdict(lambda: '10ebb0ef38af4dd988a2484738649657'),
}
# ========================================================================
MODELS['avante']['s3_folder'] = '99'
MODELS['avante']['training_start_date'] = '2018-01-01'
MODELS['avante']['vector_model_s3_path'] = 's3://saiva-models/progress_note_embeddings/v1'
MODELS['avante'][1] = 'd0c497c8b9b04f4d9e1e1e0c9297cc1f'
MODELS['avante'][3] = '94d349624d91490a898ec52d50e371c0'
MODELS['avante'][4] = 'd0c497c8b9b04f4d9e1e1e0c9297cc1f'
MODELS['avante'][5] = '94d349624d91490a898ec52d50e371c0'
MODELS['avante'][6] = '94d349624d91490a898ec52d50e371c0'
MODELS['avante'][7] = '3287fbb868094a1f8782e9cf9f32738c'
MODELS['avante'][8] = '94d349624d91490a898ec52d50e371c0'
MODELS['avante'][9] = 'd0c497c8b9b04f4d9e1e1e0c9297cc1f'
MODELS['avante'][10] = '3287fbb868094a1f8782e9cf9f32738c'
MODELS['avante'][13] = 'd0c497c8b9b04f4d9e1e1e0c9297cc1f'
MODELS['avante'][21] = 'd0c497c8b9b04f4d9e1e1e0c9297cc1f'

# ========================================================================
MODELS['trio']['s3_folder'] = '93'
MODELS['trio']['training_start_date'] = '2018-01-01'
MODELS['trio']['vector_model_s3_path'] = 's3://saiva-models/progress_note_embeddings/v1'
MODELS['trio'][7] = '86fbc7600dd54378bf7630f3ed60059d'
MODELS['trio'][42] = '9cf3c88cce99433ea7e9fb34a519016c'
MODELS['trio'][52] = 'd66eacdb13894fe9be2e29be8bc0a45e'
MODELS['trio'][55] = '9cf3c88cce99433ea7e9fb34a519016c'
MODELS['trio'][265] = '86fbc7600dd54378bf7630f3ed60059d'
MODELS['trio'][186] = 'd66eacdb13894fe9be2e29be8bc0a45e'
MODELS['trio'][21] = '86fbc7600dd54378bf7630f3ed60059d'
MODELS['trio'][1] = 'f3d1f812ef2343a6a6951b94e2536ead'
MODELS['trio'][194] = 'f53995bab7f64e988d1d554300ec1f98'
MODELS['trio'][273] = '86fbc7600dd54378bf7630f3ed60059d'
MODELS['trio'][274] = 'f3d1f812ef2343a6a6951b94e2536ead'
MODELS['trio'][275] = '8803d751893b4773a5ed8dd04729e67c'
MODELS['trio'][276] = 'd66eacdb13894fe9be2e29be8bc0a45e'
MODELS['trio'][277] = '9cf3c88cce99433ea7e9fb34a519016c'
MODELS['trio'][278] = '86fbc7600dd54378bf7630f3ed60059d'
MODELS['trio'][279] = '9cf3c88cce99433ea7e9fb34a519016c'
# ========================================================================
MODELS['infinity-infinity']['s3_folder'] = '49'
MODELS['infinity-infinity']['training_start_date'] = '2018-01-01'
MODELS['infinity-infinity']['vector_model_s3_path'] = 's3://saiva-models/progress_note_embeddings/v1'
MODELS['infinity-infinity'][75] = '9bfb02fa32f644aeafba299380742580'
MODELS['infinity-infinity'][89] = 'd32d239731844f4497e27edc51242b62'
# ========================================================================
MODELS['infinity-benchmark']['s3_folder'] = '48'
MODELS['infinity-benchmark']['training_start_date'] = '2018-01-01'
MODELS['infinity-benchmark']['vector_model_s3_path'] = 's3://saiva-models/progress_note_embeddings/v1'
MODELS['infinity-benchmark'][37] = '0a4579942cb844a4b73332d1824fe143'
# ========================================================================
MODELS['dycora']['s3_folder'] = '50'
MODELS['dycora']['training_start_date'] = '2018-01-01'
MODELS['dycora']['vector_model_s3_path'] = 's3://saiva-models/progress_note_embeddings/v1'
MODELS['dycora'][121] = '41238147855f4210ae98e067241f8dff'
MODELS['dycora'][302] = 'e6b3ffdf095747a7891746b01193db66'
# ========================================================================
MODELS['northshore']['s3_folder'] = '53'
MODELS['northshore']['training_start_date'] = '2018-01-01'
MODELS['northshore']['vector_model_s3_path'] = 's3://saiva-models/progress_note_embeddings/v1'
MODELS['northshore'][10] = '68ca9a93a6c24567a5570a687fb16033'
MODELS['northshore'][13] = 'f27dd0a1791b4c7aa9a7422547620fc5'
# =======================================python modelid_addition.py --env all=================================
MODELS['meridian']['s3_folder'] = '63'
MODELS['meridian']['training_start_date'] = '2018-01-01'
MODELS['meridian']['vector_model_s3_path'] = 's3://saiva-models/progress_note_embeddings/v1'
MODELS['meridian'][121] = 'fd5159759c8c474b872e12268e87f439'
MODELS['meridian'][122] = '4356cba3e6554d38a6d1189dc6c84708'
# ========================================================================
MODELS['palmgarden']['s3_folder'] = '98'
MODELS['palmgarden']['training_start_date'] = '2018-01-01'
MODELS['palmgarden']['vector_model_s3_path'] = 's3://saiva-models/progress_note_embeddings/v1'
MODELS['palmgarden'][4] = 'b1c5bf798a9a4eeea80b28c0c0ff4ca1'
MODELS['palmgarden'][7] = '619ce8c49c7c439aa60672e044dea2fd'
MODELS['palmgarden'][6] = '1f6306449c41489d8c2cf75df6457bc9'
MODELS['palmgarden'][14] = '187c3949db3c4ec5bdd69506484f35e9'
# ========================================================================
MODELS['vintage']['s3_folder'] = '97'
MODELS['vintage']['training_start_date'] = '2018-01-01'
MODELS['vintage']['vector_model_s3_path'] = 's3://saiva-models/progress_note_embeddings/v1'
MODELS['vintage'][1] = '5ae483a18048444ab735b10f1c9c33a4'
MODELS['vintage'][4] = '5ae483a18048444ab735b10f1c9c33a4'
MODELS['vintage'][11] = '5ae483a18048444ab735b10f1c9c33a4'
MODELS['vintage'][30] = '5ae483a18048444ab735b10f1c9c33a4'
MODELS['vintage'][31] = '08bc8beed1de4bcbb0580896c0756290'
MODELS['vintage'][34] = 'dd6d604362f143fdaf20a678081cf70d'
# ========================================================================
MODELS['mmh']['s3_folder'] = '94'
MODELS['mmh']['training_start_date'] = '2018-01-01'
MODELS['mmh']['vector_model_s3_path'] = 's3://saiva-models/progress_note_embeddings/v1'
MODELS['mmh'][1] = '0a599c2bd1de4849a1a3a53fc86ad522'
MODELS['mmh'][2] = 'b9b897bc1d444222b5b38892e6533cc5'
MODELS['mmh'][5] = '8f430751fffb46f18b250b36b26b86af'
# ========================================================================
MODELS['marquis']['s3_folder'] = '96'
MODELS['marquis']['training_start_date'] = '2018-07-01'
MODELS['marquis']['vector_model_s3_path'] = 's3://saiva-models/progress_note_embeddings/v1'
MODELS['marquis'][1] = 'ff8f0dfcef8f42e3a963afc2edcfff2d'
MODELS['marquis'][2] = 'cff4a9213a404fb7985f1bdbea8017b1'
MODELS['marquis'][3] = '8fd0dd809f2b4ea5b3534b71ca54c2c6'
MODELS['marquis'][4] = '34eb5a4cc36946d1bf0659b52ae8f4b3'
MODELS['marquis'][5] = 'ff8f0dfcef8f42e3a963afc2edcfff2d'
MODELS['marquis'][6] = '34eb5a4cc36946d1bf0659b52ae8f4b3'
MODELS['marquis'][7] = 'bf998c371ed5454f92ec87e35727c45f'
MODELS['marquis'][8] = '34eb5a4cc36946d1bf0659b52ae8f4b3'
MODELS['marquis'][11] = 'd5ac5d5e2d714867838d3be2937a127f'
MODELS['marquis'][13] = 'd5ac5d5e2d714867838d3be2937a127f'
MODELS['marquis'][14] = 'cff4a9213a404fb7985f1bdbea8017b1'
MODELS['marquis'][15] = 'ff8f0dfcef8f42e3a963afc2edcfff2d'
MODELS['marquis'][16] = 'ff8f0dfcef8f42e3a963afc2edcfff2d'
MODELS['marquis'][17] = '34eb5a4cc36946d1bf0659b52ae8f4b3'
MODELS['marquis'][18] = 'ff8f0dfcef8f42e3a963afc2edcfff2d'
MODELS['marquis'][19] = 'bf998c371ed5454f92ec87e35727c45f'
MODELS['marquis'][20] = '34eb5a4cc36946d1bf0659b52ae8f4b3'
MODELS['marquis'][21] = '10ebb0ef38af4dd988a2484738649657'
MODELS['marquis'][22] = '8fd0dd809f2b4ea5b3534b71ca54c2c6'
MODELS['marquis'][23] = '10ebb0ef38af4dd988a2484738649657'
MODELS['marquis'][24] = '10ebb0ef38af4dd988a2484738649657'
MODELS['marquis'][25] = '10ebb0ef38af4dd988a2484738649657'
MODELS['marquis'][26] = '15e5eb235489430997d9b29c0ebca172'
MODELS['marquis'][27] = 'bf998c371ed5454f92ec87e35727c45f'
MODELS['marquis'][28] = 'cff4a9213a404fb7985f1bdbea8017b1'
MODELS['marquis'][29] = '8fd0dd809f2b4ea5b3534b71ca54c2c6'
MODELS['marquis'][31] = 'cff4a9213a404fb7985f1bdbea8017b1'
MODELS['marquis'][32] = '8fd0dd809f2b4ea5b3534b71ca54c2c6'
MODELS['marquis'][33] = '8fd0dd809f2b4ea5b3534b71ca54c2c6'
MODELS['marquis'][34] = '15e5eb235489430997d9b29c0ebca172'
MODELS['marquis'][36] = 'ff8f0dfcef8f42e3a963afc2edcfff2d'
MODELS['marquis'][37] = 'ff8f0dfcef8f42e3a963afc2edcfff2d'
MODELS['marquis'][38] = 'ff8f0dfcef8f42e3a963afc2edcfff2d'
MODELS['marquis'][39] = 'ff8f0dfcef8f42e3a963afc2edcfff2d'
MODELS['marquis'][40] = '10ebb0ef38af4dd988a2484738649657'
MODELS['marquis'][42] = '10ebb0ef38af4dd988a2484738649657'
# ========================================================================
CLIENT_NPI_CONFIG = {}
CLIENT_NPI_CONFIG['phcp'] = ('1568974046', '1093169849', '1366480311', '1407857089', '1588658348',
                             '1407159536')

# ========================Copy trio to saiva_demo=========================
MODELS['saiva_demo'] = MODELS['trio']

# ========================================================================
SHOW_IN_REPORT_CUTOFF = {
    'avante': defaultdict(lambda: 15),
    'trio': defaultdict(lambda: 15),
    'infinity-infinity': defaultdict(lambda: 15),
    'infinity-benchmark': defaultdict(lambda: 15),
    'dycora': defaultdict(lambda: 15),
    'northshore': defaultdict(lambda: 15),
    'meridian': defaultdict(lambda: 15),
    'vintage': defaultdict(lambda: 15),
    'palmgarden': defaultdict(lambda: 15),
    'mmh': defaultdict(lambda: 15),
    'marquis': defaultdict(lambda: 15),
    'saiva_demo': defaultdict(lambda: 15),
    'phcp': defaultdict(lambda: 15),
}
SHOW_IN_REPORT_CUTOFF['mmh'][2] = 10
SHOW_IN_REPORT_CUTOFF['mmh'][5] = 10
SHOW_IN_REPORT_CUTOFF['marquis'][5] = 30   # brentwood, census 123 (as of May 1st, 2021), by unit
SHOW_IN_REPORT_CUTOFF['marquis'][11] = 20  # willow springs, census 140, facility
SHOW_IN_REPORT_CUTOFF['marquis'][15] = 30  # oakland, census 174, unit
SHOW_IN_REPORT_CUTOFF['marquis'][18] = 30  # elmhurst, census 189, facility
SHOW_IN_REPORT_CUTOFF['marquis'][19] = 25  # laurelbrook, census 165, facility
SHOW_IN_REPORT_CUTOFF['marquis'][26] = 20  # colingswood, census 141, facility
SHOW_IN_REPORT_CUTOFF['marquis'][27] = 20  # jewish home, census 130, facility
SHOW_IN_REPORT_CUTOFF['marquis'][29] = 30  # roosevelt, census 204, unit
SHOW_IN_REPORT_CUTOFF['marquis'][32] = 30  # canterbury, census 171, unit
SHOW_IN_REPORT_CUTOFF['marquis'][33] = 45  # woodbine, census 282, unit
SHOW_IN_REPORT_CUTOFF['marquis'][37] = 20  # linconlwood, census 136, facility
