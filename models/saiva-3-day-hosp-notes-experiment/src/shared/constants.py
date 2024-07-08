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
    'avante': defaultdict(lambda: '07305b277385447ba0d989e6f91a2a80'),
    'trio': defaultdict(lambda: '7fbb651f4d9542b1867bf1a34276df99'),
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
MODELS['avante']['s3_folder'] = '125'
MODELS['avante']['training_start_date'] = '2018-06-01'
# MODELS['avante']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
MODELS['avante'][1] = '6d3b884be04d4b3ebf43989897daa283'
MODELS['avante'][3] = '07305b277385447ba0d989e6f91a2a80'
MODELS['avante'][4] = '68fabe126a3a4946b2d350a8615232dd'
MODELS['avante'][5] = '07305b277385447ba0d989e6f91a2a80'
MODELS['avante'][6] = '6d3b884be04d4b3ebf43989897daa283'
MODELS['avante'][7] = '07305b277385447ba0d989e6f91a2a80'
MODELS['avante'][8] = '07305b277385447ba0d989e6f91a2a80'
MODELS['avante'][9] = '07305b277385447ba0d989e6f91a2a80'
MODELS['avante'][10] = '07305b277385447ba0d989e6f91a2a80'
MODELS['avante'][13] = '6d3b884be04d4b3ebf43989897daa283'
MODELS['avante'][21] = '07305b277385447ba0d989e6f91a2a80'

# ========================================================================
MODELS['trio']['s3_folder'] = '130'
MODELS['trio']['training_start_date'] = '2018-01-01'
MODELS['trio']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
MODELS['trio'][186] = '7fbb651f4d9542b1867bf1a34276df99'
MODELS['trio'][194] = '22a1b824062f49359e437827c0980d30'
MODELS['trio'][1] = '2b50729db0df48f29b87df1b011ac1e4'
MODELS['trio'][21] = '83748d7f518a47b2abbd952474101d12'
MODELS['trio'][265] = '83748d7f518a47b2abbd952474101d12'
MODELS['trio'][273] = '83748d7f518a47b2abbd952474101d12'
MODELS['trio'][274] = '7fbb651f4d9542b1867bf1a34276df99'
MODELS['trio'][275] = '83748d7f518a47b2abbd952474101d12'
MODELS['trio'][276] = '22a1b824062f49359e437827c0980d30'
MODELS['trio'][277] = '7fbb651f4d9542b1867bf1a34276df99'
MODELS['trio'][278] = '83748d7f518a47b2abbd952474101d12'
MODELS['trio'][279] = '7fbb651f4d9542b1867bf1a34276df99'
MODELS['trio'][42] = '22a1b824062f49359e437827c0980d30'
MODELS['trio'][52] = '7fbb651f4d9542b1867bf1a34276df99'
MODELS['trio'][55] = '7fbb651f4d9542b1867bf1a34276df99'
MODELS['trio'][7] = '83748d7f518a47b2abbd952474101d12'

# ========================================================================
MODELS['infinity-infinity']['s3_folder'] = '49'
MODELS['infinity-infinity']['training_start_date'] = '2018-01-01'
MODELS['infinity-infinity']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
MODELS['infinity-infinity'][75] = '9bfb02fa32f644aeafba299380742580'
MODELS['infinity-infinity'][89] = 'd32d239731844f4497e27edc51242b62'
# ========================================================================
MODELS['infinity-benchmark']['s3_folder'] = '48'
MODELS['infinity-benchmark']['training_start_date'] = '2018-01-01'
MODELS['infinity-benchmark']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
MODELS['infinity-benchmark'][37] = '0a4579942cb844a4b73332d1824fe143'
# ========================================================================
MODELS['dycora']['s3_folder'] = '50'
MODELS['dycora']['training_start_date'] = '2018-01-01'
MODELS['dycora']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
MODELS['dycora'][121] = '41238147855f4210ae98e067241f8dff'
MODELS['dycora'][302] = 'e6b3ffdf095747a7891746b01193db66'
# ========================================================================
MODELS['northshore']['s3_folder'] = '53'
MODELS['northshore']['training_start_date'] = '2018-01-01'
MODELS['northshore']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
MODELS['northshore'][10] = '68ca9a93a6c24567a5570a687fb16033'
MODELS['northshore'][13] = 'f27dd0a1791b4c7aa9a7422547620fc5'
# =======================================python modelid_addition.py --env all=================================
MODELS['meridian']['s3_folder'] = '63'
MODELS['meridian']['training_start_date'] = '2018-01-01'
MODELS['meridian']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
MODELS['meridian'][121] = 'fd5159759c8c474b872e12268e87f439'
MODELS['meridian'][122] = '4356cba3e6554d38a6d1189dc6c84708'
# ========================================================================
MODELS['palmgarden']['s3_folder'] = '98'
MODELS['palmgarden']['training_start_date'] = '2018-01-01'
MODELS['palmgarden']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
MODELS['palmgarden'][4] = 'b1c5bf798a9a4eeea80b28c0c0ff4ca1'
MODELS['palmgarden'][7] = '619ce8c49c7c439aa60672e044dea2fd'
MODELS['palmgarden'][6] = '1f6306449c41489d8c2cf75df6457bc9'
MODELS['palmgarden'][14] = '187c3949db3c4ec5bdd69506484f35e9'
# ========================================================================
MODELS['vintage']['s3_folder'] = '97'
MODELS['vintage']['training_start_date'] = '2018-01-01'
MODELS['vintage']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
MODELS['vintage'][1] = '5ae483a18048444ab735b10f1c9c33a4'
MODELS['vintage'][4] = '5ae483a18048444ab735b10f1c9c33a4'
MODELS['vintage'][11] = '5ae483a18048444ab735b10f1c9c33a4'
MODELS['vintage'][30] = '5ae483a18048444ab735b10f1c9c33a4'
MODELS['vintage'][31] = '08bc8beed1de4bcbb0580896c0756290'
MODELS['vintage'][34] = 'dd6d604362f143fdaf20a678081cf70d'
# ========================================================================
MODELS['mmh']['s3_folder'] = '94'
MODELS['mmh']['training_start_date'] = '2018-01-01'
MODELS['mmh']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
MODELS['mmh'][1] = '0a599c2bd1de4849a1a3a53fc86ad522'
MODELS['mmh'][2] = 'b9b897bc1d444222b5b38892e6533cc5'
MODELS['mmh'][5] = '8f430751fffb46f18b250b36b26b86af'
# ========================================================================
MODELS['marquis']['s3_folder'] = '96'
MODELS['marquis']['training_start_date'] = '2018-07-01'
MODELS['marquis']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
MODELS['marquis'][3] = '8fd0dd809f2b4ea5b3534b71ca54c2c6'
MODELS['marquis'][7] = 'bf998c371ed5454f92ec87e35727c45f'
MODELS['marquis'][13] = 'd5ac5d5e2d714867838d3be2937a127f'
MODELS['marquis'][26] = '15e5eb235489430997d9b29c0ebca172'
MODELS['marquis'][29] = '8fd0dd809f2b4ea5b3534b71ca54c2c6'
MODELS['marquis'][31] = 'cff4a9213a404fb7985f1bdbea8017b1'
MODELS['marquis'][36] = 'ff8f0dfcef8f42e3a963afc2edcfff2d'

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
# SHOW_IN_REPORT_CUTOFF['marquis'][29] = 45
