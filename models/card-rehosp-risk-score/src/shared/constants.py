import os
from collections import defaultdict

import s3fs

# ========================================================================
CLIENT = 'trio'
START_DATE = '2020-12-01'
END_DATE = '2021-01-31'

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
    'trio': defaultdict(lambda: '935ec28e87c04484a92a749be49eeef3'),
}
# ========================================================================
MODELS['avante']['s3_folder'] = '99'
MODELS['avante']['training_start_date'] = '2018-01-01'
MODELS['avante']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
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
MODELS['trio']['s3_folder'] = '137'
MODELS['trio']['training_start_date'] = '2018-01-01'
MODELS['trio']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
MODELS['trio'][186] = '935ec28e87c04484a92a749be49eeef3'
MODELS['trio'][194] = '935ec28e87c04484a92a749be49eeef3'
MODELS['trio'][1] = 'c380bc99e86c4d39a4d5261df616251c'
MODELS['trio'][21] = 'dd70e65e707944d0bb2d773bcbd9e63d'
MODELS['trio'][265] = '76fa4d75e597450db43584466b021c52'
MODELS['trio'][273] = 'a7451ad6f5654839a813d5946360eedd'
MODELS['trio'][274] = '76fa4d75e597450db43584466b021c52'
MODELS['trio'][275] = '935ec28e87c04484a92a749be49eeef3'
MODELS['trio'][276] = '935ec28e87c04484a92a749be49eeef3'
MODELS['trio'][277] = '935ec28e87c04484a92a749be49eeef3'
MODELS['trio'][278] = '76fa4d75e597450db43584466b021c52'
MODELS['trio'][279] = '76fa4d75e597450db43584466b021c52'
MODELS['trio'][42] = 'fb8233178fe1474089a6d88f2932b1f3'
MODELS['trio'][52] = 'fb8233178fe1474089a6d88f2932b1f3'
MODELS['trio'][55] = '935ec28e87c04484a92a749be49eeef3'
MODELS['trio'][7] = '935ec28e87c04484a92a749be49eeef3'
# ========================================================================


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
    'midwest': defaultdict(lambda: 15)
}
SHOW_IN_REPORT_CUTOFF['mmh'][2] = 10
SHOW_IN_REPORT_CUTOFF['mmh'][5] = 10
SHOW_IN_REPORT_CUTOFF['marquis'][29] = 30
SHOW_IN_REPORT_CUTOFF['marquis'][5] = 30
SHOW_IN_REPORT_CUTOFF['marquis'][19] = 30
SHOW_IN_REPORT_CUTOFF['marquis'][32] = 30
SHOW_IN_REPORT_CUTOFF['marquis'][33] = 45
SHOW_IN_REPORT_CUTOFF['midwest'][15] = 10
