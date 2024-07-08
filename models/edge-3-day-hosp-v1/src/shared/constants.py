import os
from collections import defaultdict

import s3fs

# ========================================================================
CLIENT = 'infinity-infinity'

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
    'avante': defaultdict(lambda: '4e78363600a14e65866d8c1ef7ab28fe'),
    'trio': defaultdict(lambda: '76c04bc9875e4d4091277c7e186666f4'),
    'infinity-infinity': defaultdict(lambda: '8b099bee6027412da95811bf01af3f80'),
    'infinity-benchmark': defaultdict(lambda: '8b099bee6027412da95811bf01af3f80'),
}
# ========================================================================
MODELS['avante']['s3_folder'] = '1'
MODELS['avante']['training_start_date'] = '2018-01-01'
MODELS['avante']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
MODELS['avante'][7] = 'e9943c54f0e14c04ba59d1bf53120bf9'
MODELS['avante'][21] = '4e78363600a14e65866d8c1ef7ab28fe'
MODELS['avante'][4] = 'e9943c54f0e14c04ba59d1bf53120bf9'
MODELS['avante'][1] = '9ee8d86ae829409eaecffddf6634d605'
MODELS['avante'][6] = '9ee8d86ae829409eaecffddf6634d605'
# ========================================================================
MODELS['trio']['s3_folder'] = '30'
MODELS['trio']['training_start_date'] = '2018-01-01'
MODELS['trio']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
MODELS['trio'][7] = '76c04bc9875e4d4091277c7e186666f4'
MODELS['trio'][42] = '37cd4fd1ee8f42c1819886508547f919'
MODELS['trio'][52] = 'f4c8b357ad0842b2a4c6c9a4e2dff553'
MODELS['trio'][278] = '84069f7d0717493e91b80200a7fd7893'
MODELS['trio'][55] = '5ae5064a3da943c6951556acceb532a7'
# ========================================================================
# MODELS['infinity-infinity']['s3_folder'] = '38'
# MODELS['infinity-infinity']['training_start_date'] = '2018-01-01'
# MODELS['infinity-infinity']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
# MODELS['infinity-infinity'][75] = '56dce913f9d1446c8245d614c453ca0e'

MODELS['infinity-infinity']['s3_folder'] = '52'
MODELS['infinity-infinity']['training_start_date'] = '2018-01-01'
MODELS['infinity-infinity']['progress_note_model_s3_folder'] = 'progress_note_embeddings/edge-v1'
MODELS['infinity-infinity'][75] = '8b099bee6027412da95811bf01af3f80'
# ========================================================================
# MODELS['infinity-benchmark']['s3_folder'] = '37'
# MODELS['infinity-benchmark']['training_start_date'] = '2018-01-01'
# MODELS['infinity-benchmark']['progress_note_model_s3_folder'] = 'progress_note_embeddings/v1'
# MODELS['infinity-benchmark'][37] = '0e04cc53e47f443b9543e8f6b7b250c4'

MODELS['infinity-benchmark']['s3_folder'] = '52'
MODELS['infinity-benchmark']['training_start_date'] = '2018-01-01'
MODELS['infinity-benchmark']['progress_note_model_s3_folder'] = 'progress_note_embeddings/edge-v1'
MODELS['infinity-benchmark'][37] = '8b099bee6027412da95811bf01af3f80'