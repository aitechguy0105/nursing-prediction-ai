import mlflow
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import pickle
from omegaconf import OmegaConf
import re
from eliot import log_message


from saiva.model.explanations.config import FEATURE_TYPE_MAPPING
from saiva.model.shared.utils import url_encode_cols

def preprocess_final_df(input_filepath, model_type, test_start_date, test_end_date, facility_ids=None):
    final = pd.read_parquet(input_filepath)
    # UPT model doesn't need the rows that with payername contains 'hospice'
    if model_type=='model_upt':
        final = final[~(final['payername'].str.contains('hospice', case=False, regex=True, na=False))]
    log_message(message_type='info', message=f'Final df shape: {final.shape}')

    final['client'] = final['masterpatientid'].apply(lambda z: z.split('_')[0])
    final['LFS'] = final['days_since_last_admission']

    """ We increment the census date by 1, since the prediction day always includes data upto last night.
    This means for every census date the data is upto previous night. 
    """
    final['censusdate'] = (pd.to_datetime(final['censusdate']) + timedelta(days=1))

    final[f'target_3_day_{model_type.lower()}'] = final[f'target_3_day_{model_type.lower()}'].fillna(False)
    final['facility'] = final['facilityid']
    final['facility'] = final['facility'].astype('category')

    test = final.loc[(final.censusdate >= test_start_date) & (final.censusdate <= test_end_date)]

    if facility_ids is not None:
        test = test[test['facilityid'].isin(facility_ids)]
    log_message(message_type='info', message=f'Shape of test data after using only test dates and facilities provided: {test.shape}')
    return test


def download_model_cat_cols_list_mlflow(modelid, local_path='/src/saiva/notebooks'):
    mlflow.set_tracking_uri('http://mlflow.saiva-dev')
    # local_path = '/src/saiva/notebooks'
    return mlflow.artifacts.download_artifacts(run_id=modelid, artifact_path=f'cate_columns.pickle', dst_path=local_path)

def download_model_input_features_list_mlflow(modelid, local_path='/src/saiva/notebooks'):
    mlflow.set_tracking_uri('http://mlflow.saiva-dev')
    # local_path = '/src/saiva/notebooks'
    return mlflow.artifacts.download_artifacts(run_id=modelid, artifact_path=f'input_features.csv', dst_path=local_path)

def fill_na(df, model):
    """This function tried to mimic the fill nan function from daily prediction runs.
    for detailed view you can take a look at - 
    https://github.com/saivaai/saiva-ml/blob/dev/models/saiva-3-day-hosp-v6/src/saiva/model/base_model.py#L130
    """
    # we load the config that is default one and used by actual prediction pipeline
    default_config = OmegaConf.load('/src/saiva/conf/prediction/defaults.yaml')
    
    all_feats = model.feature_name()
    x_frame = df.reindex(columns=all_feats)

    for item in default_config.postprocessing.missing_column_fill_values:
        pattern, fill_value = item.get('pattern'), item.get('value')
        mask = x_frame.columns.str.contains(pattern, regex=True)
        x_frame.loc[:, mask] = x_frame.loc[:, mask].fillna(fill_value)
    return x_frame

def feature_logging(facilityid, x_frame, model_feats, client):
    """
    Map feature names to groups
    Eg:
    TOTAL_FEATURES_BEFORE_DROP: 3500        
    TOTAL_FEATURES_AFTER_DROP: 2500
    TOTAL_FEATURES_DROPPED: 1500
    TOTAL_FEATURES_MISSING: 2500 - (3500 - 1500) 
    """
    training_feats = pd.DataFrame({'feature': list(x_frame.columns)})
    training_feats['feature_group'] = training_feats.feature.replace(
        FEATURE_TYPE_MAPPING,
        regex=True
    )
    features = training_feats.groupby('feature_group')['feature_group'].count().to_dict()
    log_message(message_type='info',
                title='MODEL_FEATURES: TOTAL_FEATURES_BEFORE_DROP',
                feature_count=len(training_feats.feature),
                feature_group=features,
                facilityid=facilityid,
                client=client,
                )
    # ====================================================================================
    dropped_columns = set(training_feats.feature).difference(set(model_feats.feature))
    log_message(message_type='info',
                title='MODEL_FEATURES: TOTAL_FEATURES_DROPPED',
                feature_count=len(dropped_columns),
                features=f'DROPPED_COLUMNS: {dropped_columns}',
                facilityid=facilityid,
                client=client,
                )
    # ====================================================================================
    missing_columns = set(model_feats.feature).difference(set(training_feats.feature))
    missing_feats = pd.DataFrame({'feature': list(missing_columns)})
    missing_feats['feature_group'] = missing_feats.feature.replace(
        FEATURE_TYPE_MAPPING,
        regex=True
    )
    features = missing_feats.groupby('feature_group')['feature_group'].count().to_dict()
    log_message(message_type='info',
                title='MODEL_FEATURES: TOTAL_FEATURES_MISSING',
                feature_count=len(missing_columns),
                feature_group=features,
                features=f'MISSING_COLUMNS: {missing_columns}',
                facilityid=facilityid,
                client=client,
                )
    # ====================================================================================
    model_feats['feature_group'] = model_feats.feature.replace(
        FEATURE_TYPE_MAPPING,
        regex=True
    )
    features = model_feats.groupby('feature_group')['feature_group'].count().to_dict()
    log_message(message_type='info',
                title='MODEL_FEATURES: TOTAL_FEATURES_AFTER_DROP',
                feature_count=len(model_feats.feature),
                feature_group=features,
                facilityid=facilityid,
                client=client
                )
    
def prep(df, model, client, iden_cols=None, pandas_categorical=None, categorical_columns=None, target_col=None):  
    # Required for latest LGBM 
    df = url_encode_cols(df)
    
    # get the targets
    target_3_day = df[target_col].astype('float32').values
    
    # -----------------------------Handle empty values------------------------------
    log_message(message_type='info', message='fill_na triggered...', client=client)
    x_frame =  fill_na(df, model)
    numeric_col = x_frame.select_dtypes(include=['number']).columns
    x_frame.loc[:, numeric_col] = x_frame.loc[:,numeric_col].astype('float32')

    # add client name to facility id
    x_frame['facility'] = re.sub('_*v6.*', '', client.lower()) + "_" + df['facilityid'].astype(str)
    x_frame['facility'] = x_frame['facility'].astype('category')
    
    if pandas_categorical is not None:
        log_message(message_type='info', message='converting categorical variables to categories...', client=client)
        for col, category in zip(categorical_columns, pandas_categorical):
            log_message(message_type='info', message=f'converting {col} categories...', client=client)
            x_frame[col] = x_frame[col].astype('category')
            x_frame[col] = x_frame[col].cat.set_categories(category)
    

    if iden_cols is None:
        raise ValueError('iden_cols is None, please provide those columns list.')
    idens = df.loc[:, iden_cols]
    log_message(message_type='info', message='extracting IDENS columns...', client=client)
    # add 'long_short_term' column to indens
    short_term_cond = ((x_frame.payertype == 'Managed Care')|(x_frame.payertype == 'Medicare A'))
    idens.loc[short_term_cond,'long_short_term']='short'
    
    nonType_cond = (x_frame.payertype == 'no payer info')
    if nonType_cond.sum()>0:
        print(f'{nonType_cond.sum()} patient days have no payer info')
        idens.loc[nonType_cond,'long_short_term']='no payer info'
    
    idens.loc[~(short_term_cond|nonType_cond),'long_short_term']='long'
    idens = idens.reset_index(drop=True)

    x_frame[categorical_columns] = x_frame[categorical_columns].apply(lambda col: col.cat.codes).replace({-1: np.nan})

    # reindex the x_frame again to make sure the order of the columns is the same as the training set
    all_feats = model.feature_name()
    log_message(message_type='info', message='reindexing x_frame to get selected features...', client=client)
    x_frame = x_frame.reindex(columns=all_feats)
    x_frame.reset_index(drop=True, inplace=True)
    x_frame = x_frame.to_numpy(dtype=np.float32)
    return x_frame, target_3_day, idens
