import gc
import sys

import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')
from shared.generate_base_features import sorter_and_deduper


def get_lab_features(base, patient_lab_results, training=False):
    """
    desc:
        -Process the Lab related features

    :param basedf: dataframe
    :param patient_lab_results: dataframe
    :param training: bool
    """
    patient_lab_results = sorter_and_deduper(
        patient_lab_results,
        sort_keys=['masterpatientid', 'resultdate', 'profiledescription', 'reportdesciption'],
        unique_keys=['masterpatientid', 'resultdate', 'profiledescription', 'reportdesciption']
    )

    if training:
        lab_types = (patient_lab_results['profiledescription'].value_counts()[:75].index.tolist())
    else:
        lab_types = (patient_lab_results['profiledescription'].value_counts().index.tolist())

    patient_lab_results = (
        patient_lab_results[
            patient_lab_results['profiledescription'].isin(lab_types)
        ].copy().reset_index()
    )

    patient_lab_results['lab_result'] = patient_lab_results.apply(
        lambda x: f"{x['profiledescription'].replace(' ', '_')}__{x['abnormalitydescription'].replace(' ', '_')}",
        axis=1
    )
    patient_lab_results['abnormalitydescription'].value_counts()
    patient_lab_results = pd.concat(
        [
            patient_lab_results,
            pd.get_dummies(patient_lab_results['lab_result'], prefix='labs_')
        ],
        axis=1,
    )
    lab_cols = [c for c in patient_lab_results.columns if c.startswith("labs__")]
    patient_lab_results['resultdate'] = patient_lab_results['resultdate'].dt.normalize()

    # there will be multiple days per patient - group the lab values by patient, day taking the max()
    #    i.e. organize it back by patient, day
    lab_results_grouped_by_day = patient_lab_results.groupby(['masterpatientid', 'resultdate'], as_index=False)[
        lab_cols].max()
    assert lab_results_grouped_by_day.isna().any(axis=None) == False
    assert lab_results_grouped_by_day.duplicated(subset=['masterpatientid', 'resultdate']).any() == False

    merged_df = base.merge(
        lab_results_grouped_by_day,
        how='left',
        left_on=['masterpatientid', 'censusdate'],
        right_on=['masterpatientid', 'resultdate']
    )
    
    # Drop resultdate column as its not required
    merged_df = merged_df.drop(columns=['resultdate'])
    
    assert merged_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
    log_message(message_type='info', message='Feature processing activity completed')
    # =============Trigger garbage collection to free-up memory ==================
    del base
    gc.collect()

    return merged_df
