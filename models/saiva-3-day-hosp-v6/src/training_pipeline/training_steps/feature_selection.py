import logging
import sys
import typing

import fire
from prettytable import PrettyTable
import pandas as pd
from eliot import to_file

from src.saiva.model.shared.constants import LOCAL_TRAINING_CONFIG_PATH
from src.training_pipeline.shared.helpers import DatasetProvider, TrainingStep
from saiva.training import load_config
from src.training_pipeline.shared.utils import setup_sentry


to_file(sys.stdout)  # ECS containers log stdout to CloudWatch


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CURRENT_STEP = TrainingStep.FEATURE_SELECTION


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


### Remove features which have 100% 0 values
def na_analysis(df: pd.DataFrame):
    lst = []
    cols = []
    total_rows = df.shape[0]
    cols = df.columns[df.columns.str.contains('cumidx|cumsum|days_since_last_event|na_indictator|vtl_|notes_')]
    for col in cols:
        # Sum of NaN values in a column
        na_values = max(df[col].eq(0).sum(), df[col].eq(9999).sum(), df[col].isnull().sum())
        lst.extend([[col, total_rows, na_values, (na_values / total_rows) * 100]])
        if ((na_values / total_rows) * 100) >= 99 and (col not in cols):
            cols.append(col)

    return lst


# Not the most efficient code but not optimizing since the cell runs pretty fast
def get_feature_group_counts(*, base_df: pd.DataFrame, feature_groups: dict):
    training_feats = base_df.columns
    features = {}
    for grp in feature_groups:
        features[grp] = len([x for x in training_feats if x in feature_groups[grp]])
    return features

def get_cumulative_group_counts(*, base_df: pd.DataFrame):
    training_feats = pd.DataFrame({'feature': list(base_df.columns)})
    training_feats['feature_group'] = training_feats.feature.replace(
        CUMULATIVE_GROUP_MAPPING,
        regex=True
    )
    features = training_feats.groupby('feature_group')['feature_group'].count().to_dict()
    cumulative_cols = CUMULATIVE_GROUP_MAPPING.values()
    features = {k: features.get(k, 0) for k in cumulative_cols}

    return features


def feature_selection(
    *, 
    run_id: str,
    force_regenerate: typing.Optional[bool] = False,
    model_type: typing.Optional[str] = 'MODEL_UPT',
    disable_sentry: typing.Optional[bool] = False,
    **kwargs
):
    """Cumulative feature groups

        :param run_id: the run id
        :param force_regenerate: force regeneration of the dataset
    """
    setup_sentry(run_id=run_id, disable_sentry=disable_sentry)

    dataset_provider = DatasetProvider(run_id=run_id, force_regenerate=force_regenerate)

    dataset_provider.download_config(step=TrainingStep.previous_step(CURRENT_STEP))
    config = load_config(LOCAL_TRAINING_CONFIG_PATH)
    model_type = model_type.lower()

    info_cols = config.automatic_training.datasets_generation.iden_cols + [f'positive_date_{model_type}', f'target_3_day_{model_type}']

    if dataset_provider.does_file_exist(filename=f'{model_type}/final_cleaned_df', step=CURRENT_STEP):
        log.info('Dataset final_cleaned_df already exists - skipping step.')
        return

    base_df = dataset_provider.get(dataset_name='final_df', step=TrainingStep.POST_FEATURE_ENGINEERING_MERGE)

    feature_groups = dataset_provider.load_json(filename='feature_groups', step=TrainingStep.POST_FEATURE_ENGINEERING_MERGE)

    feature_drop_stats = {}
    cumulative_feature_drop_stats = {}

    feature_group_count = get_feature_group_counts(base_df=base_df, feature_groups=feature_groups)
    cumulative_group_count = get_cumulative_group_counts(base_df=base_df)

    for grp in feature_groups:
        feature_drop_stats[grp] = {'before_drop_count': feature_group_count[grp]}

    for grp in cumulative_group_count:
        cumulative_feature_drop_stats[grp] = {'before_drop_count': cumulative_group_count[grp]}

    # Dropping columns with all null values
    non_idens_cols_all_null = [col for col in base_df.columns if base_df[col].isnull().all() and col not in info_cols]

    # Dropping columns with same value in all rows
    cols_with_single_value = []
    for col in base_df.columns:
        if len(base_df[col].value_counts()) == 1 and base_df[col].value_counts().iloc[0] == len(base_df) and col not in info_cols:
            cols_with_single_value.append(col)
    # dropping both list of columns
    base_df.drop(list(set(cols_with_single_value + non_idens_cols_all_null)),inplace=True,axis=1)

    log.info('Storing all_null_dropped_col_names')
    cols_to_drop = {'single_valued_columns': cols_with_single_value, 'all_null_columns': non_idens_cols_all_null}
    # Dump the merged dictionary into the JSON file
    dataset_provider.store_json(filename=f'{model_type}/all_null_dropped_col_names', step=CURRENT_STEP, data=cols_to_drop)

    df_na = pd.DataFrame(
        na_analysis(base_df),
        columns=['column_name', 'total_count', 'null_values', '%_null_values']
    )

    df_na.sort_values(['%_null_values'], ascending=False, inplace=True)

    drop_cols = df_na[
        (df_na['%_null_values'] >= 99.9) & (~df_na['column_name'].str.startswith('hosp_target'))
        ]['column_name']
    base_df.drop(
        drop_cols,
        axis=1,
        inplace=True
    )

    feature_group_count = get_feature_group_counts(base_df=base_df, feature_groups=feature_groups)
    cumulative_group_count = get_cumulative_group_counts(base_df=base_df)

    total_before_drop = 0
    total_after_drop = 0
    for grp in feature_groups:
        feature_drop_stats[grp]['after_drop_count'] = feature_group_count[grp]
        dropped_percentage = (feature_drop_stats[grp]['before_drop_count'] - feature_drop_stats[grp]['after_drop_count']) / feature_drop_stats[grp]['before_drop_count']
        feature_drop_stats[grp]['dropped_percentage'] = "{:.0%}".format(dropped_percentage)
        total_before_drop += feature_drop_stats[grp]['before_drop_count']
        total_after_drop += feature_drop_stats[grp]['after_drop_count']
    dropped_percentage = (total_before_drop - total_after_drop) / total_before_drop
    feature_drop_stats['Total'] = {'before_drop_count': total_before_drop, 'after_drop_count': total_after_drop, 'dropped_percentage': "{:.0%}".format(dropped_percentage)}

    total_before_drop = 0
    total_after_drop = 0
    for grp in cumulative_group_count:
        cumulative_feature_drop_stats[grp]['after_drop_count'] = cumulative_group_count[grp]
        if cumulative_feature_drop_stats[grp]['before_drop_count'] > 0:
            dropped_percentage = (cumulative_feature_drop_stats[grp]['before_drop_count'] - cumulative_feature_drop_stats[grp]['after_drop_count'])/cumulative_feature_drop_stats[grp]['before_drop_count']
        else:
            dropped_percentage = 0
        cumulative_feature_drop_stats[grp]['dropped_percentage'] = "{:.0%}".format(dropped_percentage)
        total_before_drop += cumulative_feature_drop_stats[grp]['before_drop_count']
        total_after_drop += cumulative_feature_drop_stats[grp]['after_drop_count']

    if total_before_drop > 0:
        dropped_percentage = (total_before_drop-total_after_drop)/total_before_drop
    else:
        dropped_percentage = 0
    cumulative_feature_drop_stats['Total'] = {'before_drop_count': total_before_drop, 'after_drop_count': total_after_drop, 'dropped_percentage': "{:.0%}".format(dropped_percentage)}

    log.info('Storing cumulative_feature_drop_stats and feature_drop_stats')
    dataset_provider.store_json(filename=f'{model_type}/feature_drop_stats', step=CURRENT_STEP, data=feature_drop_stats)
    dataset_provider.store_json(filename=f'{model_type}/cumulative_feature_drop_stats', step=CURRENT_STEP, data=cumulative_feature_drop_stats)

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

    log.info('Storing feature_group_drop_stats')
    dataset_provider.store_txt(filename=f'{model_type}/feature_group_drop_stats', step=CURRENT_STEP, data=str(x))

    x = PrettyTable()
    x.title = 'Feature Cumulative Group Drop Stats'
    x.field_names = ["Feature Group", "Before Feature Reduction", "After Feature Reduction", "% of Dropped Features"]
    for grp in cumulative_feature_drop_stats:
        x.add_row(
            [grp, cumulative_feature_drop_stats[grp]['before_drop_count'], cumulative_feature_drop_stats[grp]['after_drop_count'],
             cumulative_feature_drop_stats[grp]['dropped_percentage']]
        )

    log.info('Storing feature_cumulative_drop_stats')
    dataset_provider.store_txt(filename=f'{model_type}/feature_cumulative_drop_stats', step=CURRENT_STEP, data=str(x))

    dataset_provider.set(dataset_name=f'{model_type}/final_cleaned_df', step=CURRENT_STEP, df=base_df)

    dataset_provider.store_config(step=CURRENT_STEP, prefix=f"/{model_type}")


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes
    fire.Fire(feature_selection)