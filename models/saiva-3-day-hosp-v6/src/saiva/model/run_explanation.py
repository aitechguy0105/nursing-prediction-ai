import pandas as pd
import shap
from eliot import log_message, start_action

from .explanations.config import FEATURE_GROUP_MAPPING, FEATURE_TYPE_MAPPING
from .explanations.mapper import DataMapper
from .explanations.preprocess import DataProcessor
from .shared.utils import url_decode_cols

def generate_explanations(
        *,
        model,
        final_x,
        final_idens,
        raw_data_dict,
        client,
        ml_model,
        s3_location_path_prefix,
        save_outputs_in_local=False,
        local_folder=None
):
    with start_action(action_type="generating explanations."):
        log_message(
            message_type='info', 
            message='generating shap explanations.',
            final_x_shape = final_x.shape
        )
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(final_x)

        shap_results = []

        for idx, row in final_x.iterrows():

            decoded_cols = url_decode_cols(final_x.columns)
            shaps = pd.DataFrame(
                {
                    "feature": decoded_cols,
                    "attribution_score": shap_values[idx] if (model.model.params.get('objective') != 'binary') \
                                        else shap_values[1][idx],
                    "feature_value": final_x.iloc[idx],
                }
            )

            shaps["masterpatientid"] = final_idens.iloc[idx].masterpatientid
            shaps["facilityid"] = final_idens.iloc[idx].facilityid
            shaps["censusdate"] = final_idens.iloc[idx].censusdate
            shaps["calculate_shap"] = final_idens.iloc[idx].calculate_shap
            shaps['human_readable_name'] = ''
            shaps['mapping_status'] = 'NOT_MAPPED'

            shap_results.append(shaps)

        results = pd.concat(shap_results)
        # =================================Preprocess the data before mapping=========================================
        log_message(
            message_type='info', 
            message='mapping shap explanations.',
            final_x_shape = final_x.shape
        )
        results['mapped_feature'] = results['feature'].replace(
            FEATURE_GROUP_MAPPING,
            regex=True
        )
        results['feature_type'] = results['feature'].replace(
            FEATURE_TYPE_MAPPING,
            regex=True
        )
        progress_notes_shap_results = results[results['feature_type'] == 'Progress Note'].copy()
        progress_notes_shap_results['sum_attribution_score'] = progress_notes_shap_results['attribution_score']
        results = results[results['feature_type'] != 'Progress Note']
        results['day_count'] = results['feature'].str.extract(r'_(\d+)_day')
        # day_count = NaN for `all__` features
        # All NaN are given high value of 100 so that they come last while sorting by day_count
        results['day_count'] = results['day_count'].fillna("100").astype(int)
        results['all_time'] = results['feature'].str.extract(r'_(all)_')

        # Mark all rows as NOT_RELEVANT when feature_value is 9999.0 or nan.
        condition1 = ((results.feature_value == 9999.0) | (results.feature_value.isna()))
        results.loc[condition1, 'mapping_status'] = 'NOT_RELEVANT'

        # Mark all rows as NOT_RELEVANT when feature_value is 0  for all features except '_days_since_last_event' feature.
        condition2 = ((results.feature_value == 0.0) & ~(results.feature.str.contains('_days_since_last_event')))
        results.loc[condition2, 'mapping_status'] = 'NOT_RELEVANT'
        log_message(
            message_type='info', 
            message='removing non relevant shaps.',
            final_x_shape = final_x.shape
        )
        not_relevant_results = results[results['mapping_status'] == 'NOT_RELEVANT'].copy()
        # Make attribution_score as sum_attribution_score for all the not_relevant_results
        not_relevant_results['sum_attribution_score'] = not_relevant_results['attribution_score']
        results = results[results['mapping_status'] != 'NOT_RELEVANT']
        # Calculate Max attribution_score as sum_attribution_score for all groups of mapped_feature
        _df = results.groupby(['masterpatientid', 'facilityid', 'mapped_feature']
                              )['attribution_score'].max().reset_index()
        _df = _df.rename(columns={'attribution_score': 'sum_attribution_score'})
        results = results.merge(_df, how='left', on=['masterpatientid', 'facilityid', 'mapped_feature'])
        """ Remove duplicate feature columns.
        ie. cumsum columns have 7, 14, 30 & ALL day variants.
        Use the most recent variant ie. sort by day_count and pick the first row
        """
        results.sort_values(by=['day_count'], inplace=True, ascending=True)
        results = results.drop_duplicates(
            subset=['masterpatientid', 'facilityid', 'mapped_feature'],
            keep='first'
        )
        results['censusdate'] = pd.to_datetime(results.censusdate)

        # =================================================================================================
        dp = DataProcessor(raw_data_dict)
        raw_data = dp.fetch_processed_data()

        dm = DataMapper(attributions=results, client=client, raw_data=raw_data, model=model, report_version=ml_model.output_pdf_report_version)
        final = dm.fetch_mapped_data()

        # not_relevant_results are not dropped becuase we may use those features for debugging
        final = pd.concat([final, not_relevant_results, progress_notes_shap_results])

        final["client"] = client
        final["modelid"] = model.model_name
        final["ml_model_org_config_id"] = ml_model.id
        final["attribution_rank"] = final.groupby(
            ["masterpatientid"]).sum_attribution_score.rank(
            ascending=False
        )
        # ==============================================================================
        final = final.reindex(
            columns=[
                "censusdate",
                "masterpatientid",
                "facilityid",
                "client",
                "modelid",
                "ml_model_org_config_id",
                "feature",
                "feature_value",
                "feature_type",
                "human_readable_name",
                "attribution_score",
                "attribution_rank",
                "sum_attribution_score",
                "calculate_shap",
                "mapping_status"
            ]
        )

    final['feature_value'] = final['feature_value'].astype(str)
    final['censusdate'] = pd.to_datetime(final.censusdate)

    try:
        existing_explanation_output = pd.read_parquet(s3_location_path_prefix + '/explanations_output.parquet')

        existing_explanation_output = existing_explanation_output[
            ~(
                (existing_explanation_output.censusdate.isin(final.censusdate)) &
                (existing_explanation_output.facilityid.isin(final.facilityid)) &
                (existing_explanation_output.client.isin(final.client)) &
                (existing_explanation_output.modelid.isin(final.modelid)) &
                (existing_explanation_output.ml_model_org_config_id.isin(final.ml_model_org_config_id))
            )
        ]

        final = pd.concat([final, existing_explanation_output])

    except FileNotFoundError:
        pass

    s3_path = s3_location_path_prefix + '/explanations_output.parquet'

    log_message(
        message_type='info', 
        message='saving all shap values in s3.',
        result_df_shape=final.shape,
        s3_path=s3_path
    )

    final.to_parquet(s3_path, index=False)
    
    if save_outputs_in_local and (local_folder is not None):
        local_path = local_folder + f'/explanations_output.parquet'
        final.to_parquet(local_path, index=False)
