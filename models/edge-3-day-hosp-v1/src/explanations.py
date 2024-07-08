import shap
import pandas as pd
from eliot import start_action


# Take the shap_values and convert it into a dataframe "results" that has the
# the shap attribution_score for each feature for each of the predicted residents
def convert_shap_values_to_df(shap_values, modelid, final_x, final_idens):
    # get the column names used by the model
    all_colnames = pd.read_csv(f'/data/models/{modelid}/artifacts/input_features.csv')

    shap_results = []
    for idx, row in final_x.iterrows():
        shaps = pd.DataFrame(
            {
                "feature": all_colnames.feature.values,
                "attribution_score": shap_values[1][idx],
                "feature_value": final_x.iloc[idx],
            }
        )

        shaps["masterpatientid"] = final_idens.iloc[idx].masterpatientid
        shaps["facilityid"] = final_idens.iloc[idx].facilityid
        shaps["censusdate"] = final_idens.iloc[idx].censusdate

        shap_results.append(shaps)

    # convert the list to a dataframe
    results = pd.concat(shap_results)
    return results


def add_columns_to_shap_df(shap_df, client, modelid):
    # set attribution rank
    shap_df["attribution_rank"] = shap_df.groupby(['masterpatientid', 'facilityid']).attribution_score.rank(
        ascending=False)
    shap_df["client"] = client
    shap_df["modelid"] = modelid
    shap_df['censusdate'] = pd.to_datetime(shap_df.censusdate)

    type_mapping_dict = {
        r'^rx_.*': 'Medication',
        r'^dx_.*': 'Diagnosis',
        r'^vitals_.*': 'Vital',
        r'^demo_.*': 'Demographic',
        r'^notes_swem_.*': 'Progress Note',
        r'^stays_.*': 'Stays',
    }

    prefix_remover_dict = {
        r'^rx_' : '',
        r'^dx_' : 'Code ',
        r'^vitals_': '',
        r'^demo_': '',
        r'^notes_swem_': '',
        r'^stays_': '',
    }

    shap_df['feature_type'] = shap_df['feature'].replace(type_mapping_dict, regex=True)
    shap_df['feature_suffix'] = shap_df['feature'].replace(prefix_remover_dict, regex=True)
    shap_df['human_readable_name'] = (shap_df['feature_type'] + ' ' + shap_df[
        'feature_suffix'] + '; feature_value: ' + shap_df['feature_value'].astype(str)).sparse.to_dense()
    shap_df['mapping_status'] = 'MAPPED'
    return shap_df


def generate_explanations(
    prediction_date,
    model,
    final_csr,
    final_idens,
    raw_data_dict,
    client,
    facilityid,
    db_engine,
    save_outputs_in_s3=False,
    s3_location_path_prefix=None,
    save_outputs_in_local=False,
    local_folder=None
):
    with start_action(action_type='calling_shap_tree_explainer', facilityid=facilityid):
        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(final_csr)
        final_x = pd.DataFrame.sparse.from_spmatrix(final_csr)

    with start_action(action_type='converting_shap_values_to_df', facilityid=facilityid):
        results = convert_shap_values_to_df(shap_values, model.model_name, final_x, final_idens)
        results = add_columns_to_shap_df(results, client, model.model_name)

    with start_action(action_type='writing_shap_values_to_db', facilityid=facilityid):
        final = results.reindex(
            columns=[
                "censusdate",
                "masterpatientid",
                "facilityid",
                "client",
                "modelid",
                "feature",
                "feature_value",
                "feature_type",
                "human_readable_name",
                "attribution_score",
                "attribution_rank",
                # "sum_attribution_score",
                # "attribution_percent",
                "mapping_status"
            ]
        )

        final = final.loc[results.attribution_rank <= 100]

        db_engine.execute(f"""
            delete from shap_values 
            where censusdate = '{prediction_date}' 
            and facilityid = '{facilityid}' 
            and client = '{client}' 
            and modelid = '{model.model_name}'
        """)

        if save_outputs_in_s3 and (s3_location_path_prefix is not None):
            s3_path = s3_location_path_prefix + f'/explanations_output.parquet'
            final.to_parquet(s3_path, index=False)

        if save_outputs_in_local and (local_folder is not None):
            local_path = local_folder + f'/explanations_output.parquet'
            final.to_parquet(local_path, index=False)

        final.to_sql(
            "shap_values", db_engine, if_exists="append", index=False, method="multi"
        )
