import gc
import sys
from datetime import datetime

import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')
from shared.featurizer import BaseFeaturizer


class DiagnosisFeatures(BaseFeaturizer):
    def __init__(self, census_df, diagnosis, diagnosis_lookup_ccs_s3_file_path, training=False):
        self.census_df = census_df
        self.diagnoses_df = diagnosis
        self.training = training
        self.diagnosis_lookup_ccs_s3_file_path = diagnosis_lookup_ccs_s3_file_path
        super(DiagnosisFeatures, self).__init__()

    def generate_features(self):
        """
        - CSV file contains diagnosis mappings with ICD-10 code and categories, label etc
        - Merge these categories into self.diagnoses_df df and use all possible CCS labels as new columns
        - All diagnosis names becomes individaul columns
        - Diagnosis name columns are added to parent df
        """
        current_datetime = datetime.now()

        log_message(message_type='info', message='Diagnosis Processing...')

        # remove outliers
        self.diagnoses_df = self.diagnoses_df.query('onsetdate <= @current_datetime')

        self.diagnoses_df = self.sorter_and_deduper(
            self.diagnoses_df,
            sort_keys=['masterpatientid', 'onsetdate', 'diagnosiscode'],
            unique_keys=['masterpatientid', 'onsetdate', 'diagnosiscode']
        )
        log_message(message_type='info', message=f'Using lookup file: {self.diagnosis_lookup_ccs_s3_file_path}')
        lookup_ccs = pd.read_csv(self.diagnosis_lookup_ccs_s3_file_path)
        lookup_ccs.columns = lookup_ccs.columns.str.replace("'", "")
        lookup_ccs = lookup_ccs.apply(lambda x: x.str.replace("'", ""))
        self.diagnoses_df['indicator'] = 1
        self.diagnoses_df['diagnosiscode'] = self.diagnoses_df.diagnosiscode.str.replace('.', '')
        self.diagnoses_df['onsetdate'] = self.diagnoses_df.onsetdate.dt.date

        self.diagnoses_df_merged = self.diagnoses_df.merge(
            lookup_ccs,
            how='left',
            left_on=['diagnosiscode'],
            right_on=['ICD-10-CM CODE']
        )
        self.diagnoses_df_merged['ccs_label'] = self.diagnoses_df_merged['MULTI CCS LVL 1 LABEL'] + ' - ' + \
                                                self.diagnoses_df_merged['MULTI CCS LVL 2 LABEL']

        diagnosis_pivoted = self.diagnoses_df_merged.loc[:,
                            ['masterpatientid', 'onsetdate', 'ccs_label', 'indicator']].pivot_table(
            index=['masterpatientid', 'onsetdate'],
            columns=['ccs_label'],
            values='indicator',
            fill_value=0
        ).reset_index()

        diagnosis_pivoted['onsetdate'] = pd.to_datetime(diagnosis_pivoted.onsetdate)
        # Add dx_ to all column names
        diagnosis_pivoted.columns = 'dx_' + diagnosis_pivoted.columns

        # ===============================Downcast===============================
        diagnosis_pivoted = self.downcast_dtype(diagnosis_pivoted)

        # This merge works only because census_df has all the dates from training_start_date onwards
        final_df = self.census_df.merge(
            diagnosis_pivoted,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['dx_masterpatientid', 'dx_onsetdate']
        )
        # =============Delete & Trigger garbage collection to free-up memory ==================
        del self.census_df
        del diagnosis_pivoted
        gc.collect()

        final_df.drop(
            final_df.columns[final_df.columns.str.contains(
                'date_of_transfer|dx_onsetdate|_masterpatientid|_facilityid|_x$|_y$|bedid|censusactioncode|payername|payercode'
            )].tolist(),
            axis=1,
            inplace=True
        )

        assert final_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False

        # handle NaN by adding na indicators
        log_message(message_type='info', message='Add Na Indicators...')
        final_df = self.add_na_indicators(final_df, self.ignore_columns)

        cols = [col for col in final_df.columns if col.startswith('dx')]

        # Do cumulative summation on all diagnosis columns
        log_message(message_type='info', message='cumulative summation...')
        final_df = self.get_cumsum_features(cols, final_df)

        return final_df, self.diagnoses_df_merged
