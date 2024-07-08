import gc
import sys

import pandas as pd
from eliot import log_message

sys.path.insert(0, '/src')
from shared.featurizer import BaseFeaturizer


class DiagnosisFeatures(BaseFeaturizer):
    def __init__(self, census_df, diagnosis, s3_bucket):
        self.census_df = census_df
        self.diagnoses_df = diagnosis
        self.s3_bucket = s3_bucket
        super(DiagnosisFeatures, self).__init__()

    def generate_features(self):
        """
        - CSV file contains diagnosis mappings with ICD-10 code and categories, label etc
        - Merge these categories into self.diagnoses_df df and use all possible CCS labels as new columns
        - All diagnosis names becomes individaul columns
        - Diagnosis name columns are added to parent df
        """
        log_message(message_type='info', message='Diagnosis Processing...')
        
        self.diagnoses_df = self.sorter_and_deduper(
            self.diagnoses_df,
            sort_keys=['masterpatientid', 'onsetdate', 'diagnosiscode'],
            unique_keys=['masterpatientid', 'onsetdate', 'diagnosiscode']
        )

        lookup_ccs = pd.read_csv(f's3://{self.s3_bucket}/data/lookup/ccs_dx_icd10cm_2019_1.csv')
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

        # ===============================Downcast===============================
        diagnosis_pivoted = self.downcast_dtype(diagnosis_pivoted)
        final_df = self.census_df.merge(
            diagnosis_pivoted,
            how='outer',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['masterpatientid', 'onsetdate']
        )
        # Since its a `outer` JOIN always keep join vairable name the same across both df's ie. masterpatientid 
        # Since its a `outer` JOIN for all NaN censusdate fill it with onsetdate
        final_df['censusdate'] = final_df['censusdate'].fillna(final_df['onsetdate'])
        final_df.sort_values(by=['masterpatientid', 'censusdate'], inplace=True)

        # =============Delete & Trigger garbage collection to free-up memory ==================
        del self.census_df
        del diagnosis_pivoted
        gc.collect()
        
        final_df.drop(
            final_df.columns[final_df.columns.str.contains(
                'dx_onsetdate|_masterpatientid|bedid|_facilityid|_x$|_y$'
            )].tolist()
            , axis=1, inplace=True)
        
        # ============================ CUMSUM =================================
        # diagnosis columns
        dx_cols = list(final_df.drop(['censusdate','facilityid','masterpatientid','onsetdate'], axis=1).columns)
        # Fill all NAN with 0
        filled = final_df.groupby('masterpatientid')[dx_cols].fillna(0)
        filled['masterpatientid'] = final_df.masterpatientid
        cumsum_all_time = filled.groupby('masterpatientid')[dx_cols].cumsum()
        # Calculate the sum of each row
        cumsum_all_time['dx_sum'] = cumsum_all_time.sum(axis=1)
        
        # Drop existing dx_cols
        final_df = final_df.drop(columns=dx_cols)
        
        final_df = pd.concat(
            [final_df, cumsum_all_time], axis=1
        )  # cumsum is indexed the same as original in the same order
        
        assert final_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
        
        return final_df[['masterpatientid', 'facilityid', 'censusdate', 'dx_sum']]
