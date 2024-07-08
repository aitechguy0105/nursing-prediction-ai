"""
bedid, censusactioncode, payername, payercode are retained in demographics
featurisation file and dropped from rest of the merges.
"""

from .featurizer import BaseFeaturizer
import sys
import time
import pandas as pd
from eliot import log_message, start_action



class PatientCensus(BaseFeaturizer):
    def __init__(self, census_df, train_start_date, test_end_date):
        self.census_df = census_df
        self.train_start_date = train_start_date
        self.prediction_date = test_end_date
        super(PatientCensus, self).__init__()

    def generate_features(self):
        """
        desc: 1. Creates base df having censusdate from train_start_date to test_end_date.
              2. merging base with patient_census w.r.t censusdate.
        :return: dataframe
        """
        with start_action(action_type=f"generating census_df features", census_df_shape = self.census_df.shape):
            start = time.time()
            log_message(message_type='info', message=' Census - Create a Base by merging Census')
            # We are aware that PCC allows one patient to be in multiple facilities at the same midnight.
            # However, since such cases are rare and the adoption of the code was too complicated,
            # we have decided to count only one patient-day but ensure that the facility is not picked randomly.
            # Under this implementation, the facility with the highest number will be selected.
            base = self.sorter_and_deduper(
                self.census_df,
                sort_keys=['masterpatientid', 'censusdate', 'facilityid'],
                unique_keys=['masterpatientid', 'censusdate']
            )
            assert base.shape[0]==self.census_df.shape[0]

            # adding counts of day since last bed change
            log_message(message_type='info', message=f'Census - counting days since last bed change.', base_shape = base.shape)
            base = self.days_since_bed_change(base)

            assert base.shape[0]==self.census_df.shape[0]

            #################### Get count of allergies for each patient #######################
            allergies_df = base[['masterpatientid','allergies']]
            allergies_df = allergies_df.drop_duplicates()
            # for patients who don't have allergy record, mark as 'To Be Determined'
            allergies_df.loc[allergies_df.allergies.isnull(),'allergies'] = 'To Be Determined'
            # create count of allergies 
            allergies_df['count_allergy'] = None
            allergies_df.loc[allergies_df.allergies == 'To Be Determined', 'count_allergy'] = -1
            allergies_df.loc[allergies_df.allergies == 'No Known Allergies', 'count_allergy'] = 0

            cond = (allergies_df.count_allergy!=-1)&(allergies_df.count_allergy!=0)
            allergies_df.loc[cond,'count_allergy'] = allergies_df.loc[cond,'allergies'].str.count(',').astype(int)+1
            allergies_df = allergies_df[['masterpatientid','count_allergy']]\
                            .groupby('masterpatientid').max().reset_index()
            allergies_df['count_allergy'] = allergies_df['count_allergy'].astype('int16')
            log_message(message_type='info', message=f'Census - allergies features created.', allergies_df_shape = allergies_df.shape)
            row_num = base.shape[0]
            base = base.merge(allergies_df, on='masterpatientid', how = 'left')
            assert base.shape[0] == row_num
            assert base.count_allergy.isnull().sum()==0
            assert base.shape[0]==self.census_df.shape[0]

            # 'payertype' to categorial variable
            base['payertype'].fillna('no payer info', inplace=True)
            base.loc[:, 'payertype'] = base['payertype'].astype('category')

            # Drop unwanted columns
            base.drop(
                ['beddescription',
                    'carelevelcode', 'patientid','allergies'],
                axis=1,
                inplace=True
            )
            # Drop all rows having NaN masterpatientid
            base = base[base['masterpatientid'].notna()]
            assert base.shape[0]==self.census_df.shape[0]

            # have to have only one row per masterpatientid, censusdate pair
            assert base.duplicated(
                subset=['masterpatientid', 'censusdate']).any() == False
            self.sanity_check(base)
            log_message(
                message_type='info', 
                message=f'Census - Featurization completed.',
                base_shape = base.shape,
                time_taken=round(time.time() - start, 2)
                
            )
            return base
