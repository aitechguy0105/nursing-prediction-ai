import pandas as pd
from omegaconf import OmegaConf
from eliot import log_message, start_action
from .featurizer import BaseFeaturizer

class MDSFeatures(BaseFeaturizer):
    def __init__(self, *, census_df: pd.DataFrame, mds_df: pd.DataFrame, adt_df: pd.DataFrame, config: OmegaConf, training: bool = False) -> None:
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate']]
        self.mds_df = mds_df
        self.adt_df = adt_df
        self.config = config
        self.training = training
        super().__init__()

    @staticmethod
    def get_category(rug: str) -> str:
        """ Extract the RUG category from the RUG code"""
        if pd.isnull(rug):
            return None
        elif rug[0] != 'R':
            return rug[0]
        elif rug[2] in 'ABCDE':
            return 'R'
        elif rug[2] in 'LX':
            return 'RX'
        else:
            raise ValueError(f"Unknown RUG code: {rug}")

    def _get_normalized_mds(self):
        """ Handle the corrected assessments, if the original assessment was modified, the method assign
            deleteddate to the record.
        """
        
        df = self.mds_df.copy()
        
        modifications = df.groupby(['masterpatientid', 'incorrectassessmentid'])['lockeddate'].min()
        modifications.name = 'deleteddate'
        
        df = df.join(
            modifications,
            on=['masterpatientid', 'assessmentid'],
            how='left'
        )
        
        df = df[[
            'masterpatientid', 'assessmentdate', 'assessmenttypekey',
            'adlscore', 'lockeddate', 'medicarecmi', 'medicarerug', 'deleteddate'
        ]]

        return df

    def _add_rug_categories(self):

        log_message(
            message_type='info',
            message=f'MDS - adding RUG categories.',
        )
        self.res_df['mds_rug_category_1st_previous_value'] = self.res_df['mds_medicarerug_1st_previous_value'].apply(self.get_category)
        self.res_df['mds_rug_category_2nd_previous_value'] = self.res_df['mds_medicarerug_2nd_previous_value'].apply(self.get_category)
        log_message(
            message_type='info',
            message=f'MDS - added RUG categories.',
            res_df_shape = self.res_df.shape,
        )
        return

    def _add_depression_info(self):

        log_message(
            message_type='info',
            message=f'MDS - adding depression info.',
        )
        for ord_n in [self.humanify_number(i) for i in range(1,3)]:
            self.res_df[f'mds_depression_{ord_n}_previous_value'] = None
            self.res_df.loc[
                self.res_df[f'mds_medicarerug_{ord_n}_previous_value'].isin(self.config.featurization.mds.depression_info.depression_codes),
                f'mds_depression_{ord_n}_previous_value'
            ] = 1
            self.res_df.loc[
                self.res_df[f'mds_medicarerug_{ord_n}_previous_value'].isin(self.config.featurization.mds.depression_info.non_depression_codes),
                f'mds_depression_{ord_n}_previous_value'
            ] = 0
            # If the rug code is missed, the depression info should be missed too
            # But if the rug code is valid, but doesn't provide info on the depression level, it should be set to -1
            self.res_df[f'mds_depression_{ord_n}_previous_value'].fillna(-1, inplace=True)
            self.res_df.loc[
                self.res_df[f'mds_medicarerug_{ord_n}_previous_value'].isnull(),
                f'mds_depression_{ord_n}_previous_value'
            ] = None
        log_message(
            message_type='info',
            message=f'MDS - added depression info.',
            res_df_shape = self.res_df.shape,
        )
        return

    def _add_nursing_level(self):

        log_message(
            message_type='info',
            message=f'MDS - adding nursing level.',
        )
        for ord_n in [self.humanify_number(i) for i in range(1,3)]:
            self.res_df[f'mds_nursing_level_{ord_n}_previous_value'] = None
            self.res_df.loc[
                self.res_df[f'mds_medicarerug_{ord_n}_previous_value'].isin(self.config.featurization.mds.nursing_level.more_nursing_codes),
                f'mds_nursing_level_{ord_n}_previous_value'
            ] = 1
            self.res_df.loc[
                self.res_df[f'mds_medicarerug_{ord_n}_previous_value'].isin(self.config.featurization.mds.nursing_level.less_nursing_codes),
                f'mds_nursing_level_{ord_n}_previous_value'
            ] = 0
            self.res_df[f'mds_nursing_level_{ord_n}_previous_value'].fillna(-1, inplace=True)
            self.res_df.loc[
                self.res_df[f'mds_medicarerug_{ord_n}_previous_value'].isnull(),
                f'mds_nursing_level_{ord_n}_previous_value'
            ] = None
        log_message(
            message_type='info',
            message=f'MDS - added nursing level.',
            res_df_shape = self.res_df.shape,
        )
        return

    def _add_rehab_level(self):

        log_message(
            message_type='info',
            message=f'MDS - adding rehab level.',
        )
        for ord_n in [self.humanify_number(i) for i in range(1,3)]:
            self.res_df[f'mds_rehab_level_{ord_n}_previous_value'] = (
                self.res_df[f'mds_medicarerug_{ord_n}_previous_value']
                .str[:2]
                .map(self.config.featurization.mds.rehab_level.rehab_mapping)
            )
            mask = (self.res_df[f'mds_rehab_level_{ord_n}_previous_value'].isnull())&(self.res_df[f'mds_medicarerug_{ord_n}_previous_value'].notnull())
            self.res_df.loc[mask, f'mds_rehab_level_{ord_n}_previous_value'] = 0
        log_message(
            message_type='info',
            message=f'MDS - added rehab level.',
            res_df_shape = self.res_df.shape,
        )
        return

    def _drop_needless_features(self):
        
        cols_to_keep = ['masterpatientid', 'censusdate', 'facilityid']
        cols_to_keep += [key for key, value in self.config.featurization.mds.feature_dtypes.items() if value]

        self.res_df = self.res_df[cols_to_keep]
        return

    def generate_features(self):

        with start_action(action_type=f"MDS - generating mds features"):

            log_message(
                message_type='info',
                message=f'MDS - preparing the dataframes.',
            )
            events = self._get_normalized_mds()

            events = events.loc[events['assessmenttypekey'].isin(self.config.featurization.mds.mds_assessment_types_to_use)]

            self.adt_df['createddate'] = self.adt_df.get('createddate', None)
            reset = (
                self.adt_df.loc[self.adt_df['actiontype']=='Admission',['masterpatientid', 'begineffectivedate', 'createddate']]
                .rename(columns = {'begineffectivedate': 'assessmentdate', 'createddate': 'lockeddate'})
                .set_index(['masterpatientid', 'lockeddate', 'assessmentdate'])
                .index
            )

            # the line below should be skipped when we are using conditional census, I need to update code after merging with dev
            # we still need this line if the census if not conditional, otherwise we may have weird census records, when
            # the patient was just admitted, but the assessments results are taken from the previous stay.
            reset.set_levels(reset.levels[2], level=1, inplace=True) # lockeddate = assessmentdate

            log_message(
                message_type='info',
                message=f'MDS - getting the values from the last assessment conditionally.',
            )
            self.res_df = self.conditional_get_last_values(
                df = events,
                prefix = 'mds',
                event_date_column = 'assessmentdate',
                event_reported_date_column = 'lockeddate',
                event_deleted_date_column = 'deleteddate',
                value_columns = ['medicarerug', 'medicarecmi', 'adlscore'],
                groupby_column = None,
                missing_event_dates = 'drop',
                n = 2,
                reset = reset
            )
            log_message(
                message_type='info',
                message=f'MDS - created the base dataframe with the last RUG, CMI and ADL values.',
                res_df_shape = self.res_df.shape,
            )

            self._add_rug_categories()
            self._add_depression_info()
            self._add_nursing_level()
            self._add_rehab_level()

            self.res_df['mds_adlscore_diff'] = self.res_df['mds_adlscore_1st_previous_value'] - self.res_df['mds_adlscore_2nd_previous_value']
            self.res_df['mds_medicarecmi_diff'] = self.res_df['mds_medicarecmi_1st_previous_value'] - self.res_df['mds_medicarecmi_2nd_previous_value']

            self._drop_needless_features()
            self.res_df = self.res_df.astype({key: value for key, value in self.config.featurization.mds.feature_dtypes.items() if value})

            return self.res_df