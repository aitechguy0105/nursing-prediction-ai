import sys

import pandas as pd
from eliot import log_message, start_action
from datetime import timedelta
import time
from .featurizer import BaseFeaturizer


class AdtFeatures(BaseFeaturizer):
    """
    ADG: Admissions, Discharges, Transfers
    """
    def __init__(self, census_df, adt_df, config, training=False):
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate']]
        self.adt_df = adt_df
        self.config = config
        self.training = training
        super(AdtFeatures, self).__init__()
    
    # Return new admissions_df based on days_since_last_admission_v2 definition
    def get_admissions_df_v2(self):
        # Redifine admission_df so we can use it for days since last admission v2
        admission_df = self.adt_df[(self.adt_df.actiontype=='Admission')|(self.adt_df.actiontype=='Return from Leave')]\
                    [['masterpatientid', 'begineffectivedate']]
        admission_df = admission_df.sort_values(['masterpatientid', 'begineffectivedate']).reset_index(drop=True)
        admission_df['original_begineffectivedate'] = admission_df['begineffectivedate']
        admission_df['begineffectivedate'] = admission_df['begineffectivedate'].dt.normalize()

        # create dataframe for exits from facility (Discharge or Leave)
        exit_df = self.adt_df[(self.adt_df.actiontype=='Discharge')|(self.adt_df.actiontype=='Leave')]
        exit_df['original_begineffectivedate'] = exit_df['begineffectivedate']
        exit_df['begineffectivedate'] = exit_df['begineffectivedate'].dt.normalize()

        # In days since last admission v2 we must ignore admissions that follow a leave of absence which happened within the same day as the admission date
        # So, drop all admission_df rows in which there is at least one exit_df row that has:
        # 1. the same masterpatientid
        # 2. the non-normalized begineffectivedate of the admission is greater than the non-normalized begineffectivedate of the exit record
        # 3. the normalized begineffective of the admission is the same as the normalized begineffectivedate of the exit record
        admission_df = admission_df.merge(exit_df, on=['masterpatientid', 'begineffectivedate'], how='left', suffixes=('', '_exit'))
        admission_df = admission_df[admission_df.original_begineffectivedate_exit.isnull() | ~(admission_df.original_begineffectivedate > admission_df.original_begineffectivedate_exit)]

        # drop all columns ending in _exit and the original_begineffectivedate
        drop_cols = ['original_begineffectivedate'] + [col for col in admission_df.columns if col.endswith('_exit')] 
        admission_df.drop(columns=drop_cols, inplace=True)

        del exit_df
        return admission_df.drop_duplicates(subset=['masterpatientid', 'begineffectivedate']).reset_index(drop=True)
        
    def get_days_since_admission(self):
        admission_df = self.adt_df[(self.adt_df.actiontype=='Admission')|(self.adt_df.actiontype=='Return from Leave')]\
                    [['masterpatientid', 'begineffectivedate']]
        admission_df = admission_df.sort_values(['masterpatientid', 'begineffectivedate']).reset_index(drop=True)
        admission_df['begineffectivedate'] = admission_df['begineffectivedate'].dt.normalize()
        admission_df = admission_df.drop_duplicates(subset=['masterpatientid', 'begineffectivedate']).reset_index(drop=True)

        # days since first admission
        log_message(message_type='info', message='ADT - days since first admission features.')
        first_admission = admission_df.groupby(['masterpatientid']).first().reset_index()
        days_since_first = self.census_df.merge(
                        first_admission[['masterpatientid', 'begineffectivedate']], 
                        left_on=['masterpatientid','censusdate'],
                        right_on = ['masterpatientid','begineffectivedate'], 
                        how = 'outer')
        days_since_first.loc[days_since_first.censusdate.isnull(), 'censusdate'] = days_since_first.\
                                                        loc[days_since_first.censusdate.isnull(), 'begineffectivedate']
        days_since_first = days_since_first.sort_values(['masterpatientid','censusdate']).reset_index(drop=True)
        days_since_first['first_admission_date']=days_since_first.groupby('masterpatientid')['begineffectivedate'].ffill()
        days_since_first['days_since_first_admission']=(days_since_first.censusdate-days_since_first.\
                                                        first_admission_date).dt.days
        # days since last admission
        log_message(message_type='info', message='ADT - days since last admission features.')
        if self.config.featurization.adt.days_since_last_admission_v2:
            admission_df = self.get_admissions_df_v2()

        days_since_last= self.census_df.merge(
                            admission_df, 
                            left_on=['masterpatientid','censusdate'],
                            right_on = ['masterpatientid','begineffectivedate'], 
                            how = 'outer')
        days_since_last.loc[days_since_last.censusdate.isnull(), 'censusdate'] = days_since_last.\
                                                        loc[days_since_last.censusdate.isnull(), 'begineffectivedate']
        days_since_last = days_since_last.sort_values(['masterpatientid','censusdate']).reset_index(drop=True)
        days_since_last['last_admission_date']=days_since_last.groupby('masterpatientid')['begineffectivedate'].ffill()
        days_since_last['days_since_last_admission']=(days_since_last.censusdate-days_since_last.\
                                                      last_admission_date).dt.days
        days_since_admission=self.census_df.merge(
                        days_since_first[['masterpatientid', 'censusdate', 'days_since_first_admission']],
                        on=['masterpatientid', 'censusdate'],
                        how='left')
        days_since_admission=days_since_admission.merge(
                        days_since_last[['masterpatientid', 'censusdate', 'days_since_last_admission']],
                        on=['masterpatientid', 'censusdate'],
                        how='left')
        assert days_since_admission.shape[0] == self.census_df.shape[0]
        return days_since_admission
    
    def get_arrival_type_and_days(self):
        log_message(message_type='info', message='ADT - arrival type & days since last arrival features.')
        # Initial Admission
        
        init_admission = self.adt_df.loc[self.adt_df.actiontype=='Admission', ['masterpatientid', 'begineffectivedate']]\
                        .groupby('masterpatientid').first().reset_index()                    
        init_admission.rename(columns={'begineffectivedate': 'effectivedate'}, inplace=True)
        init_admission['arrival_type'] = 'Initial Admission'

        # Readmission
        readmission = self.adt_df.loc[self.adt_df.actiontype=='Admission', ['masterpatientid', 'begineffectivedate']]\
                        .groupby('masterpatientid', as_index=False).apply(lambda x: x.iloc[1:,:]).reset_index(drop=True)
        readmission.sort_values(['masterpatientid', 'begineffectivedate'], inplace=True)
        readmission.rename(columns={'begineffectivedate': 'effectivedate'}, inplace=True)
        readmission.drop_duplicates(['masterpatientid', 'effectivedate'], keep='last', inplace=True)
        readmission['arrival_type'] = 'Readmission'

        # return from leave
        return_from_leave = self.adt_df.loc[self.adt_df.actiontype=='Return from Leave', ['masterpatientid', 'begineffectivedate']].copy()
        return_from_leave.sort_values(['masterpatientid', 'begineffectivedate'], inplace=True)
        return_from_leave.rename(columns={'begineffectivedate': 'effectivedate'}, inplace=True)
        return_from_leave.drop_duplicates(['masterpatientid', 'effectivedate'], keep='last', inplace=True)
        return_from_leave['arrival_type'] = 'Return from Leave'
        #combine all arrival type
        actiontype_df = pd.concat([init_admission, readmission, return_from_leave])
        # ED visit
        # In contrast to other arrival types ED Visit is defined by endeffectivedate
        # since we are interesting in a moment when the patient returns back from the ED visit
        if 'outcome' in self.adt_df.columns:
            ed_visit = self.adt_df.loc[self.adt_df.outcome=='ED Visit Only', ['masterpatientid', 'endeffectivedate']].copy()
            ed_visit.sort_values(['masterpatientid', 'endeffectivedate'], inplace=True)
            ed_visit.rename(columns={'endeffectivedate': 'effectivedate'}, inplace=True)
            ed_visit.drop_duplicates(['masterpatientid', 'effectivedate'], keep='last', inplace=True)
            ed_visit['arrival_type'] = 'ED Visit'
            actiontype_df = pd.concat([actiontype_df, ed_visit])
            del ed_visit
            
        # ED visit can have the same `effectivedate` as return from leave, in this case ED visit should be chosen.
        actiontype_df.sort_values(['masterpatientid', 'effectivedate', 'arrival_type'], inplace=True)
        actiontype_df.drop_duplicates(subset=['masterpatientid', 'effectivedate'], keep='first', inplace=True)
        actiontype_df['effectivedate'] = actiontype_df['effectivedate'].dt.normalize()
        actiontype_df.drop_duplicates(subset=['masterpatientid', 'effectivedate'], keep='last', inplace=True)
        actiontype_df.reset_index(drop=True, inplace=True)

        actiontype_df = self.census_df.merge(actiontype_df[['masterpatientid', 'arrival_type', 'effectivedate']], 
                                  left_on=['masterpatientid', 'censusdate'],
                                  right_on=['masterpatientid', 'effectivedate'],
                                  how='outer')
        actiontype_df['censusdate'].fillna(actiontype_df['effectivedate'], inplace=True) 
        actiontype_df = actiontype_df.sort_values(['masterpatientid', 'censusdate']).reset_index(drop=True)
        actiontype_df['arrival_type'] = actiontype_df.groupby('masterpatientid')['arrival_type'].ffill()
        actiontype_df['effectivedate'] = actiontype_df.groupby('masterpatientid')['effectivedate'].ffill()

        #Adding days since last arrival
        actiontype_df['days_since_last_arrival'] = (actiontype_df.censusdate-actiontype_df.effectivedate).dt.days

        #Remove the unnecessary rows by left joining
        actiontype_df = self.census_df.merge(actiontype_df, 
                                on = ['censusdate', 'facilityid', 'masterpatientid'],
                                how='left')
        actiontype_df.drop(columns='effectivedate', inplace=True)
        del init_admission, readmission, return_from_leave
        assert actiontype_df.shape[0] == self.census_df.shape[0]
        return actiontype_df
    
    def get_absent_type_and_days(self):
        log_message(message_type='info', message='ADT - arrive from type & number of days in there features.')
        arrival_cols = ['masterpatientid', 'begineffectivedate', 'endeffectivedate']
        if 'admitdischargelocationtype' in self.adt_df.columns:
            arrival_cols.append('admitdischargelocationtype')
        arrival_from_df = self.adt_df.loc[:,arrival_cols]
        
        arrival_from_df = arrival_from_df.sort_values(['masterpatientid', 'begineffectivedate', 'endeffectivedate'])
        arrival_from_df['begineffectivedate'] = arrival_from_df['begineffectivedate'].dt.normalize()
        arrival_from_df = arrival_from_df\
                        .drop_duplicates(('masterpatientid', 'begineffectivedate'), keep='last')
        
        if 'admitdischargelocationtype' in self.adt_df.columns:
            arrival_from_df.loc[arrival_from_df.admitdischargelocationtype.isnull(), 'admitdischargelocationtype'] = 'Unknown'
            arrival_from_df = arrival_from_df.rename(columns={'admitdischargelocationtype':'arrival_from'})

        arrival_from_df = arrival_from_df.drop(columns='endeffectivedate')
        arrival_from_feature = self.census_df.merge(arrival_from_df, 
                                  left_on=['masterpatientid', 'censusdate'],
                                  right_on=['masterpatientid', 'begineffectivedate'],
                                  how = 'outer')
        arrival_from_feature.loc[arrival_from_feature.censusdate.isnull(),'censusdate'] = \
                                    arrival_from_feature.loc[arrival_from_feature.censusdate.isnull(),'begineffectivedate']
        arrival_from_feature = arrival_from_feature.sort_values(['masterpatientid', 'censusdate']).reset_index(drop=True)

        arrival_from_feature[['arrival_from']] = arrival_from_feature.groupby('masterpatientid')\
                                                                    [['arrival_from']].ffill()
        #Remove the unnecessary rows by left joining
        arrival_from_feature = self.census_df.merge(arrival_from_feature, 
                                on = ['censusdate', 'facilityid', 'masterpatientid'],
                                how='left')
        
        arrival_from_feature = arrival_from_feature.drop(columns='begineffectivedate') 
        
        assert arrival_from_feature.shape[0]==self.census_df.shape[0]
        return arrival_from_feature
    
    def get_transfer_reason(self):
        log_message(message_type='info', message='ADT - transfer reason features.')
        transferreason = self.adt_df.loc[self.adt_df.transferreason.notnull(), ['masterpatientid', 'endeffectivedate', 'transferreason']].copy()
        transferreason = transferreason.sort_values(['masterpatientid', 'endeffectivedate'])\
                                        .reset_index(drop=True)
        transferreason['endeffectivedate'] = transferreason['endeffectivedate'].dt.normalize()                                
        transferreason = transferreason.groupby(['masterpatientid', 'endeffectivedate']).last().reset_index()
        transferreason_feature = self.census_df.merge(
                                    transferreason,
                                    left_on = ['masterpatientid', 'censusdate'],
                                    right_on=['masterpatientid', 'endeffectivedate'],
                                    how='outer')
        del transferreason
        transferreason_feature.loc[transferreason_feature.censusdate.isnull(), 'censusdate'] = \
                              transferreason_feature\
                              .loc[transferreason_feature.censusdate.isnull(), 'endeffectivedate']  
        transferreason_feature = transferreason_feature.sort_values(['masterpatientid', 'censusdate'])\
                                .reset_index(drop=True)
        transferreason_feature['transferreason'] = transferreason_feature.groupby('masterpatientid')\
                                                    ['transferreason'].ffill()
        transferreason_feature  = transferreason_feature.drop(columns=['endeffectivedate'])
        transferreason_feature = self.census_df.merge(
                            transferreason_feature,
                            on = ['masterpatientid', 'facilityid', 'censusdate'],
                            how='left')
        assert transferreason_feature.shape[0]==self.census_df.shape[0]
        return transferreason_feature

    def generate_features(self):
        """
        This feature type including 7 features:
        1. number of days since first admission
        2. number of days since last admission
        3. the arrival type of the patients, initial-admission, re-admission, .
        4. number of days since last arrival
        5. where the patients arrive from, home, hospital,.
        6. why they were they before arrival 
        7. the reason of transfer
        """

        # preparing census & ADT data
        with start_action(action_type=f"ADT - generating adt features"):
            log_message(
                message_type='info', 
                message='ADT - preparing data for Admissions, Discharges, Transfers features.'
            )
            start = time.time()
            self.census_df.loc[:,'censusdate'] = pd.to_datetime(self.census_df['censusdate'])
            self.census_df = self.census_df.sort_values(['masterpatientid', 'censusdate']).reset_index(drop=True)

            # preparing ADT data 
            self.adt_df.sort_values(['masterpatientid', 'begineffectivedate', 'endeffectivedate'], inplace=True)
            self.adt_df.reset_index(drop=True, inplace=True)

            # get days since admission features
            days_since_admission = self.get_days_since_admission()
            final = self.census_df.merge(days_since_admission,
                               on = ['censusdate', 'facilityid', 'masterpatientid'])
            log_message(
                message_type='info', 
                message=f'ADT - final data after merging days_since_admission features.', 
                final_shape = final.shape, 
                days_since_admission_shape = days_since_admission.shape
            )
            self.sanity_check(final)
            del days_since_admission

            # arrival type & days since last arrival
            arrival_type_df = self.get_arrival_type_and_days()
            final = final.merge(
                            arrival_type_df, 
                            on = ['masterpatientid', 'facilityid', 'censusdate'])
            self.sanity_check(final)
            log_message(
                message_type='info', 
                message=f'ADT - final data after merging arrival_type_df features.',
                final_shape = final.shape, 
                arrival_type_df_shape = arrival_type_df.shape
            )
            del arrival_type_df
            
            # arrive from type & number of days in there
            if 'arrival_from' in self.adt_df.columns and 'admitdischargelocationtype' in self.adt_df.columns:
                arrival_from_df = self.get_absent_type_and_days()
                final = final.merge(arrival_from_df,
                                   on = ['censusdate', 'facilityid', 'masterpatientid'])
                self.sanity_check(final)
                log_message(
                    message_type='info', 
                    message=f'ADT - final data after merging arrival_from_df features.', 
                    final_shape = final.shape,
                    arrival_from_df_shape = arrival_from_df.shape
                )
                del arrival_from_df
            # transfer reason
            if 'transferreason' in self.adt_df.columns:
                transferreason_df = self.get_transfer_reason()
                final = final.merge(transferreason_df,
                                   on = ['censusdate', 'facilityid', 'masterpatientid'])
                self.sanity_check(final)
                log_message(
                    message_type='info', 
                    message=f'ADT - final data after merging transferreason_df features.',
                    final_shape = final.shape, 
                    transferreason_df_shape = transferreason_df.shape
                )
                del transferreason_df

            # change all variable features with string to categorical data 
            for col in ['arrival_type','arrival_from', 'transferreason']:
                if col in final.columns:
                    final[col] = final[col].astype('category')

            self.sanity_check(final)
            log_message(
                message_type='info', 
                message=f'ADT - exiting ADT featurization code.', 
                final_shape = final.shape,
                time_taken=round(time.time() - start, 2)
            )
        return final
