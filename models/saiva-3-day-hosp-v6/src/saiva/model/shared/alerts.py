import gc
import sys

from eliot import log_message, start_action

from .utils import clean_feature_names
from .featurizer import BaseFeaturizer
from .constants import STOP_WATCH_STDALERTTYPEID
import time
from pathlib import Path
import pickle
from omegaconf import OmegaConf


class AlertFeatures(BaseFeaturizer):

    def __init__(self, census_df, alerts, config, training=False):
        self.census_df = census_df[['masterpatientid', 'facilityid', 'censusdate']]
        self.alerts = alerts
        self.training = training
        self.processed_path = Path('/data/processed')
        self.config = config
        super(AlertFeatures, self).__init__()
        
    def nonsystem_alerts_with_stop_watch_feature(self, nonsystem_alerts_df, result_df):
        nonsystem_alerts_df.loc[nonsystem_alerts_df.stdalertid==-1,'alertdescription'] = 'free_text_alertdescription'

        nonsystem_alertdesc_pivot = self.pivot_patient_date_event_count(nonsystem_alerts_df, groupby_column='alertdescription', 
                                        date_column='createddate', prefix='alert_NanTrigger')

        result_df = result_df.merge(
            nonsystem_alertdesc_pivot,
            how='outer',
            on=['masterpatientid', 'censusdate']
        )

        nonsystem_alerts_df.loc[nonsystem_alerts_df.stdalerttypeid.isin(STOP_WATCH_STDALERTTYPEID),'stdalerttyp'] = 'alert_NanTrigger_stop_and_watch'

        stop_watch_pivot = self.pivot_patient_date_event_count(nonsystem_alerts_df, groupby_column='stdalerttyp', 
                                date_column='createddate')
        
        result_df = result_df.merge(
            stop_watch_pivot,
            how='outer',
            on=['masterpatientid', 'censusdate']
        )
        return result_df
    
    def nonsystem_top5_alerts(self, nonsystem_alerts_df, result_df):
        
        nonsystem_alerts_df = nonsystem_alerts_df.loc[nonsystem_alerts_df.alertdescription != '-1']
        # Remove non-ascii characters from alert description and coming further feature names
        nonsystem_alerts_df['alertdescription'] = clean_feature_names(nonsystem_alerts_df['alertdescription'])

        alerts_pivoted = self.pivot_patient_date_event_count(nonsystem_alerts_df, groupby_column='alertdescription', 
                        date_column='createddate', prefix='alert_indicator')

        result_df = result_df.merge(
            alerts_pivoted,
            how='outer',
            on=['masterpatientid', 'censusdate']
        )

        #++++++++++++++++++++++ alerts_NanType_event_pivoted shows when NanType alerts happen ++++++++++++++++++
        nan_triggereditemtype = nonsystem_alerts_df.loc[:,
                                            ['masterpatientid', 'createddate', 'stdalertid', 'indicator']]
        
        alerts_NanType_event_pivoted = self.pivot_patient_date_event_count(nan_triggereditemtype, groupby_column='stdalertid', 
                date_column='createddate', prefix='alert_NanType_event')

        result_df = result_df.merge(
            alerts_NanType_event_pivoted,
            how='outer',
            on=['masterpatientid', 'censusdate']
        ) 

        return result_df
    

    def generate_features(self):
        """
            - alertdescription values are made as columns
            - For each type of `triggereditemtype` create column indicating its count for a
              given masterpatientid & createddate
            """
        with start_action(action_type=f"Alerts - generating alerts features", alerts_shape = self.alerts.shape):
            start = time.time()
            self.alerts['createddate'] = self.alerts.createddate.dt.normalize()
            self.alerts['alertdescription'] = self.alerts['alertdescription'].str.lower()
            self.alerts = self.sorter_and_deduper(
                self.alerts,
                sort_keys=['masterpatientid', 'createddate'],
                unique_keys=['masterpatientid', 'createddate', 'alertdescription', 'triggereditemtype', 'stdalertid']
            )
            generate_na_indicators = self.config.featurization.alerts.generate_na_indicators

            # ==================Filter triggereditemtype=T and alertdescription values made as columns=====================
            patient_alerts_system = self.alerts.loc[self.alerts.triggereditemtype.notna()]
            patient_alerts_therapy = patient_alerts_system.loc[patient_alerts_system.triggereditemtype == 'T'].copy()
            if patient_alerts_therapy.shape[0] != 0:
                patient_alerts_therapy['alertdescription'] = patient_alerts_therapy.alertdescription.str.split(':').str[0]

                alerts_therapy_event_pivoted = self.pivot_patient_date_event_count(patient_alerts_therapy, groupby_column='alertdescription', 
                    date_column='createddate', prefix='alert_therapy_event')

                # meger alerts_therapy_event_pivoted to the result_df 
                result_df = self.census_df.merge(
                    alerts_therapy_event_pivoted,
                    how='outer',
                    on=['masterpatientid', 'censusdate']
                )
                log_message(
                    message_type = 'info', 
                    message = f'Alerts - alerts therapy features created.', 
                    patient_alerts_therapy_shape = patient_alerts_therapy.shape, 
                    result_df_shape=result_df.shape
                )
                del patient_alerts_therapy 
                del alerts_therapy_event_pivoted
                
            else:
                result_df = self.census_df.copy()
                
            
            
            # ===================allergy count column is created=====================
            allergy_alerts = patient_alerts_system[patient_alerts_system.triggereditemtype == 'A'].copy()
            if allergy_alerts.shape[0] != 0:
                allergy_alert_counts = allergy_alerts.groupby([
                    'masterpatientid', 'createddate']).alertdescription.count().reset_index().rename(
                    {'alertdescription': 'count_indicator_allergy'},
                    axis=1
                )
                allergy_alert_counts.columns = ['masterpatientid', 'censusdate'] + \
                                                ['alert_' + name for name in allergy_alert_counts.columns[2:]]
                result_df = result_df.merge(
                    allergy_alert_counts,
                    how='outer',
                    on=['masterpatientid', 'censusdate']
                )
                log_message(
                    message_type = 'info', 
                    message = f'Alerts - allergy_alert_counts features created.',
                    allergy_alert_counts=allergy_alert_counts.shape, 
                    result_df_shape= result_df.shape
                )
            #++++++++++++++++++++++ alerts_allergy_event_pivoted shows when allergy alerts happen ++++++++++++++++++ 
                allergy_alerts['alertdescription_type'] = allergy_alerts['alertdescription'].str.split(':').str[0]

                alerts_allergy_event_pivoted = self.pivot_patient_date_event_count(allergy_alerts, groupby_column='alertdescription_type', 
                    date_column='createddate', prefix='alert_allergy_event')

                # meger alerts_allergy_event_pivoted to the result_df 
                result_df = result_df.merge(
                    alerts_allergy_event_pivoted,
                    how='outer',
                    on=['masterpatientid', 'censusdate']
                )
                log_message(
                    message_type = 'info', 
                    message = f'Alerts - alerts_allergy_event_pivoted features created.', 
                    alerts_allergy_event_pivoted_shape = alerts_allergy_event_pivoted.shape, 
                    result_df_shape = result_df.shape
                )
                del alerts_allergy_event_pivoted
                del allergy_alerts

            # =================dispense count column is created===================
            dispense_alerts = patient_alerts_system[patient_alerts_system.triggereditemtype == 'D'].copy()
            if dispense_alerts.shape[0] != 0:
                dispense_alert_counts = dispense_alerts.groupby([
                    'masterpatientid', 'createddate']).alertdescription.count().reset_index().rename(
                    columns={'alertdescription': 'count_indicator_dispense'}
                )
                dispense_alert_counts.columns = ['masterpatientid', 'censusdate']+\
                            ['alert_' + name for name in dispense_alert_counts.columns[2:]]
                result_df = result_df.merge(
                    dispense_alert_counts,
                    how='outer',
                    on=['masterpatientid', 'censusdate']
                )
                log_message(
                    message_type = 'info', 
                    message = f'Alerts - dispense_alert_counts features created.', 
                    dispense_alert_counts_shape = dispense_alert_counts.shape, 
                    result_df_shape = result_df.shape
                )
            # =================order count column is created===================
            order_alerts = patient_alerts_system[patient_alerts_system.triggereditemtype == 'O'].copy()
            if order_alerts.shape[0] != 0:
                order_alert_counts = order_alerts.groupby(
                    ['masterpatientid', 'createddate']).alertdescription.count().reset_index().rename(
                    columns={'alertdescription': 'count_indicator_order'}
                )
                order_alert_counts.columns = ['masterpatientid', 'censusdate'] + \
                            ['alert_' + name for name in order_alert_counts.columns[2:]]
                result_df = result_df.merge(
                    order_alert_counts,
                    how='outer',
                    on=['masterpatientid', 'censusdate']
                )
                log_message(
                    message_type = 'info', 
                    message = f'Alerts - order_alert_counts features created.',
                    order_alert_counts_shape = order_alert_counts.shape, 
                    result_df_shape = result_df.shape
                )
            # =================Nan triggereditemtype ===================
            nonsystem_alerts = self.alerts.loc[self.alerts.triggereditemtype.isna()]
            nonsystem_alerts['indicator'] = 1
            
            if not nonsystem_alerts.empty:
                log_message(
                    message_type = 'info', 
                    message = f'Alerts - creating non system alerts.', 
                    nonsystem_alerts_shape = nonsystem_alerts.shape
                )
                if self.config.featurization.alerts.nonsystem_alerts_with_stop_watch_feature:
                    result_df = self.nonsystem_alerts_with_stop_watch_feature(nonsystem_alerts, result_df)
                else:
                    if self.training:
                        alertdescription_types = (nonsystem_alerts['alertdescription'].value_counts()[:5].index.tolist())
                        conf = OmegaConf.create({'featurization': {'alerts': {'alertdescription_types': alertdescription_types}}})
                        OmegaConf.save(conf, '/src/saiva/conf/training/generated/alerts.yaml')
                    else:
                        alertdescription_types = self.config.featurization.alerts.alertdescription_types
                        if alertdescription_types is None:
                            alertdescription_types = (nonsystem_alerts['alertdescription'].value_counts().index.tolist())
                    if len(alertdescription_types)>0:
                        nonsystem_alerts = (
                            nonsystem_alerts[
                                nonsystem_alerts['alertdescription'].isin(alertdescription_types)
                            ].copy().reset_index()
                        )

                    assert not self.training, "Deprecation Warning: Please use nonsystem_alerts_with_stop_watch_feature"
                    result_df = self.nonsystem_top5_alerts(nonsystem_alerts, result_df)
            assert result_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
            # =============Trigger garbage collection & downcast to free-up memory ==================
            gc.collect()
            result_df = self.downcast_dtype(result_df)
            result_df = result_df.sort_values(['facilityid','masterpatientid','censusdate'])
            # handle NaN by adding na indicators
            if generate_na_indicators:
                log_message(
                    message_type = 'info', 
                    message = f'Alerts - adding na indicators.'
                )
                result_df = self.add_na_indicators(result_df, self.ignore_columns)
            else:
                log_message(
                    message_type = 'info', 
                    message = f'Alerts - not adding na indicators.'
                )
            cols = [col for col in result_df.columns if col.startswith('alert')]      
            ################################Get count days since last event##########################
            # preparing data 
            events_df = result_df[['facilityid','masterpatientid','censusdate']+cols].copy()

            events_df.loc[:,cols] = events_df.loc[:,cols].fillna(0)
            events_df = self.downcast_dtype(events_df)

            # get counts of days since last event for all events
            log_message(
                message_type='info', 
                message=f'Alerts - counts of days since last event features', 
                events_df_shape=events_df.shape
           )
            days_last_event_df = self.apply_n_days_last_event(events_df, cols)

            #########################################################################################   
            # Do cumulative index
            log_message(
                message_type='info', 
                message='Alerts - cumulative summation, patient days with any events cumsum.'
            )        
            cumidx_df = self.get_cumsum_features(cols, result_df, cumidx=True)

            # Do cumulative summation
            log_message(
                message_type='info', 
                message='Alerts - cumulative summation, total number of events cumsum.'
            )        
            cumsum_df = self.get_cumsum_features(cols, events_df, cumidx=False) 

            result_df = self.census_df.merge(cumidx_df, on=['facilityid','masterpatientid','censusdate'])
            result_df = result_df.merge(cumsum_df, on=['facilityid','masterpatientid','censusdate'])
            result_df = result_df.merge(days_last_event_df, on=['masterpatientid','censusdate'])

            del events_df, cumidx_df, cumsum_df
            del days_last_event_df 

            self.sanity_check(result_df)
            del self.census_df
            log_message(
                message_type='info', 
                message=f'Alerts - exiting alerts featurization code.', 
                        result_df_shape = result_df.shape,
                        time_taken=round(time.time() - start, 2)
            )
            return result_df