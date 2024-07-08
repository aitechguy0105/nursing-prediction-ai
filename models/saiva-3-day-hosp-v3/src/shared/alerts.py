import gc
import sys

from eliot import log_message

sys.path.insert(0, '/src')
from shared.utils import clean_multi_columns
from shared.utils import clean_feature_names
from shared.featurizer import BaseFeaturizer


class AlertFeatures(BaseFeaturizer):
    def __init__(self, census_df, alerts, training=False):
        self.census_df = census_df
        self.alerts = alerts
        self.training = training
        super(AlertFeatures, self).__init__()

    def generate_features(self):
        """
            - alertdescription values are made as columns
            - For each type of `triggereditemtype` create column indicating its count for a
              given masterpatientid & createddate
            """
        log_message(message_type='info', message='Alerts Processing...')
        self.alerts = self.sorter_and_deduper(
            self.alerts,
            sort_keys=['masterpatientid', 'createddate'],
            unique_keys=['masterpatientid', 'createddate', 'alertdescription']
        )
        # ==================Filter triggereditemtype=T and alertdescription values made as columns=====================
        patient_alerts_system = self.alerts.loc[self.alerts.triggereditemtype.notna()]
        patient_alerts_therapy = patient_alerts_system.loc[patient_alerts_system.triggereditemtype == 'T'].copy()
        if patient_alerts_therapy.shape[0] != 0:
            patient_alerts_therapy['createddate'] = patient_alerts_therapy.createddate.dt.normalize()
            patient_alerts_therapy['alertdescription'] = patient_alerts_therapy.alertdescription.str.split(':').str[0]
            patient_alerts_therapy['indicator'] = 1
            patient_alerts_therapy_pivoted = patient_alerts_therapy.loc[:,
                                             ['masterpatientid', 'createddate', 'alertdescription',
                                              'indicator']].pivot_table(
                index=['masterpatientid', 'createddate'],
                columns='alertdescription',
                values='indicator',
                aggfunc=sum).reset_index()
            patient_alerts_therapy_pivoted.columns = 'alert_' + patient_alerts_therapy_pivoted.columns
            self.census_df = self.drop_columns(self.census_df, '_x$|_y$')
            result_df = self.census_df.merge(
                patient_alerts_therapy_pivoted,
                how='left',
                left_on=['masterpatientid', 'censusdate'],
                right_on=['alert_masterpatientid', 'alert_createddate']
            )
        else:
            result_df = self.census_df.copy()
        del self.census_df
        # ===================allergy count column is created=====================
        allergy_alerts = patient_alerts_system[patient_alerts_system.triggereditemtype == 'A'].copy()
        if allergy_alerts.shape[0] != 0:
            allergy_alerts['createddate'] = allergy_alerts.createddate.dt.normalize()
            allergy_alert_counts = allergy_alerts.groupby([
                'masterpatientid', 'createddate']).alertdescription.count().reset_index().rename(
                {'alertdescription': 'count_indicator_allergy'},
                axis=1
            )
            allergy_alert_counts.columns = 'alert_' + allergy_alert_counts.columns
            result_df = self.drop_columns(result_df, '_x$|_y$')
            result_df = result_df.merge(
                allergy_alert_counts,
                how='left',
                left_on=['masterpatientid', 'censusdate'],
                right_on=['alert_masterpatientid', 'alert_createddate']
            )

        # =================dispense count column is created===================
        dispense_alerts = patient_alerts_system[patient_alerts_system.triggereditemtype == 'D'].copy()
        if dispense_alerts.shape[0] != 0:
            dispense_alerts['createddate'] = dispense_alerts.createddate.dt.normalize()
            dispense_alert_counts = dispense_alerts.groupby([
                'masterpatientid', 'createddate']).alertdescription.count().reset_index().rename(
                columns={'alertdescription': 'count_indicator_dispense'}
            )
            dispense_alert_counts.columns = 'alert_' + dispense_alert_counts.columns
            result_df = self.drop_columns(result_df, '_x$|_y$')
            result_df = result_df.merge(
                dispense_alert_counts,
                how='left',
                left_on=['masterpatientid', 'censusdate'],
                right_on=['alert_masterpatientid', 'alert_createddate']
            )

        # =================order count column is created===================
        order_alerts = patient_alerts_system[patient_alerts_system.triggereditemtype == 'O'].copy()
        if order_alerts.shape[0] != 0:
            order_alerts['createddate'] = order_alerts.createddate.dt.normalize()
            order_alert_counts = order_alerts.groupby(
                ['masterpatientid', 'createddate']).alertdescription.count().reset_index().rename(
                columns={'alertdescription': 'count_indicator_order'}
            )
            order_alert_counts.columns = 'alert_' + order_alert_counts.columns
            result_df = self.drop_columns(result_df, '_x$|_y$')
            result_df = result_df.merge(
                order_alert_counts,
                how='left',
                left_on=['masterpatientid', 'censusdate'],
                right_on=['alert_masterpatientid', 'alert_createddate']
            )

        # =================alertdescription values made as columns===================
        nonsystem_alerts = self.alerts.loc[self.alerts.triggereditemtype.isna()]

        if self.training:
            # pick top 5 alertdescriptions having highest frequency
            alertdescription_types = (nonsystem_alerts['alertdescription'].value_counts()[:5].index.tolist())
        else:
            # consider all the alertdescriptions, but we pick only the features that are part of training
            alertdescription_types = (nonsystem_alerts['alertdescription'].value_counts().index.tolist())

        nonsystem_alerts = (
            nonsystem_alerts[
                nonsystem_alerts['alertdescription'].isin(alertdescription_types)
            ].copy().reset_index()
        )

        nonsystem_alerts['createddate'] = nonsystem_alerts.createddate.dt.normalize()
        nonsystem_alerts['indicator'] = 1
        nonsystem_alerts = nonsystem_alerts.loc[nonsystem_alerts.alertdescription != '-1']
        alerts_pivoted = nonsystem_alerts.loc[:,
                         ['masterpatientid', 'createddate', 'alertdescription', 'indicator']].pivot_table(
            index=['masterpatientid', 'createddate'],
            columns=['alertdescription'],
            values=['indicator'],
            aggfunc=sum).reset_index()

        # Flatten Multilevel columns
        alerts_pivoted.columns = clean_multi_columns(alerts_pivoted.columns)
        # Remove non-ascii characters from feature names
        alerts_pivoted.columns = clean_feature_names(alerts_pivoted.columns)

        alerts_pivoted.columns = 'alert_' + alerts_pivoted.columns
        result_df = self.drop_columns(result_df, '_x$|_y$')
        result_df = result_df.merge(
            alerts_pivoted,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['alert_masterpatientid', 'alert_createddate']
        )

        # drop any duplicated columns
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]

        # drop unwanted columns
        result_df = self.drop_columns(
            result_df,
            '_masterpatientid|_facilityid|createddate|_x$|_y$|bedid|censusactioncode|payername|payercode'
        )

        # =============Trigger garbage collection & downcast to free-up memory ==================
        gc.collect()
        result_df = self.downcast_dtype(result_df)

        assert result_df.duplicated(subset=['masterpatientid', 'censusdate']).any() == False

        # handle NaN by adding na indicators
        log_message(message_type='info', message='Add Na Indicators...')
        result_df = self.add_na_indicators(result_df, self.ignore_columns)
        cols = [col for col in result_df.columns if col.startswith('alert')]
        # Do cumulative summation on all alerts columns
        log_message(message_type='info', message='cumulative summation...')
        result_df = self.get_cumsum_features(cols, result_df)

        return result_df
