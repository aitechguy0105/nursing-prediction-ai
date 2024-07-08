import re
from datetime import datetime

import pandas as pd
from eliot import log_message
from explanations.config import TZ, VITAL_FIELD_MAP
from explanations.config import exp_dictionary
from explanations.utils import fetch_report_config, get_config_value


def format_float_str(float_val):
    return '{:.2f}'.format(float_val).rstrip('0').rstrip('.')


def get_leading_reason_string(days, match_name, date):
    """
    Eg: "No bowel movement noted for 72 hrs" Alert N days ago (on 04/09/2021)
    """
    if days == 0:
        return f'{match_name} today (on {date:%m/%d/%Y})'
    elif days == 1:
        return f'{match_name} {days} day ago (on {date:%m/%d/%Y})'
    else:
        return f'{match_name} {days} days ago (on {date:%m/%d/%Y})'


def map_alert_type(alert_type):
    if alert_type == 'allergy':
        alert_type_char = 'A'
        alert_type_string = 'Allergy'
    elif alert_type == 'dispense':
        alert_type_char = 'D'
        alert_type_string = 'Dispense'
    elif alert_type == 'order':
        alert_type_char = 'O'
        alert_type_string = 'Order'
    else:
        assert ('unexpected alert_count_indicator')
    return (alert_type_char, alert_type_string)


class DataMapper(object):
    def __init__(self, *, attributions, client, raw_data, model, report_version):
        self.attributions = attributions
        self.client = client
        self.raw_data = raw_data
        self.model = model
        self.current_date = datetime.now(TZ)
        self.report_config = fetch_report_config(report_version=report_version)

    def fetch_mapped_data(self):
        # run only if meds data is present in the client data dict.
        condition = (self.attributions.mapped_feature.str.startswith('cumsum_med_')) & (
                self.attributions.mapping_status == 'NOT_MAPPED')
        if 'patient_meds' in self.raw_data.keys() and len(self.attributions[condition]) > 0:
            log_message(message_type='info', message='Run Med Mapper....')
            med_duration = get_config_value(
                config_dict=exp_dictionary,
                client=self.client,
                keys=['med_duration']
            )[0]

            self.attributions.loc[condition, ['human_readable_name', 'mapping_status']] = (
                self.attributions
                    .loc[condition]
                    .apply(lambda x: self.med_mapper(x,
                                                     self.raw_data['patient_meds'],
                                                     x.masterpatientid,
                                                     x.censusdate,
                                                     med_duration
                                                     ),
                           axis=1)
                    .values
            )
        # ==============================================================================
        #     run only if labs data is present in the cliet data dict.
        condition = (self.attributions.mapped_feature.str.startswith('cumsum_labs_')) & (
                self.attributions.mapping_status == 'NOT_MAPPED')

        if 'patient_lab_results' in self.raw_data.keys() and len(self.attributions[condition]) > 0:
            log_message(message_type='info', message='Run Lab Mapper....')
            lab_config, lab_duration_config = get_config_value(
                exp_dictionary,
                client=self.client,
                keys=['labs', 'lab_duration']
            )

            self.attributions.loc[condition, ['human_readable_name', 'mapping_status']] = (
                self.attributions
                    .loc[condition]
                    .apply(lambda x: self.lab_mapper(x,
                                                     self.raw_data['patient_lab_results'],
                                                     x.masterpatientid,
                                                     x.censusdate,
                                                     lab_config,
                                                     lab_duration_config
                                                     ),
                           axis=1)
                    .values
            )
        # ==============================================================================
        if 'patient_orders' in self.raw_data.keys():
            log_message(message_type='info', message='Run Order Mapper....')

            condition = (self.attributions.mapped_feature.str.startswith('cumsum_order_') & (
                    self.attributions.feature_type == 'Diagnostic Order')) & (
                                self.attributions.mapping_status == 'NOT_MAPPED')

            if len(self.attributions[condition]) > 0:
                diagnostic_order_duration = get_config_value(
                    exp_dictionary,
                    client=self.client,
                    keys=['diagnostic_order_duration']
                )[0]
                self.attributions.loc[condition, ['human_readable_name', 'mapping_status']] = (
                    self.attributions.
                        loc[condition]
                        .apply(
                        lambda x: self.diagnostic_order_mapper(x, self.raw_data['patient_orders'],
                                                               x.masterpatientid,
                                                               x.censusdate,
                                                               diagnostic_order_duration), axis=1)
                        .values
                )

            condition = (self.attributions.mapped_feature.str.startswith('cumsum_order_') & (
                    self.attributions.feature_type == 'Diet Order')) & (
                                self.attributions.mapping_status == 'NOT_MAPPED')

            if len(self.attributions[condition]) > 0:
                diet_config, diet_duration_config = get_config_value(
                    exp_dictionary,
                    client=self.client,
                    keys=['diet_order', 'diet_order_duration']
                )

                self.attributions.loc[condition, ['human_readable_name', 'mapping_status']] = (
                    self.attributions.
                        loc[condition]
                        .apply(
                        lambda x: self.diet_order_mapper(x,
                                                         self.raw_data['patient_orders'],
                                                         x.masterpatientid,
                                                         x.censusdate,
                                                         diet_config,
                                                         diet_duration_config), axis=1)
                        .values
                )
        # ==============================================================================
        condition = (self.attributions.feature_type == 'Alert') & (self.attributions.mapping_status == 'NOT_MAPPED')

        if 'patient_alerts' in self.raw_data.keys() and len(self.attributions[condition]) > 0:
            log_message(message_type='info', message='Run Alerts Mapper....')
            alerts_config, alert_duration_config, alertcount_config = get_config_value(
                exp_dictionary,
                client=self.client,
                keys=['alerts', 'alert_duration', 'alert_count']
            )

            self.attributions.loc[condition, ['human_readable_name', 'mapping_status']] = (
                self.attributions.
                    loc[condition]
                    .apply(
                    lambda x: self.alert_mapper(x,
                                                self.raw_data['patient_alerts'],
                                                x.masterpatientid,
                                                x.censusdate,
                                                alerts_config,
                                                alert_duration_config,
                                                alertcount_config), axis=1)
                    .values
            )

        # ==============================================================================
        log_message(message_type='info', message='Run Diagnosis Mapper....')
        diagnosis_config = get_config_value(
            exp_dictionary,
            client=self.client,
            keys=['diagnosis']
        )[0]
        condition = (self.attributions.mapped_feature.str.startswith('cumsum_dx_')) & (
                self.attributions.mapping_status == 'NOT_MAPPED')

        if len(self.attributions[condition]) > 0:
            self.attributions.loc[condition, ['human_readable_name', 'mapping_status']] = (
                self.attributions
                    .loc[condition]
                    .apply(lambda x: self.dx_mapper(x,
                                                    self.raw_data['patient_diagnosis'],
                                                    x.masterpatientid,
                                                    x.censusdate,
                                                    diagnosis_config
                                                    ),
                           axis=1).values
            )

        # ==============================================================================
        log_message(message_type='info', message='Run Vital Mapper....')
        condition = (self.attributions.feature_type == 'Vital') & (self.attributions.mapping_status == 'NOT_MAPPED')

        if len(self.attributions[condition]) > 0:
            vitals_config = get_config_value(
                self.report_config,
                client=self.client,
                keys=['vitals_threshold']
            )[0]

            self.attributions.loc[condition, ['human_readable_name', 'mapping_status']] = (
                self.attributions
                    .loc[condition]
                    .apply(lambda x: self.vital_mapper(x,
                                                       self.raw_data['patient_vitals'],
                                                       x.masterpatientid,
                                                       x.censusdate,
                                                       vitals_config),
                           axis=1)
                    .values
            )
        # ==============================================================================
        log_message(message_type='info', message='Run Rehsops Mapper....')
        transfer_duration_config = get_config_value(
            exp_dictionary,
            client=self.client,
            keys=['days_since_last_transfer']
        )[0]

        condition = (self.attributions.feature_type == 'patient_rehosps') & (
                self.attributions.mapping_status == 'NOT_MAPPED')
        if len(self.attributions[condition]) > 0:
            self.attributions.loc[condition, ['human_readable_name', 'mapping_status']] = (
                self.attributions.
                    loc[condition]
                    .apply(lambda x: self.rehosp_mapper(x,
                                                        self.raw_data['patient_rehosps'],
                                                        x.masterpatientid,
                                                        x.censusdate,
                                                        transfer_duration_config),
                           axis=1)
                    .values
            )
        # ==============================================================================
        log_message(message_type='info', message='Run Admissions Mapper....')
        admissions_duration_config = get_config_value(
            exp_dictionary,
            client=self.client,
            keys=['days_since_last_admission']
        )[0]
        condition = (self.attributions.feature == 'admissions_days_since_last_admission') & (
                self.attributions.mapping_status == 'NOT_MAPPED')

        if len(self.attributions[condition]) > 0:
            self.attributions.loc[condition, ['human_readable_name', 'mapping_status']] = (
                self.attributions
                    .loc[condition]
                    .apply(lambda x: self.admissions_mapper(x, admissions_duration_config),
                           axis=1)
                    .values
            )
        # ==============================================================================
        return self.attributions

    def filter_date_range(self, day_count, df, df_date, date_to_use):
        """ day_count=100 if greater than 30"""
        if day_count != 100:
            df = df[df_date >= (date_to_use - pd.to_timedelta(day_count, unit='d'))]

        return df.tail(1).copy().reset_index(drop=True)

    def alert_mapper(self, alert_attribution, alert_data, mpid_to_use, date_to_use, alerts_config,
                     alert_duration_config, alertcount_config):
        """
        - Check whether pre-configured alert strings are present in alert_matches
        - we do a `contains` here rather than list lookup for exact match
        """
        human_readable_name = ''
        mapping_status = 'NOT_MAPPED'
        day_count = alert_attribution['day_count']

        if alert_attribution['mapped_feature'].startswith('cumsum_alert_') and '_count_' not in alert_attribution[
            'mapped_feature']:
            alert_name = alert_attribution['mapped_feature'].replace('cumsum_alert_count_indicator_', '').replace(
                'cumsum_alert_indicator_', '').replace('cumsum_alert_', '')
            alert_name = ' '.join(alert_name.split('_'))
            alert_matches = alert_data[(alert_data.masterpatientid == mpid_to_use) &
                                       (alert_data.alertdescription == alert_name)]

            alert_reason = self.filter_date_range(
                day_count,
                alert_matches,
                alert_matches.createddate,
                date_to_use
            )

            if len(alert_reason) > 0:
                days = (date_to_use.to_pydatetime().date() - alert_reason.iloc[0]['createddt']).days
                human_readable_name = get_leading_reason_string(
                    days=days,
                    match_name=f'"{alert_name}" Alert',
                    date=alert_reason.iloc[0]['createddt']
                )

                mapping_status = 'DATA_FOUND'
                # If alert record falls within the duration & alert string
                # contains words from the configured list
                if days <= alert_duration_config:
                    if [alert_string.lower() in alert_name.lower() for alert_string in alerts_config]:
                        mapping_status = 'MAPPED'
            else:
                mapping_status = 'DATA_NOT_FOUND'

        elif alert_attribution['mapped_feature'].startswith('cumsum_alert_count_indicator_'):
            # TODO: Disabled in config
            alert_type = alert_attribution['mapped_feature'].replace('cumsum_alert_count_indicator_', '')
            (alert_type_char, alert_type_string) = map_alert_type(alert_type)
            feature_value = alert_attribution['feature_value']
            alert_matches = alert_data[(alert_data.masterpatientid == mpid_to_use) &
                                       (alert_data.triggereditemtype == alert_type_char)]
            alert_reason = self.filter_date_range(
                day_count,
                alert_matches,
                alert_matches.createddate,
                date_to_use
            )

            if len(alert_reason) > 0:
                days = (date_to_use.to_pydatetime().date() - alert_reason.iloc[0]['createddt']).days
                alertdescription = alert_reason.iloc[0]['alertdescription'].replace('\n', ' ')
                human_readable_name = get_leading_reason_string(
                    days=days,
                    match_name=f'"{alertdescription}" Alert',
                    date=alert_reason.iloc[0]['createddt']
                )

                if alert_type_string.lower() in alertcount_config:
                    mapping_status = 'MAPPED'
                else:
                    mapping_status = 'DATA_FOUND'
            else:
                mapping_status = 'DATA_NOT_FOUND'

        elif alert_attribution['mapped_feature'].startswith('na_indictator'):
            if 'na_indictator_alert_indicator_' in alert_attribution['mapped_feature']:
                alert_type = alert_attribution['mapped_feature'].replace('na_indictator_alert_indicator_', '').replace(
                    'na_indictator_alert_', '')
            else:
                alert_type = alert_attribution['mapped_feature'].replace('na_indictator_alert_count_indicator_', '')

            feature_value = alert_attribution['feature_value']
            if feature_value:
                human_readable_name = f"{alert_type} alerts missing"
            else:
                human_readable_name = f"{alert_type} alerts present"
            mapping_status = 'DATA_FOUND'

        return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

    def med_mapper(self, med_attribution, med_data, mpid_to_use, date_to_use, med_duration):
        # med does exist
        med_grouping_name = med_attribution['mapped_feature'].replace('cumsum_med_', '')

        med_matches = med_data[(med_data.masterpatientid == mpid_to_use) &
                               (med_data.gpisubclassdescription == med_grouping_name)]
        # if not found in subclassdescription, search the gpiclass instead (that is how the feature is defined)
        if (len(med_matches) == 0):
            med_matches = med_data[(med_data.masterpatientid == mpid_to_use) &
                                   (med_data.gpiclass == med_grouping_name)]

        day_count = med_attribution['day_count']
        med_reason = self.filter_date_range(
            day_count,
            med_matches,
            med_matches.orderdate,
            date_to_use
        )

        if (len(med_reason) > 0):
            med_name = med_reason.iloc[0]['only_med_name']
            is_med_insignificant = med_reason.iloc[0]['insignificant_med']
            days = (date_to_use.to_pydatetime().date() - med_reason.iloc[0]['orderdate']).days

            human_readable_name = get_leading_reason_string(
                days=days,
                match_name=f'{med_name} ordered',
                date=med_reason.iloc[0]['orderdate']
            )
            # important medicines have is_med_insignificant marked as False.
            if (not is_med_insignificant) and med_attribution['feature_value'] and days <= med_duration:
                mapping_status = 'MAPPED'
            else:
                mapping_status = 'DATA_FOUND'
        else:
            mapping_status = 'DATA_NOT_FOUND'
            human_readable_name = ''

        return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

    def lab_mapper(self, lab_attribution, lab_data, mpid_to_use, date_to_use, lab_config, lab_duration_config):
        # lab does exist
        profile_and_abnormality = lab_attribution['mapped_feature'].replace('cumsum_labs__', '')
        profile, abnormality = profile_and_abnormality.split('__')
        profile = profile.replace('_', ' ')
        abnormality = abnormality.replace('_', ' ')
        lab_matches = lab_data[(lab_data.masterpatientid == mpid_to_use) &
                               (lab_data.profiledescription == profile) &
                               (lab_data.abnormalitydescription == abnormality)]

        day_count = lab_attribution['day_count']
        abnormality.replace('non numberic', '(Non-Numeric)')

        lab_reason = self.filter_date_range(
            day_count,
            lab_matches,
            lab_matches.resultdate,
            date_to_use
        )

        if (len(lab_reason) > 0):
            days = (date_to_use - lab_reason.iloc[0]['resultdate']).days
            human_readable_name = get_leading_reason_string(
                days=days,
                match_name=f'Lab Result for "{profile}"',
                date=lab_reason.iloc[0]['resultdate']
            )

            human_readable_name += f" was {abnormality}"
            human_readable_name = human_readable_name.replace('normal', 'Normal')
            # If any integer value in the lab result include it as part of human_readable_name
            if bool(re.search(r'\d', lab_reason.iloc[0]['result'])):
                human_readable_name += f": {lab_reason.iloc[0]['result']}"

            if abnormality.lower() in lab_config and days <= lab_duration_config:
                mapping_status = 'MAPPED'
            else:
                mapping_status = 'DATA_FOUND'
        else:
            mapping_status = 'DATA_NOT_FOUND'
            human_readable_name = ''

        return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

    def diet_order_mapper(self, diet_attribution, order_data, mpid_to_use, date_to_use, diet_config,
                          diet_duration_config):
        # diet_order_mapper maps diet orders and diet supplements separately.
        diet_name = diet_attribution['mapped_feature'].replace('cumsum_order_', '')
        if 'dietsupplement' not in diet_name:
            diet_matches = order_data[(order_data.masterpatientid == mpid_to_use) &
                                      ((order_data.diettype == diet_name) | (order_data.diettexture == diet_name))]
        else:
            diet_matches = order_data[(order_data.masterpatientid == mpid_to_use) &
                                      (order_data.ordercategory == 'Dietary - Supplements')]
        day_count = diet_attribution['day_count']
        diet_reason = self.filter_date_range(
            day_count,
            diet_matches,
            diet_matches.orderdate,
            date_to_use
        )

        if (len(diet_reason) > 0):
            days = (date_to_use.to_pydatetime().date() - diet_reason.iloc[0]["orderdate"].date()).days
            if 'dietsupplement' in diet_name:
                human_readable_name = get_leading_reason_string(
                    days=days,
                    match_name=f'Diet supplement for {diet_reason.iloc[0]["dietsupplement"]}',
                    date=diet_reason.iloc[0]["orderdate"].date()
                )
            else:
                human_readable_name = get_leading_reason_string(
                    days=days,
                    match_name=f'Diet order for {diet_name}',
                    date=diet_reason.iloc[0]["orderdate"].date()
                )

            # Check whether the name is present in pre-configured list &
            # orderdate occurs in last diet_duration_config days
            if diet_name.lower() in diet_config and days <= diet_duration_config:
                mapping_status = 'MAPPED'
            else:
                mapping_status = 'DATA_FOUND'
        else:
            mapping_status = 'DATA_NOT_FOUND'
            human_readable_name = ''
        return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

    def diagnostic_order_mapper(self, diag_attribution, order_data, mpid_to_use, date_to_use,
                                diagnostic_order_duration):
        """
        We pick most recent diagnostic data row which occurs in a given date range
        """
        diag_matches = order_data[(order_data.masterpatientid == mpid_to_use) &
                                  (order_data.ordercategory == 'Diagnostic')]

        day_count = diag_attribution['day_count']
        diag_reason = self.filter_date_range(
            day_count,
            diag_matches,
            diag_matches.orderdate,
            date_to_use
        )

        if (len(diag_reason) > 0):
            days = (date_to_use.to_pydatetime().date() - diag_reason.iloc[0]['orderdate'].date()).days
            diagnostic_order_str = diag_reason.iloc[0]['orderdescription'].replace('\n', ' ')
            result_str = f'{int(diag_attribution["feature_value"])} Diagnostic order in last {day_count} days. Last Order: "{diagnostic_order_str}"'

            human_readable_name = get_leading_reason_string(
                days=days,
                match_name=result_str,
                date=diag_reason.iloc[0]['orderdate'].date()
            )

            if days <= diagnostic_order_duration:
                mapping_status = 'MAPPED'
            else:
                mapping_status = 'DATA_FOUND'
        else:
            mapping_status = 'DATA_NOT_FOUND'
            human_readable_name = ''
        return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

    def dx_mapper(self, dx_attribution, diag_data, mpid_to_use, date_to_use, diagnosis_config):
        diag_label = dx_attribution['mapped_feature'].replace('cumsum_dx_', '')

        diag_matches = diag_data[(diag_data.masterpatientid == mpid_to_use) &
                                 (diag_data.ccs_label == diag_label)]
        day_count = dx_attribution['day_count']
        diag_reason = self.filter_date_range(
            day_count,
            diag_matches,
            diag_matches.onsetdate,
            date_to_use
        )

        if (len(diag_reason) > 0):
            # do not show onsetdate for patients whose initialadmissiondate equals onsetdate.
            # patients set diagnosis onsetdate as initialadmissiondate when they're unaware of the real onsetdate.
#             if diag_reason.iloc[0]['onsetdate'] != diag_reason.iloc[0]['initialadmissiondate'].date():
#                 human_readable_name = f"Diagnosis of {diag_reason.iloc[0]['diagnosiscode']} : {diag_reason.iloc[0]['diagnosisdesc']} on {diag_reason.iloc[0]['onsetdate']:%m/%d/%Y}"
#             else:
#                 human_readable_name = f"Diagnosis of {diag_reason.iloc[0]['diagnosiscode']} : {diag_reason.iloc[0]['diagnosisdesc']}"
            human_readable_name = f"Diagnosis of {diag_reason.iloc[0]['diagnosiscode']} : {diag_reason.iloc[0]['diagnosisdesc']} on {diag_reason.iloc[0]['onsetdate']:%m/%d/%Y}"

            # For every configured string, do a string contains on model feature label
            if [dx_string.lower() in diag_label.lower() for dx_string in diagnosis_config]:
                mapping_status = 'MAPPED'
            else:
                mapping_status = 'DATA FOUND'
        else:
            mapping_status = 'DATA_NOT_FOUND'
            human_readable_name = ''
        return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

    def vital_mapper(self, vital_attribution, vital_data, mpid_to_use, date_to_use, vitals_config):
        aggfunc_mapping = {'min': 'Minimum',
                           'max': 'Maximum',
                           'count': 'Count',
                           'mean': 'Average'}

        human_readable_name = ''
        mapping_status = 'NOT_MAPPED'

        # e.g vtl_Pulse_3_day_count
        daily_match = re.match(r'^vtl_(.*)_\d_day_(max|min|mean|count)$', vital_attribution['mapped_feature'])
        na_match = re.match(r'^na_indictator_vtl_(.*)_3_day_(.*)$', vital_attribution['mapped_feature'])

        if daily_match:
            vital_type = daily_match.groups()[0]
            aggfunc = daily_match.groups()[1]

            vital_matches = vital_data[(vital_data.masterpatientid == mpid_to_use) &
                                       (vital_data.vitalsdescription == vital_type) &
                                       (vital_data.value == vital_attribution['feature_value'])]

            day_count = 9
            vital_matches = vital_matches[vital_matches.date >= (date_to_use - pd.to_timedelta(day_count, unit='d'))]

            if (len(vital_matches) > 0):
                # to get feature_value with trailing 0s and . elimiated
                feature_value_str = format_float_str(vital_attribution['feature_value'])
                aggregation = aggfunc_mapping[aggfunc]
                vitals_display_date = vital_matches.iloc[-1]['date'] # Get the most recent vital
                if vital_type == 'Pulse' or vital_type == 'Weight':
                    vital_attribution['feature_value'] = int(vital_attribution['feature_value'])
                human_readable_name = f"{aggregation} {vital_type}: {feature_value_str} on {vitals_display_date:%m/%d/%Y} at {vitals_display_date:%H:%M}"

                if aggregation == 'Maximum' or aggregation == 'Minimum':
                    human_readable_name = f"{vital_type} of {vital_attribution['feature_value']} recorded on {vitals_display_date:%m/%d/%Y} at {vitals_display_date:%H:%M}"

                    vital_config_name = VITAL_FIELD_MAP.get(vital_type)
                    if (vital_config_name) and (
                            (vital_attribution['feature_value'] < vitals_config[vital_config_name]['min']) or
                            (vital_attribution['feature_value'] > vitals_config[vital_config_name]['max'])
                    ):
                        mapping_status = 'MAPPED'
                    else:
                        mapping_status = 'DATA_FOUND'
                elif aggregation == 'Count':
                    human_readable_name = f"{vital_attribution['feature_value']} measurements of {vital_type} taken in last 3 days"
                    mapping_status = 'DATA_FOUND'

                elif aggregation == 'Average':
                    human_readable_name = f"Average of {vital_type} recorded on last 3 days is {vital_attribution['feature_value']}"
                    mapping_status = 'DATA_FOUND'

            else:
                mapping_status = 'DATA_NOT_FOUND'

        elif na_match:
            vital_type = na_match.groups()[0]
            aggfunc = na_match.groups()[1]
            aggregation = aggfunc_mapping[aggfunc]

            feature_value_str = format_float_str(abs(vital_attribution['feature_value']))
            if int(feature_value_str) > 0:
                human_readable_name = f'{vital_type} values missing'
            else:
                human_readable_name = f'{vital_type} values present'
            mapping_status = 'DATA_FOUND'

        return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

    def rehosp_mapper(self, rehosp_attribution, rehosp_data, mpid_to_use, date_to_use, transfer_duration_config):
        human_readable_name = ''
        mapping_status = 'NOT_MAPPED'
        feature_value = int(rehosp_attribution["feature_value"])
        if rehosp_attribution['mapped_feature'].startswith('hosp_days_'):
            rehosp_matches = rehosp_data[(rehosp_data.masterpatientid == mpid_to_use)]
            # now get the most recent rehosp data (since it is already sorted by rehosp_date)
            rehosp_reason = rehosp_matches.tail(1).copy().reset_index(drop=True)
            if (len(rehosp_reason) > 0):
                human_readable_name = f"{feature_value}"
                if feature_value == 1:
                    human_readable_name += " day since last transfer"
                else:
                    human_readable_name += " days since last transfer"
                if feature_value < transfer_duration_config:
                    mapping_status = 'MAPPED'
                else:
                    mapping_status = 'DATA_FOUND'
            else:
                mapping_status = 'DATA_NOT_FOUND'
        elif rehosp_attribution['mapped_feature'].startswith('hosp_count_'):
            rehosp_matches = rehosp_data[(rehosp_data.masterpatientid == mpid_to_use)]
            # now get the most recent order (since it is already sorted by date_of_transfer)
            rehosp_reason = rehosp_matches.tail(1).copy().reset_index(drop=True)

            if (len(rehosp_reason) > 0):
                if feature_value == 1:
                    human_readable_name = f'{feature_value} prior hospitalization'
                else:
                    human_readable_name = f'{feature_value} prior hospitalizations'
                mapping_status = 'MAPPED'
            else:
                mapping_status = 'DATA_NOT_FOUND'
        return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

    def admissions_mapper(self, admissions_attribution, admissions_duration_config):
        human_readable_name = ''
        mapping_status = 'NOT_MAPPED'
        feature_value = int(admissions_attribution["feature_value"])
        human_readable_name = f"{feature_value}"
        if feature_value == 1:
            human_readable_name += " day since last admission"
        else:
            human_readable_name += " days since last admission"

        if feature_value < admissions_duration_config:
            mapping_status = 'MAPPED'
        else:
            mapping_status = 'DATA_FOUND'
        return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])
