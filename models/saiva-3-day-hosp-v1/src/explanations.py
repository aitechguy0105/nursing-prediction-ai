import shap
import pandas as pd
import re
from shared.explanations_config import exp_dictionary

def process_medications(raw_df_dict):
    patient_meds = raw_df_dict['patient_meds']
    patient_meds['only_med_name'] = patient_meds['orderdescription'].str.replace(r' Tablet.*| Liquid.*| Powder.*| Packet.*| Solution.*| Suspension.*','')
    patient_meds = patient_meds.sort_values(by='orderdate', ascending = True)
    return patient_meds

def process_diet_and_diagnostic_orders(raw_df_dict):
    patient_orders = raw_df_dict['patient_orders']
    patient_orders2 = patient_orders.sort_values(by='orderdate', ascending = True)
    return patient_orders

def process_alerts(raw_df_dict):
    patient_alerts = raw_df_dict['patient_alerts']
    patient_alerts['createddt'] = patient_alerts['createddate'].dt.date
    patient_alerts = patient_alerts.sort_values(by='createddate', ascending = True)
    return patient_alerts

def process_diagnosis(raw_df_dict):
    patient_diagnosis = raw_df_dict['patient_diagnosis']
    room_details_df = raw_df_dict['patient_room_details']
    patient_diagnosis = patient_diagnosis.merge(
            room_details_df,
            how='left',
            on=['patientid', 'masterpatientid', 'facilityid']
        )
    patient_diagnosis = patient_diagnosis.sort_values(by='onsetdate', ascending = True)
    return patient_diagnosis

def process_vitals(raw_df_dict):
    patient_vitals = raw_df_dict['patient_vitals']
    patient_vitals['orderdt'] = patient_vitals['date'].dt.date
    patient_vitals = patient_vitals.sort_values(by='date', ascending = True)
    return patient_vitals

def process_rehosps(raw_df_dict):
    patient_rehosps = raw_df_dict['patient_rehosps']
    patient_rehosps = patient_rehosps.sort_values(by='dateoftransfer', ascending = True)
    return patient_rehosps

def process_labs(raw_df_dict):
    patient_labs = raw_df_dict['patient_lab_results']
    patient_labs['resultdt'] = patient_labs['resultdate'].dt.date
    patient_labs = patient_labs.sort_values(by='resultdate', ascending = True)
    return patient_labs

def process_raw_data(raw_df_dict):
    ret_dict = {}

    if len(raw_df_dict['patient_meds']):
        # only if there are meds result rows, do we want to process meds
        ret_dict['patient_meds'] = process_medications(raw_df_dict)
    if len(raw_df_dict['patient_orders']):
        # only if there are order result rows, do we want to process meds
        ret_dict['patient_orders'] = process_diet_and_diagnostic_orders(raw_df_dict)
    if len(raw_df_dict['patient_alerts']):
        # only if there are alert result rows, do we want to process meds
        ret_dict['patient_alerts'] = process_alerts(raw_df_dict)
    ret_dict['patient_diagnosis'] = process_diagnosis(raw_df_dict)
    ret_dict['patient_vitals'] = process_vitals(raw_df_dict)
    ret_dict['patient_rehosps'] = process_rehosps(raw_df_dict)
    if 'patient_lab_results' in raw_df_dict.keys() and len(raw_df_dict['patient_lab_results']):
        # only if there are lab result rows, do we want to process labs
        ret_dict['patient_lab_results'] = process_labs(raw_df_dict)
    
    return ret_dict

def process_attributions(numeric_attributions):
    sorted_numeric_attributions = numeric_attributions.sort_values('attribution_score', ascending=False)
    
    # use this dict to combine attributions.  For example, cumsum_all_med_Heparain and cumsum_7_day_Heparin,
    # become cumsum_med_Heparin and then get deduped into one row and the attribution percentages get added
    # together
    feature_mapping_dict = {
        'cumsum_all_med_' : 'cumsum_med_',
        r'cumsum_\d+_day_med_': 'cumsum_med_',
        'cumsum_all_order_': 'cumsum_order_',
        r'cumsum_\d+_day_order_': 'cumsum_order_',
        'cumsum_all_alert_': 'cumsum_alert_',
        r'cumsum_\d+_day_alert_': 'cumsum_alert_',
        'cumsum_all_dx_': 'cumsum_dx_',
        r'cumsum_\d+_day_dx_': 'cumsum_dx_',
        'cumsum_all_labs_': 'cumsum_labs_',
        r'cumsum_\d+_day_labs_': 'cumsum_labs_',
    }
    
    type_mapping_dict = {
        r'^cumsum_all_med_.*' : 'Medication',
        r'^cumsum_\d+_day_med_.*': 'Medication',
        r'^na_indictator_med_.*': 'Medication',
        r'^cumsum.*diagnostic_orders': 'Diagnostic Order',
        r'^na_indictator_order_.*diagnostic_orders': 'Diagnostic Order',
        r'^cumsum_all_order_.*': 'Diet Order',
        r'^cumsum_\d+_day_order_.*': 'Diet Order',
        r'^na_indictator_order_.*': 'Diet Order',       
        r'^cumsum_all_alert_.*': 'Alert',
        r'^cumsum_\d+_day_alert_.*': 'Alert',
        r'^na_indictator_alert_.*': 'Alert',
        r'^cumsum_all_dx_.*' : 'Diagnosis',
        r'^cumsum_\d+_day_dx_.*': 'Diagnosis',
        r'^na_indictator_dx_.*': 'Diagnosis',
        r'^cumsum_all_labs_.*' : 'Lab',
        r'^cumsum_\d+_day_labs_.*': 'Lab',
        r'^na_indictator_labs_.*': 'Lab',
        r'^vtl_.*': 'Vital',
        r'^rol_avg_\d+_day_vtl_.*': 'Vital',
        r'^rol_std_\d+_day_vtl_.*': 'Vital',
        r'^diff_\d+_day_vtl_.*': 'Vital',
        r'^na_indictator_vtl_.*': 'Vital',
        r'^demo_.*': 'Demographic',
        r'^na_indictator_religion$': 'Demographic',
        r'^na_indictator_education$': 'Demographic',
        r'^na_indictator_race$': 'Demographic',
        r'^na_indictator_citizenship$': 'Demographic',
        r'^na_indictator_state$': 'Demographic',
        r'^dateofbirth_.*': 'Demographic',
        r'^e_pn_.*': 'Progress Note',
        r'^e_eMar_.*': 'eMar',
        r'^hosp_count.*': 'patient_rehosps',
        r'^hosp_days.*': 'patient_rehosps',
        r'^na_indictator_hosp_.*': 'patient_rehosps',
        r'^censusdate_.*': 'patient_census',
        r'^na_indictator_roomratetypedescription$': 'patient_census',
        r'^na_indictator_carelevelcode$': 'patient_census',
        r'^na_indictator_beddescription$': 'patient_census',
    }

    sorted_numeric_attributions['human_readable_name'] = ''
    sorted_numeric_attributions['mapping_status'] = 'NOT_MAPPED'
    sorted_numeric_attributions['mapped_feature'] = sorted_numeric_attributions['feature'].replace(feature_mapping_dict, regex=True)
    sorted_numeric_attributions['feature_type'] = sorted_numeric_attributions['feature'].replace(type_mapping_dict, regex=True)
    sorted_numeric_attributions['day_count'] = sorted_numeric_attributions['feature'].str.extract(r'_(\d+)_day')
    sorted_numeric_attributions['all_time'] = sorted_numeric_attributions['feature'].str.extract(r'_(all)_')
    sum_attributions = sorted_numeric_attributions['attribution_score'].sum()
    sorted_numeric_attributions['attribution_percent'] = sorted_numeric_attributions['attribution_score']/sum_attributions*100.0
    assert(abs(100.0 - sorted_numeric_attributions['attribution_percent'].sum()) <= 0.0001)
    
    attribution_df = sorted_numeric_attributions.groupby(['mapped_feature'])['attribution_score', 'attribution_percent'].sum()
    attribution_df = attribution_df.rename(columns={'attribution_score': 'sum_attribution_score',
                                                    'attribution_percent': 'sum_attribution_percent'})
    sorted_numeric_attributions = sorted_numeric_attributions.merge(attribution_df, how='left', on=['mapped_feature'])
    sorted_numeric_attributions.sort_values(by=['day_count'], inplace=True, ascending=True)
    """
    There are some diagnosis grouping present in 'all' dict but not present in 7/14/30 days dict.
    Therefore not deduping the diagnosis dataframe w.r.t mapped feature.
    """
    sorted_numeric_attributions_only_diagnosis = sorted_numeric_attributions[sorted_numeric_attributions['feature_type']=='Diagnosis']
    sorted_numeric_attributions = sorted_numeric_attributions[sorted_numeric_attributions['feature_type'] != 'Diagnosis']
    deduped_numeric_attributions = sorted_numeric_attributions.drop_duplicates(subset='mapped_feature', keep='first').copy()
    deduped_numeric_attributions = pd.concat([deduped_numeric_attributions,sorted_numeric_attributions_only_diagnosis])
    deduped_numeric_attributions['cumsum_attribution_percent'] = deduped_numeric_attributions['sum_attribution_percent'].cumsum()
    return deduped_numeric_attributions

# returns 0 if day_count was not present
def get_day_count(attribution_row):
    day_count = 0
    if pd.notna(attribution_row['day_count']):
        day_count = int(attribution_row['day_count'])
    return day_count

# return a string like "2 Alerts for "Answer on bowel control" in Last 14 Days"
#                   or "3 Orders for Sympathomimetics in EHR System"
#                   or "1 Dispense Alert in Last 7 Days"
def get_leading_reason_string(event_count_str, day_count, all_time_str, 
                              singular_form, plural_form, match_name='', alert_type_str=''):
    event_count = int(event_count_str)
    
    ret_str = f'{event_count}'
    if alert_type_str != '':
        ret_str += f' {alert_type_str}'
    if (event_count == 1):
        ret_str += f' {singular_form}'
    else:
        ret_str += f' {plural_form}'
    if match_name != '':
        ret_str += f' for "{match_name}"'
    
    # now add 'in EHR System' or 'in n days'
    if all_time_str == 'all':
        ret_str += ' in EHR System.'
    elif day_count != 0:
        ret_str += f' in last {day_count} days.'
    
    return ret_str   

# return a string like " (Last Alert on 02/22/2020)""
#                   or " (Last Order: "Lovenox" on 02/22/2020)""
def get_last_detail_string(noun, last_details_on, last_details=''):
    ret_str = f' Last {noun}'
    if last_details != '':
        ret_str += f': "{last_details}"'
    ret_str += f' on {last_details_on:%m/%d/%Y}'
    return ret_str

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
        assert('unexpected alert_count_indicator')
    return (alert_type_char, alert_type_string)

def alert_mapper(alert_attribution, processed_df_dict, mpid_to_use, date_to_use):
    alert_data = processed_df_dict['patient_alerts']
    human_readable_name = ''
    mapping_status = 'NOT_MAPPED'
    if (alert_attribution['mapped_feature'].startswith('cumsum_alert_indicator_') or
        alert_attribution['mapped_feature'].startswith('cumsum_alert_count_indicator_')) \
            and (alert_attribution['feature_value'] == 0):
        # if feature_value is 0 for cumsum features, it means that no alerts exist for that med feature
        mapping_status = 'NOT_RELEVANT'
    elif alert_attribution['mapped_feature'].startswith('cumsum_alert_indicator_'):
        alert_name = alert_attribution['mapped_feature'].replace('cumsum_alert_indicator_', '')
        # print("alert_name is {}".format(alert_name))
        alert_matches = alert_data[(alert_data.masterpatientid == mpid_to_use) &
                                   (alert_data.alertdescription == alert_name)]
        day_count = get_day_count(alert_attribution)
        if day_count != 0:
            # print("day_count={}".format(day_count))
            alert_matches = alert_matches[alert_matches.createddate >= (date_to_use - pd.to_timedelta(day_count, unit='d'))]
        
        # now get the most recent alert (since it is already sorted by created_date)
        alert_reason = alert_matches.tail(1).copy()
        if len(alert_reason) > 0:
            human_readable_name = get_leading_reason_string(alert_attribution['feature_value'], day_count,
                                                            all_time_str='', singular_form="Alert",
                                                            plural_form="Alerts", match_name= alert_name)
                
            human_readable_name += get_last_detail_string('Alert', alert_reason.iloc[0]['createddt'])
            if day_count in exp_dictionary['Alert_Indicator'].keys() and \
                    alert_name.lower() in exp_dictionary['Alert_Indicator'][day_count]:
                mapping_status = 'MAPPED'
            else:
                mapping_status = 'DATA_FOUND'
            
        else:
            mapping_status = 'DATA_NOT_FOUND'
    elif alert_attribution['mapped_feature'].startswith('cumsum_alert_count_indicator_'):
        alert_type = alert_attribution['mapped_feature'].replace('cumsum_alert_count_indicator_', '')
        (alert_type_char, alert_type_string) = map_alert_type(alert_type)
        feature_value = alert_attribution['feature_value']
        # print(f'found {alert_type_string}:{alert_type_char} with feature value {feature_value}')
        alert_matches = alert_data[(alert_data.masterpatientid == mpid_to_use) &
                                   (alert_data.triggereditemtype == alert_type_char)]
        day_count = get_day_count(alert_attribution)
        if day_count != 0:
            # print("day_count={}".format(day_count))
            alert_matches = alert_matches[alert_matches.createddate >= (date_to_use - pd.to_timedelta(day_count, unit='d'))]
        
        # now get the most recent alert (since it is already sorted by created_date)
        alert_reason = alert_matches.tail(1).copy()
        if len(alert_reason) > 0:
            human_readable_name = get_leading_reason_string(alert_attribution['feature_value'], day_count,
                                                            all_time_str='', singular_form="Alert",
                                                            plural_form="Alerts", alert_type_str=alert_type_string)
            human_readable_name += get_last_detail_string('Alert', alert_reason.iloc[0]['createddt'],
                                                          alert_reason.iloc[0]['alertdescription'].replace('\n',' '))
            if day_count in exp_dictionary['Alert_Count_Indicator'].keys() and \
                    alert_type_string.lower() in exp_dictionary['Alert_Count_Indicator'][day_count]:
                mapping_status = 'MAPPED'
            else:
                mapping_status = 'DATA_FOUND'

        else:
            mapping_status = 'DATA_NOT_FOUND'
        
    return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

def fetch_filter_meds():
    """
    Get the list of meds that are in-Significant and drop them
    """
    df = pd.read_csv(f'/src/static/medication_list.csv')
    d_list = list(df.query("Significance=='No'")['pharmacymedicationname'])
    return '|'.join(d_list)

def med_mapper(med_attribution, med_data, mpid_to_use, date_to_use):
    human_readable_name = ''
    mapping_status = 'NOT_MAPPED'
    if (med_attribution['mapped_feature'].startswith('cumsum_med_')) and (med_attribution['feature_value'] == 0):
        # if feature_value is 0 for cumsum features, it means that no meds exist for that med feature
        mapping_status = 'NOT_RELEVANT'
    elif med_attribution['mapped_feature'].startswith('cumsum_med_'):
        # med does exist
        med_grouping_name = med_attribution['mapped_feature'].replace('cumsum_med_', '')
        # print("med_grouping_name is {}".format(med_grouping_name))
        med_matches = med_data[(med_data.masterpatientid == mpid_to_use) &
                                (med_data.gpisubclassdescription == med_grouping_name)]
        # if not found in subclassdescription, search the gpiclass instead (that is how the feature is defined)
        if (len(med_matches) == 0):
            med_matches = med_data[(med_data.masterpatientid == mpid_to_use) &
                                (med_data.gpiclass == med_grouping_name)]
        day_count = get_day_count(med_attribution)
        if day_count != 0:
            # print("day_count={}".format(day_count))
            med_matches = med_matches[med_matches.orderdate >= (date_to_use - pd.to_timedelta(day_count, unit='d'))]
        
        # now get the most recent med (since it is already sorted by order_date)
        med_reason = med_matches.tail(1).copy().reset_index(drop=True)
        if (len(med_reason) > 0):
            med_name = med_reason.iloc[0]['only_med_name']
            is_med_insignificant = med_reason.iloc[0]['insignificant_med']
            human_readable_name = f"{med_name} ordered on {med_reason.iloc[0]['orderdate']:%m/%d/%Y}" 
            # important medicines have is_med_insignificant marked as False.
            if (not is_med_insignificant) and day_count in exp_dictionary['Patient_Meds'].keys() and \
                    med_grouping_name.lower() in exp_dictionary['Patient_Meds'][day_count] and med_attribution['feature_value']:
                mapping_status = 'MAPPED'
            else:
                mapping_status = 'DATA_FOUND'
        else:
            mapping_status = 'DATA_NOT_FOUND'
    return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

def lab_mapper(lab_attribution, processed_df_dict, mpid_to_use, date_to_use):
    lab_data = processed_df_dict['patient_lab_results']
    human_readable_name = ''
    mapping_status = 'NOT_MAPPED'
    if (lab_attribution['mapped_feature'].startswith('cumsum_labs_')) and (lab_attribution['feature_value'] == 0):
        # if feature_value is 0 for cumsum features, it means that no labs exist for that lab feature
        mapping_status = 'NOT_RELEVANT'
    elif (lab_attribution['mapped_feature'].startswith('cumsum_labs_')):
        # lab does exist
        profile_and_abnormality = lab_attribution['mapped_feature'].replace('cumsum_labs__', '')
        profile, abnormality = profile_and_abnormality.split('__')
        profile = profile.replace('_', ' ')
        abnormality = abnormality.replace('_', ' ')
        # print(f\"profile is {profile} and abnormality is {abnormality}\")
        lab_matches = lab_data[(lab_data.masterpatientid == mpid_to_use) &
                                (lab_data.profiledescription == profile) &
                                (lab_data.abnormalitydescription == abnormality)]
        day_count = get_day_count(lab_attribution)
        abnormality.replace('non numberic', '(Non-Numeric)')

        if day_count != 0:
            # print(\"day_count={}\".format(day_count))
            lab_matches = lab_matches[lab_matches.resultdate >= (date_to_use - pd.to_timedelta(day_count, unit='d'))]
        # now get the most recent lab (since it is already sorted by order_date)
        lab_reason = lab_matches.tail(1).copy().reset_index(drop=True)
        if (len(lab_reason) > 0):           
            human_readable_name = get_leading_reason_string(lab_attribution['feature_value'], day_count, lab_attribution['all_time'],
                                                    "Lab", "Labs", match_name=profile)
            human_readable_name += f" Result was {abnormality}"
            human_readable_name = human_readable_name.replace('normal', 'Normal')
            if day_count != 0 and abnormality.lower() in exp_dictionary['Patient_Labs']['all_labs']:
                if abnormality.lower() in exp_dictionary['Patient_Labs']['labs_with_values']:
                    human_readable_name += f": {lab_reason.iloc[0]['result']}"
                mapping_status = 'MAPPED'
            else:
                mapping_status = 'DATA_FOUND'
        else:
            mapping_status = 'DATA_NOT_FOUND'
    return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

def diet_order_mapper(diet_attribution, processed_df_dict, mpid_to_use, date_to_use):
    order_data = processed_df_dict['patient_orders']
    human_readable_name = ''
    mapping_status = 'NOT_MAPPED'
    if (diet_attribution['mapped_feature'].startswith('cumsum_order_')) and (diet_attribution['feature_value'] == 0):
        mapping_status = 'NOT_RELEVANT'
    elif diet_attribution['mapped_feature'].startswith('cumsum_order_'):
        diet_name = diet_attribution['mapped_feature'].replace('cumsum_order_', '')
        # print("diet_name is {}".format(diet_name))
        diet_matches = order_data[(order_data.masterpatientid == mpid_to_use) & 
                                  ((order_data.diettype == diet_name) | (order_data.diettexture == diet_name) | 
                                   (order_data.dietsupplement == diet_name))]
        day_count = get_day_count(diet_attribution)
        if day_count != 0:
            # print("day_count={}".format(day_count))
            diet_matches = diet_matches[diet_matches.orderdate >= (date_to_use - pd.to_timedelta(day_count, unit='d'))]
        
        # now get the most recent order (since it is already sorted by order_date)
        diet_reason = diet_matches.tail(1).copy().reset_index(drop=True)
        if (len(diet_reason) > 0):
            human_readable_name = f'Diet order for {diet_name} in last {day_count} days (on {diet_reason.iloc[0]["orderdate"].date():%m/%d/%Y})'
            if day_count in exp_dictionary['Diet_Order'].keys() and diet_name.lower() in exp_dictionary['Diet_Order'][day_count]:
                mapping_status = 'MAPPED'
            else:
                mapping_status = 'DATA_FOUND'
        else:
            mapping_status = 'DATA_NOT_FOUND'
    return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

def diagnostic_order_mapper(diag_attribution, processed_df_dict, mpid_to_use, date_to_use):
    order_data = processed_df_dict['patient_orders']
    human_readable_name = ''
    mapping_status = 'NOT_MAPPED'
    if (diag_attribution['mapped_feature'].startswith('cumsum_order_')) and (diag_attribution['feature_value'] == 0):
        mapping_status = 'NOT_RELEVANT'

    elif diag_attribution['mapped_feature'].startswith('cumsum_order_'):
        diag_matches = order_data[(order_data.masterpatientid == mpid_to_use) & 
                                  (order_data.ordercategory == 'Diagnostic')]
        day_count = get_day_count(diag_attribution)
        if day_count != 0:
            # print("day_count={}".format(day_count))
            diag_matches = diag_matches[diag_matches.orderdate >= (date_to_use - pd.to_timedelta(day_count, unit='d'))]
        
        # now get the most recent order (since it is already sorted by order_date)
        diag_reason = diag_matches.tail(1).copy().reset_index(drop=True)
        
        if (len(diag_reason) > 0):
            human_readable_name = get_leading_reason_string(diag_attribution['feature_value'], day_count,
                                                                all_time_str='', singular_form="Diagnostic Order", 
                                                                plural_form="Diagnostic Orders")
            human_readable_name += get_last_detail_string('Order', diag_reason.iloc[0]['orderdate'].date(),
                                                          diag_reason.iloc[0]['orderdescription'].replace('\n',' '))
            if day_count in exp_dictionary['Diagnostic_Order']:
                mapping_status = 'MAPPED' 
            else:
                mapping_status = 'DATA_FOUND'      
        else:
            mapping_status = 'DATA_NOT_FOUND'
    return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

def dx_mapper(dx_attribution, processed_df_dict, mpid_to_use, date_to_use):
    diag_data = processed_df_dict['patient_diagnosis']
    human_readable_name = ''
    mapping_status = 'NOT_MAPPED'
    if (dx_attribution['mapped_feature'].startswith('cumsum_dx_')) and (dx_attribution['feature_value'] == 0):
        mapping_status = 'NOT_RELEVANT'

    elif dx_attribution['mapped_feature'].startswith('cumsum_dx_'):
        diag_label = dx_attribution['mapped_feature'].replace('cumsum_dx_', '')
        
        diag_matches = diag_data[(diag_data.masterpatientid == mpid_to_use) & 
                                  (diag_data.ccs_label == diag_label)]
        day_count = get_day_count(dx_attribution)
        if day_count != 0:
            # print("day_count={}".format(day_count))
            diag_matches = diag_matches[diag_matches.onsetdate >= (date_to_use - pd.to_timedelta(day_count, unit='d'))]        
        # now get the most recent order (since it is already sorted by onset_date)
        diag_reason = diag_matches.tail(1).copy().reset_index(drop=True)
        
        if (len(diag_reason) > 0):
            # do not show onsetdate for patients whose initialadmissiondate equals onsetdate.
            # patients set diagnosis onsetdate as initialadmissiondate when they're unaware of the real onsetdate.
            if diag_reason.iloc[0]['onsetdate'] != diag_reason.iloc[0]['initialadmissiondate'].date():
                human_readable_name = f"Diagnosis of {diag_reason.iloc[0]['diagnosiscode']} : {diag_reason.iloc[0]['diagnosisdesc']} on {diag_reason.iloc[0]['onsetdate']:%m/%d/%Y}"
            else:
                human_readable_name = f"Diagnosis of {diag_reason.iloc[0]['diagnosiscode']} : {diag_reason.iloc[0]['diagnosisdesc']}"
            if day_count in exp_dictionary['Patient_Diagnosis'].keys() and diag_label.lower() in exp_dictionary['Patient_Diagnosis'][
                day_count]:
                mapping_status = 'MAPPED'
            elif (dx_attribution['all_time']=='all') and \
                    (any(diag_all_string in diag_label.lower() for diag_all_string in exp_dictionary['Patient_Diagnosis']['all'])):
                mapping_status = 'MAPPED'
            else:
                mapping_status = 'DATA FOUND'
        else:
            mapping_status = 'DATA_NOT_FOUND'
    return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

def format_float_str(float_val):
    return '{:.2f}'.format(float_val).rstrip('0').rstrip('.')

def vital_mapper(vital_attribution, processed_df_dict, mpid_to_use, date_to_use):
    aggfunc_mapping = {'min': 'Minimum',
                       'max': 'Maximum',
                       'std': 'Variation',
                       'median': 'Average'}
    vital_data = processed_df_dict['patient_vitals']
    human_readable_name = ''
    mapping_status = 'NOT_MAPPED'

    # e.g vtl_max_Pain Level
    daily_match = re.match(r'^vtl_(max|min|median|std)_(.*)$', vital_attribution['mapped_feature'])
    # e.g. diff_7_day_vtl_min_O2 sats
    diff_match = re.match(r'^diff_\d+_day_vtl_(max|min|median|std)_(.*)$', vital_attribution['mapped_feature'])
    # e.g. rol_avg_7_day_vtl_std_Pain Level
    rol_avg_match = re.match(r'^rol_avg_\d+_day_vtl_(max|min|median|std)_(.*)$', vital_attribution['mapped_feature'])
    # e.g rol_std_14_day_vtl_max_Pain Level
    rol_std_match = re.match(r'^rol_std_\d+_day_vtl_(max|min|median|std)_(.*)$', vital_attribution['mapped_feature'])
    if daily_match:
        aggfunc = daily_match.groups()[0]
        vital_type = daily_match.groups()[1]
        # print(f"matched {aggfunc} and {vital_type}")

        vital_matches = vital_data[(vital_data.masterpatientid == mpid_to_use) &
                                   (vital_data.vitalsdescription == vital_type) &
                                   (vital_data.value == vital_attribution['feature_value'])]

        day_count = 1
        vital_matches = vital_matches[vital_matches.date >= (date_to_use - pd.to_timedelta(day_count, unit='d'))]

        if (len(vital_matches) > 0):
            # to get feature_value with trailing 0s and . elimiated
            feature_value_str = format_float_str(vital_attribution['feature_value'])
            date_to_use_dt = date_to_use.date()
            aggregation = aggfunc_mapping[aggfunc]
            vitals_display_date = vital_matches.iloc[0]['date']
            human_readable_name = f"{aggregation} {vital_type}: {feature_value_str} on {vitals_display_date:%m/%d/%Y} at {vitals_display_date:%H:%M}"
            #vitals - pulse
            if aggregation=='Maximum' and vital_type=='Pulse':
                human_readable_name = f"Maximum recorded pulse {vital_attribution['feature_value']} on {vitals_display_date:%m/%d/%Y} at {vitals_display_date:%H:%M}"
                if (vital_attribution['feature_value'] > 109):
                    mapping_status = 'MAPPED'
                else:
                    mapping_status = 'DATA_FOUND'
            # vitals - bmi
            elif vital_type == 'bmi':
                if vital_attribution['feature_value']>34:
                    human_readable_name = f"Obese BMI of {vital_attribution['feature_value']} recorded on {vitals_display_date:%m/%d/%Y}"
                    mapping_status = 'MAPPED'
                elif vital_attribution['feature_value']>14 and vital_attribution['feature_value']<17.6:
                    human_readable_name = f"Low BMI of {vital_attribution['feature_value']} recorded on {vitals_display_date:%m/%d/%Y}"
                    mapping_status = 'MAPPED'
                else:
                    mapping_status = 'DATA_FOUND'
            # vitals - diastolicvalue
            elif aggregation=='Minimum' and vital_type=='diastolicvalue':
                if vital_attribution['feature_value']>25 and vital_attribution['feature_value']<70:
                    human_readable_name = f"Low diastolic of {vital_attribution['feature_value']} recorded on {vitals_display_date:%m/%d/%Y} at {vitals_display_date:%H:%M}"
                    mapping_status = 'MAPPED'
                elif vital_attribution['feature_value']>92 and vital_attribution['feature_value']<200:
                    human_readable_name = f"High diastolic of {vital_attribution['feature_value']} recorded on {vitals_display_date:%m/%d/%Y} at {vitals_display_date:%H:%M}"
                    mapping_status = 'MAPPED'
                else:
                    mapping_status = 'DATA_FOUND'
            # vitals - O2 sats
            elif aggregation=='Minimum' and vital_type=='O2 sats':
                human_readable_name = f"Low O2 saturation of {vital_attribution['feature_value']} recorded on {vitals_display_date:%m/%d/%Y} at {vitals_display_date:%H:%M}"
                if vital_attribution['feature_value']>70 and vital_attribution['feature_value']<90:
                    mapping_status = 'MAPPED'
                else:
                    mapping_status = 'DATA_FOUND'
            # vitals - Temperature
            elif vital_type == 'Temperature':
                if aggregation == 'Minimum' and vital_attribution['feature_value'] > 85 and vital_attribution['feature_value'] < 97.5:
                    human_readable_name = f"Low body temperature of {vital_attribution['feature_value']} recorded on {vitals_display_date:%m/%d/%Y} at {vitals_display_date:%H:%M}"
                    mapping_status = 'MAPPED'
                elif aggregation == 'Maximum' and vital_attribution['feature_value'] > 99.9 and vital_attribution['feature_value'] < 108.6:
                    human_readable_name = f"High body temperature of {vital_attribution['feature_value']} recorded on {vitals_display_date:%m/%d/%Y} at {vitals_display_date:%H:%M}"
                    mapping_status = 'MAPPED'
                else:
                    mapping_status = 'DATA_FOUND'
            # vitals - Blood Sugar
            elif vital_type == 'Blood Sugar':

                if aggregation == 'Minimum' and vital_attribution['feature_value'] > 15 and vital_attribution['feature_value'] < 70:
                    human_readable_name = f"Low blood sugar of {vital_attribution['feature_value']} recorded on {vitals_display_date:%m/%d/%Y} at {vitals_display_date:%H:%M}"
                    mapping_status = 'MAPPED'
                elif aggregation == 'Maximum' and vital_attribution['feature_value'] > 245 and vital_attribution['feature_value'] < 525:
                    human_readable_name = f"High blood sugar of {vital_attribution['feature_value']} recorded on {vitals_display_date:%m/%d/%Y} at {vitals_display_date:%H:%M}"
                    mapping_status = 'MAPPED'
                else:
                    mapping_status = 'DATA_FOUND'
            # vitals - Respiration
            elif aggregation == 'Maximum' and vital_type == 'Respiration':
                human_readable_name = f"High respiration of {vital_attribution['feature_value']} recorded on {vitals_display_date:%m/%d/%Y} at {vitals_display_date:%H:%M}"
                if vital_attribution['feature_value'] > 27 and vital_attribution['feature_value'] < 70:
                    mapping_status = 'MAPPED'
                else:
                    mapping_status = 'DATA_FOUND'
            # vitals - BP-Systolic
            elif aggregation=='Minimum' and vital_type=='BP-Systolic':
                if vital_attribution['feature_value']>85 and vital_attribution['feature_value']<115:
                    human_readable_name = f"Low systolic of {vital_attribution['feature_value']} recorded on {vitals_display_date:%m/%d/%Y} at {vitals_display_date:%H:%M}"
                    mapping_status = 'MAPPED'
                elif vital_attribution['feature_value']>160 and vital_attribution['feature_value']<200:
                    human_readable_name = f"High systolic of {vital_attribution['feature_value']} recorded on {vitals_display_date:%m/%d/%Y} at {vitals_display_date:%H:%M}"
                    mapping_status = 'MAPPED'
                else:
                    mapping_status = 'DATA_FOUND'

        else:
            mapping_status = 'DATA_NOT_FOUND'

    elif diff_match:
        aggfunc = diff_match.groups()[0]
        vital_type = diff_match.groups()[1]
        day_count = get_day_count(vital_attribution)
        if day_count == 1:
            day_word = "day"
        else:
            day_word = "days"
        # print(f"matched diff {aggfunc} and {vital_type}")
        if vital_attribution['feature_value'] < 0:
            direction = 'decreased'
        else:
            direction = 'increased'
        feature_value_str = format_float_str(abs(vital_attribution['feature_value']))
        human_readable_name = f'Daily {aggfunc_mapping[aggfunc]} {vital_type} {direction} by {feature_value_str} over the last {day_count} {day_word}'
        mapping_status = 'DATA_FOUND'
    elif rol_avg_match:
        aggfunc = rol_avg_match.groups()[0]
        vital_type = rol_avg_match.groups()[1]
        day_count = get_day_count(vital_attribution)
        # print(f"matched rol_avg {aggfunc} and {vital_type}")
        feature_value_str = format_float_str(abs(vital_attribution['feature_value']))
        human_readable_name = f'Average of daily {aggfunc_mapping[aggfunc]} {vital_type} over the last {day_count} days: {feature_value_str}'
        mapping_status = 'DATA_FOUND'
    elif rol_std_match:
        aggfunc = rol_std_match.groups()[0]
        vital_type = rol_std_match.groups()[1]
        day_count = get_day_count(vital_attribution)
        # print(f"matched std_avg {aggfunc} and {vital_type}")
        feature_value_str = format_float_str(abs(vital_attribution['feature_value']))
        human_readable_name = f'Variation of daily {aggfunc_mapping[aggfunc]} {vital_type} over the last {day_count} days: {feature_value_str}'
        mapping_status = 'DATA_FOUND'

    return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

def rehosp_mapper(rehosp_attribution, processed_df_dict, mpid_to_use, date_to_use):
    rehosp_data = processed_df_dict['patient_rehosps']
    human_readable_name = ''
    mapping_status = 'NOT_MAPPED'
    feature_value = rehosp_attribution["feature_value"]
    if rehosp_attribution['mapped_feature'].startswith('hosp_days_'):
        rehosp_matches = rehosp_data[(rehosp_data.masterpatientid == mpid_to_use)]
        # now get the most recent order (since it is already sorted by order_date)
        rehosp_reason = rehosp_matches.tail(1).copy().reset_index(drop=True)
        if (len(rehosp_reason) > 0):
            human_readable_name = f'{int(feature_value)} days since last transfer'
            if feature_value <= 30:
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
            if int(feature_value)<=1:
                human_readable_name = f'{int(feature_value)} prior hospitalization'
            else:
                human_readable_name = f'{int(feature_value)} prior hospitalizations'
            mapping_status = 'MAPPED'
        else:
            mapping_status = 'DATA_NOT_FOUND'
    return pd.Series([human_readable_name, mapping_status], index=['a', 'b'])

def generate_explanations(
    prediction_date,
    model,
    final_x,
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
    
    
    
    ret_dict = process_raw_data(raw_data_dict)

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(final_x)

    shap_results = []

    for idx, row in final_x.iterrows():
        shaps = pd.DataFrame(
            {
                "feature": final_x.columns,
                "attribution_score": shap_values[1][idx],
                "feature_value": final_x.iloc[idx],
            }
        )

        shaps["masterpatientid"] = final_idens.iloc[idx].masterpatientid
        shaps["facilityid"] = final_idens.iloc[idx].facilityid
        shaps["censusdate"] = final_idens.iloc[idx].censusdate

        shap_results.append(shaps)


    results = pd.concat(shap_results)
    processed_attributions = results.groupby(['masterpatientid','facilityid']).apply(process_attributions).reset_index(drop=True)
    processed_attributions['censusdate'] = pd.to_datetime(processed_attributions.censusdate)
    # run only if meds data is present in the cliet data dict.
    if 'patient_meds' in ret_dict.keys():
        meds = fetch_filter_meds()
        # adding extra column to distinguish display medications.
        med_data = ret_dict['patient_meds']
        # medicines marked 'no' is given 'True' and vice-versa for insignificant_med columns.
        med_data["insignificant_med"] = med_data["only_med_name"].str.contains(meds, na=False, case=False)
        processed_attributions.loc[processed_attributions.feature_type == 'Medication', ['human_readable_name','mapping_status']] = (
        processed_attributions
        .loc[processed_attributions.feature_type == 'Medication']
        .apply(lambda x: med_mapper(x, med_data, x.masterpatientid, x.censusdate), axis=1)
        .values
        )
#     run only if labs data is present in the cliet data dict.
    if 'patient_lab_results' in ret_dict.keys():
        processed_attributions.loc[processed_attributions.feature_type == 'Lab', ['human_readable_name','mapping_status']] = (
        processed_attributions
        .loc[processed_attributions.feature_type == 'Lab']
        .apply(lambda x: lab_mapper(x, ret_dict, x.masterpatientid, x.censusdate), axis=1)
        .values
        )
    if 'patient_orders' in ret_dict.keys():
        processed_attributions.loc[processed_attributions.feature_type == 'Diagnostic Order', ['human_readable_name', 'mapping_status']] = (
        processed_attributions.
        loc[processed_attributions.feature_type == 'Diagnostic Order']
        .apply(lambda x: diagnostic_order_mapper(x, ret_dict, x.masterpatientid, x.censusdate), axis=1)
        .values
        )

        processed_attributions.loc[processed_attributions.feature_type == 'Diet Order', ['human_readable_name', 'mapping_status']] = (
            processed_attributions.
            loc[processed_attributions.feature_type == 'Diet Order']
            .apply(lambda x: diet_order_mapper(x, ret_dict, x.masterpatientid, x.censusdate), axis=1)
            .values
        )
    if 'patient_alerts' in ret_dict.keys():
        processed_attributions.loc[processed_attributions.feature_type == 'Alert', ['human_readable_name', 'mapping_status']] = (
        processed_attributions.
        loc[processed_attributions.feature_type == 'Alert']
        .apply(lambda x: alert_mapper(x, ret_dict, x.masterpatientid, x.censusdate), axis=1)
        .values
        )
    
    # processed_attributions.loc[processed_attributions.feature_type == 'Diagnosis', ['human_readable_name', 'mapping_status']] = (
    # processed_attributions
    # .loc[processed_attributions.feature_type == 'Diagnosis']
    # .apply(lambda x: dx_mapper(x, ret_dict, x.masterpatientid, x.censusdate), axis=1)
    # .values
    # )
    processed_attributions.loc[processed_attributions.feature_type == 'Diagnosis', ['human_readable_name', 'mapping_status']] = (
        processed_attributions
            .loc[processed_attributions.feature_type == 'Diagnosis']
            .apply(lambda x: dx_mapper(x, ret_dict, x.masterpatientid, x.censusdate), axis=1)
            .values
    )
    # deduping the duplicates diagnosis rows,thus  if 7 and 14 are both mapped then we are going to show only 7.
    processed_attributions.loc[processed_attributions.feature_type == 'Diagnosis', :] = \
        processed_attributions.loc[(processed_attributions.feature_type == 'Diagnosis') & (~processed_attributions.duplicated(
            subset=['masterpatientid', 'mapping_status', 'mapped_feature'],keep='first')),:]
    
    processed_attributions.loc[processed_attributions.feature_type == 'Vital', ['human_readable_name', 'mapping_status']] = (
    processed_attributions
    .loc[processed_attributions.feature_type == 'Vital']
    .apply(lambda x: vital_mapper(x, ret_dict, x.masterpatientid, x.censusdate), axis=1)
    .values
    )
    processed_attributions.loc[
        processed_attributions.feature_type == 'patient_rehosps', ['human_readable_name', 'mapping_status']] = (
        processed_attributions.
            loc[processed_attributions.feature_type == 'patient_rehosps']
            .apply(lambda x: rehosp_mapper(x, ret_dict, x.masterpatientid, x.censusdate), axis=1)
            .values
    )
    
    processed_attributions["client"] = client
    processed_attributions["modelid"] = model.model_name
    processed_attributions["attribution_rank"] = processed_attributions.groupby(["masterpatientid"]).attribution_percent.rank(
        ascending=False
    )
    
    final = processed_attributions.reindex(
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
            "sum_attribution_score",
            "attribution_percent",
            "mapping_status"
        ]
    )
    
    final = (
        final
        .loc[
            final.attribution_rank <= 100
        ]
    )

    db_engine.execute(
        f"""delete from shap_values where censusdate = '{prediction_date}' and facilityid = '{facilityid}' and client = '{client}' and modelid = '{model.model_name}'"""
    )

    if save_outputs_in_s3 and (s3_location_path_prefix is not None):
        s3_path = s3_location_path_prefix + f'/explanations_output.parquet'
        final.to_parquet(s3_path, index=False)

    if save_outputs_in_local and (local_folder is not None):
        local_path = local_folder + f'/explanations_output.parquet'
        final.to_parquet(local_path, index=False)

    final.to_sql(
        "shap_values", db_engine, if_exists="append", index=False, method="multi"
    )
