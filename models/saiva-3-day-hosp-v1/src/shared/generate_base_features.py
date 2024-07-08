import gc
import sys
import timeit

import pandas as pd
from eliot import log_message, start_action

sys.path.insert(0, '/src')
from shared.utils import downcast_dtype


def sorter_and_deduper(df, sort_keys=[], unique_keys=[]):
    """
    input: dataframe
    output: dataframe
    desc: 1. sort the dataframe w.r.t sort_keys.
          2. drop duplicates w.r.t unique keys.(keeping latest values)
    """
    df.sort_values(by=sort_keys, inplace=True)
    df.drop_duplicates(
        subset=unique_keys,
        keep='last',
        inplace=True
    )
    assert df.duplicated(subset=unique_keys).sum() == 0, f'''Still have dupes!'''
    return df


def clean_multi_columns(cols):
    """
    input: list
    output: list
    desc: 1. Join sub-column names into a single column structure
            after pivoting dataframes      
    """
    new_cols = []
    for col in cols:
        if col[1] == '':
            new_cols.append(col[0])
        else:
            new_cols.append('_'.join(col))
    return new_cols


def get_census_demographic_features(patient_census, patient_demographics,
                                    train_start_date, prediction_date):
    """
    desc: 1. Creates base df having censusdate from train_start_date to test_end_date.
          2. merging base with patient_census w.r.t censusdate.
          3. merging above output with patient_demographics w.r.t masterpatientid.
    :return: dataframe
    """
    log_message(message_type='info', message='Create a Base by merging Census & Demographics')
    patient_demographics = sorter_and_deduper(
        patient_demographics,
        sort_keys=['masterpatientid'],
        unique_keys=['masterpatientid']
    )
    base = pd.DataFrame({'censusdate': pd.date_range(start=train_start_date, end=prediction_date)})
    base1 = base.merge(patient_census, how='left', on=['censusdate'])
    base2 = base1.merge(patient_demographics, how='left', on=['masterpatientid'])

    # =============Delete, Trigger garbage collection & downcast to free-up memory ==================
    del base1
    gc.collect()
    base2 = downcast_dtype(base2)

    # have to have only one row per masterpatientid, censusdate pair
    assert base2.duplicated(subset=['masterpatientid', 'censusdate']).any() == False

    return base2


def get_vitals_features(base, patient_vitals):
    """
    Apply Aggregate functions on diastolicvalue, warnings, bmi & vitalsdescription
    and create new columns. These columns are added to parent df 
    desc:
        1. Extracting diastolicvalue, warnings, bmi & vitalsdescription columns from patient_vitals 
           as a separate dataframe having ['masterpatientid', 'facilityid', 'date'] as index
        2. Filtering & replacing unwanted values in each of these dataframe
        3. pivoting the dataframes and applying aggregation functions.
        4. Merge the sub-column names of these individual dataframes
        5. Removing duplicates and merging all the above dataframes into a single dataframe
        6. Merging vitals with base to form base4 and returning the dataframe.
    :param base: dataframe
    :return: base4 : dataframe
    """
    log_message(message_type='info', message='Vitals Processing...')
    patient_vitals = sorter_and_deduper(
        patient_vitals,
        sort_keys=['masterpatientid', 'date'],
        unique_keys=['masterpatientid', 'date', 'vitalsdescription']
    )
    # diastolic value not in range(30,200) will be made Nan.
    patient_vitals.loc[:, 'diastolicvalue'] = (
        patient_vitals.loc[:, 'diastolicvalue'].mask(patient_vitals.loc[:, 'diastolicvalue'] > 200).mask(
            patient_vitals.loc[:, 'diastolicvalue'] < 30).values)
    # BP - Systolic value not in range(40,300) will be made Nan.
    patient_vitals.loc[patient_vitals.vitalsdescription == 'BP - Systolic', 'value'] = (
        patient_vitals.loc[patient_vitals.vitalsdescription == 'BP - Systolic', 'value'].mask(
            patient_vitals.loc[patient_vitals.vitalsdescription == 'BP - Systolic', 'value'] > 300).mask(
            patient_vitals.loc[patient_vitals.vitalsdescription == 'BP - Systolic', 'value'] < 40).values)
    # Respiration value not in range(6,50) will be made Nan.
    patient_vitals.loc[patient_vitals.vitalsdescription == 'Respiration', 'value'] = (
        patient_vitals.loc[patient_vitals.vitalsdescription == 'Respiration', 'value'].mask(
            patient_vitals.loc[patient_vitals.vitalsdescription == 'Respiration', 'value'] > 50).mask(
            patient_vitals.loc[patient_vitals.vitalsdescription == 'Respiration', 'value'] < 6).values)
    # Temperature value not in range(80,200) will be made Nan.
    patient_vitals.loc[patient_vitals.vitalsdescription == 'Temperature', 'value'] = (
        patient_vitals.loc[patient_vitals.vitalsdescription == 'Temperature', 'value']
        .mask(patient_vitals.loc[patient_vitals.vitalsdescription == 'Temperature', 'value'] > 200).mask(
            patient_vitals.loc[patient_vitals.vitalsdescription == 'Temperature', 'value'] < 80).values)
    # Pulse value not in range(20,300) will be made Nan.
    patient_vitals.loc[patient_vitals.vitalsdescription == 'Pulse', 'value'] = (
        patient_vitals.loc[patient_vitals.vitalsdescription == 'Pulse', 'value'].mask(
            patient_vitals.loc[patient_vitals.vitalsdescription == 'Pulse', 'value'] > 300).mask(
            patient_vitals.loc[patient_vitals.vitalsdescription == 'Pulse', 'value'] < 20).values)
    # O2 sats value below 80 will be made Nan.
    patient_vitals.loc[patient_vitals.vitalsdescription == 'O2 sats', 'value'] = (
        patient_vitals.loc[patient_vitals.vitalsdescription == 'O2 sats', 'value'].mask(
            patient_vitals.loc[patient_vitals.vitalsdescription == 'O2 sats', 'value'] < 80).values)
    # Weight value not in range(80,660) will be made Nan.
    patient_vitals.loc[patient_vitals.vitalsdescription == 'Weight', 'value'] = (
        patient_vitals.loc[patient_vitals.vitalsdescription == 'Weight', 'value'].mask(
            patient_vitals.loc[patient_vitals.vitalsdescription == 'Weight', 'value'] > 660).mask(
            patient_vitals.loc[patient_vitals.vitalsdescription == 'Weight', 'value'] < 80).values)
    # Blood Sugar value not in range(25,450) will be made Nan.
    patient_vitals.loc[patient_vitals.vitalsdescription == 'Blood Sugar', 'value'] = (
        patient_vitals.loc[patient_vitals.vitalsdescription == 'Blood Sugar', 'value'].mask(
            patient_vitals.loc[patient_vitals.vitalsdescription == 'Blood Sugar', 'value'] > 450).mask(
            patient_vitals.loc[patient_vitals.vitalsdescription == 'Blood Sugar', 'value'] < 25).values)
    # Pain Level value not in range(0,10) will be made Nan.
    patient_vitals.loc[patient_vitals.vitalsdescription == 'Pain Level', 'value'] = (
        patient_vitals.loc[patient_vitals.vitalsdescription == 'Pain Level', 'value'].mask(
            patient_vitals.loc[patient_vitals.vitalsdescription == 'Pain Level', 'value'] > 10).mask(
            patient_vitals.loc[patient_vitals.vitalsdescription == 'Pain Level', 'value'] < 0).values)

    vitals = patient_vitals.set_index(keys=['masterpatientid', 'facilityid', 'date']).drop(columns='patientid')
    vitals['warnings'] = vitals.warnings.notna()  # Non-missing values get mapped to True
    # diastolic contains index + diastolicvalue column
    diastolic = vitals.pop('diastolicvalue')
    diastolic = diastolic.dropna()

    warnings = vitals.pop('warnings')
    bmi = vitals.pop('bmi')

    vitals = vitals.reset_index()
    diastolic = diastolic.reset_index()
    warnings = warnings.reset_index()
    bmi = bmi.reset_index()

    # remove non-numerical values
    bmi['bmi'] = bmi.bmi.replace(
        {'Height required': None, 'Height and weight required': None, 'Weight required': None}).astype(float)

    vitals['date'] = vitals.pop('date').dt.date
    diastolic['date'] = diastolic.pop('date').dt.date
    warnings['date'] = warnings.pop('date').dt.date
    bmi['date'] = bmi.pop('date').dt.date

    aggs = ['median', 'std', 'max', 'min']
    vitals_pivoted = vitals.pivot_table(
        index=['masterpatientid', 'facilityid', 'date'],
        values='value',
        columns='vitalsdescription',
        aggfunc=aggs
    ).reset_index()

    diastolic_pivoted = diastolic.pivot_table(
        index=['masterpatientid', 'facilityid', 'date'],
        values='diastolicvalue',
        aggfunc=aggs
    ).reset_index()

    warnings_pivoted = warnings.pivot_table(
        index=['masterpatientid', 'facilityid', 'date'],
        values='warnings',
        aggfunc=sum
    ).reset_index()

    bmi_pivoted = bmi.pivot_table(
        index=['masterpatientid', 'facilityid', 'date'],
        values='bmi',
        aggfunc=max
    ).reset_index()

    vitals_pivoted.columns = clean_multi_columns(vitals_pivoted.columns)
    diastolic_pivoted.columns = clean_multi_columns(diastolic_pivoted.columns)

    warnings_pivoted['date'] = pd.to_datetime(warnings_pivoted['date'])
    bmi_pivoted['date'] = pd.to_datetime(bmi_pivoted['date'])

    vitals_pivoted = vitals_pivoted.drop_duplicates(subset=['masterpatientid', 'date'], keep='last')
    diastolic_pivoted = diastolic_pivoted.drop_duplicates(subset=['masterpatientid', 'date'], keep='last')
    warnings_pivoted = warnings_pivoted.drop_duplicates(subset=['masterpatientid', 'date'], keep='last')
    bmi_pivoted = bmi_pivoted.drop_duplicates(subset=['masterpatientid', 'date'], keep='last')

    vitals_base = vitals_pivoted.merge(diastolic_pivoted, how='left', on=['masterpatientid', 'facilityid', 'date'])
    vitals_base2 = vitals_base.merge(warnings_pivoted, how='left', on=['masterpatientid', 'facilityid', 'date'])
    vitals_final = vitals_base2.merge(bmi_pivoted, how='left', on=['masterpatientid', 'facilityid', 'date'])
    vitals_final.columns = 'vtl_' + vitals_final.columns

    # ===============================Downcast===============================
    vitals_final = downcast_dtype(vitals_final)

    # ================================= Merge to base ==================================
    base1 = base.merge(
        vitals_final,
        how='left',
        left_on=['masterpatientid', 'facilityid', 'censusdate'],
        right_on=['vtl_masterpatientid', 'vtl_facilityid', 'vtl_date']
    )

    # =============Delete & Trigger garbage collection to free-up memory ==================
    del vitals
    del vitals_pivoted
    del diastolic_pivoted
    del warnings_pivoted
    del bmi_pivoted
    del vitals_base
    del vitals_base2
    del vitals_final
    del base
    gc.collect()

    assert base1.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
    return base1


def get_diagnosis_features(base, patient_diagnosis, s3_bucket):
    """
    - CSV file contains diagnosis mappings with ICD-10 code and categories, label etc
    - Merge these categories into patient_diagnosis df and use all possible CCS labels as new columns 
    - All diagnosis names becomes individaul columns 
    - Diagnosis name columns are added to parent df
    """
    log_message(message_type='info', message='Diagnosis Processing...')

    patient_diagnosis = sorter_and_deduper(
        patient_diagnosis,
        sort_keys=['masterpatientid', 'onsetdate', 'diagnosiscode'],
        unique_keys=['masterpatientid', 'onsetdate', 'diagnosiscode']
    )

    lookup_ccs = pd.read_csv(f's3://{s3_bucket}/data/lookup/ccs_dx_icd10cm_2019_1.csv')
    lookup_ccs.columns = lookup_ccs.columns.str.replace("'", "")
    lookup_ccs = lookup_ccs.apply(lambda x: x.str.replace("'", ""))
    patient_diagnosis['indicator'] = 1
    patient_diagnosis['diagnosiscode'] = patient_diagnosis.diagnosiscode.str.replace('.', '')
    patient_diagnosis['onsetdate'] = patient_diagnosis.onsetdate.dt.date

    patient_diagnosis_merged = patient_diagnosis.merge(
        lookup_ccs, how='left',
        left_on=['diagnosiscode'],
        right_on=['ICD-10-CM CODE']
    )
    patient_diagnosis_merged['ccs_label'] = patient_diagnosis_merged['MULTI CCS LVL 1 LABEL'] + ' - ' + \
                                            patient_diagnosis_merged['MULTI CCS LVL 2 LABEL']

    diagnosis_pivoted = patient_diagnosis_merged.loc[:,
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
    diagnosis_pivoted = downcast_dtype(diagnosis_pivoted)

    base1 = base.merge(
        diagnosis_pivoted,
        how='left',
        left_on=['masterpatientid', 'censusdate'],
        right_on=['dx_masterpatientid', 'dx_onsetdate']
    )

    # =============Delete & Trigger garbage collection to free-up memory ==================
    del base
    del diagnosis_pivoted
    gc.collect()

    assert base1.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
    return base1, patient_diagnosis_merged


def get_meds_features(base, patient_meds):
    """
    - gpiclass & gpisubclassdescription columns are extracted
      and added to parent df
    """
    log_message(message_type='info', message='Meds Processing...')
    patient_meds = sorter_and_deduper(
        patient_meds,
        sort_keys=['masterpatientid', 'orderdate', 'gpiclass', 'gpisubclassdescription'],
        unique_keys=['masterpatientid', 'orderdate', 'gpiclass', 'gpisubclassdescription']
    )

    # copy corresponding gpiclass value for all None gpisubclassdescription
    # gpisubclassdescription is the actual medication name which will be one hot encoded
    patient_meds.loc[patient_meds.gpisubclassdescription.isna(), 'gpisubclassdescription'] = patient_meds.loc[
        patient_meds.gpisubclassdescription.isna(), 'gpiclass']

    patient_meds['orderdate'] = patient_meds.orderdate.dt.date
    patient_meds['indicator'] = 1
    meds_pivoted = patient_meds.loc[:,
                   ['masterpatientid', 'orderdate', 'gpisubclassdescription', 'indicator']].pivot_table(
        index=['masterpatientid', 'orderdate'],
        columns=['gpisubclassdescription'],
        values='indicator',
        fill_value=0).reset_index()

    # Add med_ to all column names    
    meds_pivoted.columns = 'med_' + meds_pivoted.columns

    meds_pivoted = meds_pivoted.drop_duplicates(
        subset=['med_masterpatientid', 'med_orderdate']
    )

    meds_pivoted['med_orderdate'] = pd.to_datetime(meds_pivoted.med_orderdate)

    # ===============================Downcast===============================
    meds_pivoted = downcast_dtype(meds_pivoted)

    base1 = base.merge(
        meds_pivoted,
        how='left',
        left_on=['masterpatientid', 'censusdate'],
        right_on=['med_masterpatientid', 'med_orderdate']
    )

    # =============Delete & Trigger garbage collection to free-up memory ==================
    del base
    del meds_pivoted
    gc.collect()

    assert base1.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
    return base1, patient_meds


def get_orders_features(base, patient_orders):
    """
    - diagnostic_orders_count & dietsupplement_count columns added  
    - diettype, diettexture, dietsupplement values are made as separate columns and added to parent df
    """
    log_message(message_type='info', message='Orders Processing...')
    patient_orders = sorter_and_deduper(
        patient_orders,
        sort_keys=['masterpatientid', 'orderdate'],
        unique_keys=['masterpatientid', 'orderdate', 'orderdescription']
    )
    # =========================Count of diagnostic_orders========================
    diagnostic_orders = patient_orders.loc[patient_orders.ordercategory == 'Diagnostic']
    diagnostic_orders['orderdate'] = diagnostic_orders.orderdate.dt.date
    diagnostic_orders['count_indicator_diagnostic_orders'] = 1

    diagnostic_pivoted = diagnostic_orders.drop(
        columns=['patientid', 'ordercategory', 'ordertype', 'orderdescription', 'pharmacymedicationname',
                 'diettype', 'diettexture', 'dietsupplement']).pivot_table(
        index=['masterpatientid', 'facilityid', 'orderdate'], values=['count_indicator_diagnostic_orders'],
        aggfunc=sum
    ).reset_index()

    diagnostic_pivoted['orderdate'] = pd.to_datetime(diagnostic_pivoted.orderdate)
    diagnostic_pivoted.columns = 'order_' + diagnostic_pivoted.columns

    base1 = base.merge(
        diagnostic_pivoted,
        how='left',
        left_on=['masterpatientid', 'facilityid', 'censusdate'],
        right_on=['order_masterpatientid', 'order_facilityid', 'order_orderdate']
    )

    del base
    del diagnostic_pivoted
    del diagnostic_orders

    # =========================diettype columns added========================
    diet_orders = patient_orders[patient_orders.ordercategory == 'Dietary - Diet']
    diet_orders['orderdate'] = diet_orders.orderdate.dt.date
    diet_orders['indicator'] = 1
    diet_orders = diet_orders.drop_duplicates(
        subset=['masterpatientid', 'orderdate', 'diettype', 'diettexture']
    )

    diet_type_pivoted = diet_orders.loc[:, ['masterpatientid', 'orderdate', 'diettype', 'indicator']].pivot_table(
        index=['masterpatientid', 'orderdate'],
        columns=['diettype'],
        values='indicator',
        aggfunc=min
    ).reset_index()

    diet_type_pivoted.head()
    diet_type_pivoted['orderdate'] = pd.to_datetime(diet_type_pivoted.orderdate)
    diet_type_pivoted.columns = 'order_' + diet_type_pivoted.columns
    base2 = base1.merge(
        diet_type_pivoted,
        how='left',
        left_on=['masterpatientid', 'censusdate'],
        right_on=['order_masterpatientid', 'order_orderdate']
    )
    del base1
    del diet_type_pivoted

    # =========================diettexture columns added========================
    diet_texture_pivoted = diet_orders.loc[:,
                           ['masterpatientid', 'orderdate', 'diettexture', 'indicator']].pivot_table(
        index=['masterpatientid', 'orderdate'],
        columns=['diettexture'],
        values='indicator',
        aggfunc=min
    ).reset_index()

    diet_texture_pivoted['orderdate'] = pd.to_datetime(diet_texture_pivoted.orderdate)
    diet_texture_pivoted.columns = 'order_' + diet_texture_pivoted.columns
    base3 = base2.merge(
        diet_texture_pivoted,
        how='left',
        left_on=['masterpatientid', 'censusdate'],
        right_on=['order_masterpatientid', 'order_orderdate']
    )

    del base2
    del diet_texture_pivoted

    # =========================dietsupplement columns added========================
    diet_supplements = patient_orders[patient_orders.ordercategory == 'Dietary - Supplements']
    if len(diet_supplements):
        diet_supplements['orderdate'] = diet_supplements.orderdate.dt.date
        diet_supplements['indicator'] = 1
        diet_supplements = diet_supplements.drop_duplicates(subset=['masterpatientid', 'orderdate', 'dietsupplement'])

        diet_supplements_pivoted = diet_supplements.loc[:,
                                   ['masterpatientid', 'orderdate', 'dietsupplement', 'indicator']].pivot_table(
            index=['masterpatientid', 'orderdate'],
            columns='dietsupplement',
            values='indicator',
            aggfunc=min).reset_index()

        diet_supplements_pivoted['orderdate'] = pd.to_datetime(diet_supplements_pivoted.orderdate)

        # =========================dietsupplement count column added========================
        diet_supplements_counts = diet_supplements.groupby(
            ['masterpatientid', 'facilityid', 'orderdate']).dietsupplement.count().reset_index().rename(
            columns={'dietsupplement': 'count_indicator_dietsupplement'})

        diet_supplements_counts['orderdate'] = pd.to_datetime(diet_supplements_counts.orderdate)
        diet_supplements_pivoted.columns = 'order_' + diet_supplements_pivoted.columns
        diet_supplements_counts.columns = 'order_' + diet_supplements_counts.columns

        base4 = base3.merge(
            diet_supplements_pivoted,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['order_masterpatientid', 'order_orderdate']
        )

        base5 = base4.merge(
            diet_supplements_counts,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['order_masterpatientid', 'order_orderdate']
        )
        del diet_supplements_pivoted
        del base4
        del diet_supplements_counts
        
    else:
        base5 = base3.copy()
    # drop any duplicated columns
    base5 = base5.loc[:, ~base5.columns.duplicated()]

    del base3
    

    # =============Trigger garbage collection & downcast to free-up memory ==================
    gc.collect()
    base5 = downcast_dtype(base5)

    assert base5.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
    return base5


def get_alerts_features(base, patient_alerts):
    """
    - alertdescription values are made as columns
    - For each type of `triggereditemtype` create column indicating its count for a
      given masterpatientid & createddate
    """
    log_message(message_type='info', message='Alerts Processing...')
    patient_alerts = sorter_and_deduper(
        patient_alerts,
        sort_keys=['masterpatientid', 'createddate'],
        unique_keys=['masterpatientid', 'createddate', 'alertdescription']
    )
    # ==================Filter triggereditemtype=T and alertdescription values made as columns=====================
    patient_alerts_system = patient_alerts.loc[patient_alerts.triggereditemtype.notna()]
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
        base1 = base.merge(
            patient_alerts_therapy_pivoted,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['alert_masterpatientid', 'alert_createddate']
        )
    else:
        base1 = base.copy()
    del base
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
        base1 = base1.merge(
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
        base1 = base1.merge(
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
        base1 = base1.merge(
            order_alert_counts,
            how='left',
            left_on=['masterpatientid', 'censusdate'],
            right_on=['alert_masterpatientid', 'alert_createddate']
        )
    
    # =================alertdescription values made as columns===================
    nonsystem_alerts = patient_alerts.loc[patient_alerts.triggereditemtype.isna()]
    nonsystem_alerts['createddate'] = nonsystem_alerts.createddate.dt.normalize()
    nonsystem_alerts['indicator'] = 1
    nonsystem_alerts = nonsystem_alerts.loc[nonsystem_alerts.alertdescription != '-1']
    alerts_pivoted = nonsystem_alerts.loc[:,
                     ['masterpatientid', 'createddate', 'alertdescription', 'indicator']].pivot_table(
        index=['masterpatientid', 'createddate'],
        columns=['alertdescription'],
        values=['indicator'],
        aggfunc=sum).reset_index()

    alerts_pivoted.columns = clean_multi_columns(alerts_pivoted.columns)
    alerts_pivoted.columns = 'alert_' + alerts_pivoted.columns

    base2 = base1.merge(
        alerts_pivoted,
        how='left',
        left_on=['masterpatientid', 'censusdate'],
        right_on=['alert_masterpatientid', 'alert_createddate']
    )

    # drop any duplicated columns
    base2 = base2.loc[:, ~base2.columns.duplicated()]

    # =============Trigger garbage collection & downcast to free-up memory ==================
    del base1
    gc.collect()
    base2 = downcast_dtype(base2)

    assert base2.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
    return base2


def get_rehosp_features(base, patient_rehosps, patient_census):
    """
    - last_hospitalisation_date, days_since_last_hosp per masterpatientid & censusdate
    - Boolean indicating re-hospitalised in next 3 or 7 days
    """
    log_message(message_type='info', message='Rehosp Processing...')
    patient_rehosps = sorter_and_deduper(
        patient_rehosps,
        sort_keys=['masterpatientid', 'dateoftransfer'],
        unique_keys=['masterpatientid', 'dateoftransfer']
    )

    patient_rehosps['dateoftransfer'] = pd.to_datetime(patient_rehosps.dateoftransfer.dt.date)
    rehosp = patient_rehosps.merge(patient_census, on=['masterpatientid'])
    last_hosp = rehosp[rehosp.dateoftransfer < rehosp.censusdate]
    last_hosp['count_prior_hosp'] = last_hosp.groupby(
        ['masterpatientid', 'censusdate']).dateoftransfer.cumcount() + 1

    # applying groupby last_hosp on 'masterpatientid', 'censusdate'. Taking the last row 
    # from the group and renaming dateoftransfer to last_hosp_date.
    last_hosp = last_hosp.groupby(['masterpatientid', 'censusdate']).tail(
        n=1).loc[:, ['masterpatientid', 'censusdate', 'dateoftransfer', 'count_prior_hosp']].rename(
        columns={'dateoftransfer': 'last_hosp_date'})

    last_hosp['days_since_last_hosp'] = (last_hosp.censusdate - last_hosp.last_hosp_date).dt.days

    # next_hosp dataframe is formed from rehosp df where dateoftransfer > censusdate and applying
    # groupby on 'masterpatientid', 'censusdate'. Taking the first row from the group and renaming
    # dateoftransfer to next_hosp_date.
    next_hosp = rehosp[rehosp.dateoftransfer >= rehosp.censusdate].groupby(
        ['masterpatientid', 'censusdate']).head(n=1).loc[:, ['masterpatientid', 'censusdate', 'dateoftransfer']
                ].rename(columns={'dateoftransfer': 'next_hosp_date'})

    # Check whether paient was re-hospitalised in next 3 or 7 days and create boolean column for the same
    next_hosp['target_3_day_hosp'] = (next_hosp.next_hosp_date - next_hosp.censusdate) <= pd.to_timedelta('4 days')
    next_hosp['target_7_day_hosp'] = (next_hosp.next_hosp_date - next_hosp.censusdate) <= pd.to_timedelta('8 days')

    last_hosp.columns = 'hosp_' + last_hosp.columns
    next_hosp.columns = 'hosp_' + next_hosp.columns

    # ===============================Downcast===============================
    last_hosp = downcast_dtype(last_hosp)
    next_hosp = downcast_dtype(next_hosp)

    base1 = base.merge(
        last_hosp,
        how='left',
        left_on=['masterpatientid', 'censusdate'],
        right_on=['hosp_masterpatientid', 'hosp_censusdate']
    )

    base1 = base1.merge(
        next_hosp,
        how='left',
        left_on=['masterpatientid', 'censusdate'],
        right_on=['hosp_masterpatientid', 'hosp_censusdate']
    )

    # df.columns.duplicated() returns a list containing boolean 
    # values indicating whether a column is duplicate
    base1 = base1.loc[:, ~base1.columns.duplicated()]
    log_message(message_type='info', message='Created Base 12')

    # =============Trigger garbage collection to free-up memory ==================
    del base
    gc.collect()

    assert base1.duplicated(subset=['masterpatientid', 'censusdate']).any() == False
    return base1


def get_clean_df(base12):
    """
    desc:
        1. merge last_hosp and next_hosp to base11 to form final output
        2. drop duplicate columns, unwanted columns and check sanity of rows.
    :return: combined : dataframe
    """
    log_message(message_type='info', message='creating_01_output...')

    base12 = base12.loc[:, base12.columns[~base12.columns.str.contains(
        '_masterpatientid|_facilityid|vtl_date|hosp_date|onsetdate|orderdate|createddate|_x$|_y$')].tolist()]

    # make sure there are no two rows that have the same masterpatientid and censusdate.  If there is, it means
    # there is a bug somewhere in our feature processing pipeline
    assert base12.duplicated(subset=['masterpatientid', 'censusdate']).any() == False

    return base12


def base_feature_processing(result_dict, train_start_date, prediction_date, s3_bucket,
                            s3_location_path_prefix=None, save_outputs_in_s3=False,
                            local_folder=None, save_outputs_in_local=False):
    with start_action(action_type='base_feature_processing'):
        start_time = timeit.default_timer()
        log_message(message_type='info', message='Entering base_feature_processing', save_outputs_in_s3=save_outputs_in_s3,
                    s3_location_path_prefix=s3_location_path_prefix, local_folder=local_folder,
                    save_outputs_in_local=save_outputs_in_local)
        assert len(result_dict) != 0, f'''Empty Dictionary!'''
        # Since census is used multiple places sort & dedupe once
        result_dict['patient_census'] = sorter_and_deduper(
            result_dict['patient_census'],
            sort_keys=['masterpatientid', 'censusdate'],
            unique_keys=['masterpatientid', 'censusdate']
        )

        # Define a dictionary to hold intermediate dataframes
        intermediate_dfs = {}

        intermediate_dfs['base1'] = get_census_demographic_features(
            result_dict['patient_census'],
            result_dict['patient_demographics'],
            train_start_date,
            prediction_date
        )

        intermediate_dfs['base2'] = get_vitals_features(
            intermediate_dfs['base1'],
            result_dict['patient_vitals']
        )
        # =====================Trigger Garbage collector and free up memory========================
        if not save_outputs_in_local and not save_outputs_in_s3:
            del intermediate_dfs['base1']
            gc.collect()

        intermediate_dfs['base3'], result_dict['patient_diagnosis'] = get_diagnosis_features(
            intermediate_dfs['base2'],
            result_dict['patient_diagnosis'],
            s3_bucket
        )
        # =====================Trigger Garbage collector and free up memory========================
        if not save_outputs_in_local and not save_outputs_in_s3:
            del intermediate_dfs['base2']
            gc.collect()

        if len(result_dict['patient_meds']):
            intermediate_dfs['base4'], result_dict['patient_meds'] = get_meds_features(
                intermediate_dfs['base3'],
                result_dict['patient_meds']
            )
        else:
            # if there are no meds then copy base3 to base4.
            intermediate_dfs['base4'] = intermediate_dfs['base3']
        # =====================Trigger Garbage collector and free up memory========================
        if not save_outputs_in_local and not save_outputs_in_s3:
            del intermediate_dfs['base3']
            gc.collect()

        if len(result_dict['patient_orders']):
            intermediate_dfs['base5'] = get_orders_features(
                intermediate_dfs['base4'],
                result_dict['patient_orders']
            )
        else:
            # if there are no orders then copy base4 to base5.
            intermediate_dfs['base5'] = intermediate_dfs['base4']
        # =====================Trigger Garbage collector and free up memory========================
        if not save_outputs_in_local and not save_outputs_in_s3:
            del intermediate_dfs['base4']
            gc.collect()

        if len(result_dict['patient_alerts']):
            intermediate_dfs['base6'] = get_alerts_features(
                intermediate_dfs['base5'],
                result_dict['patient_alerts']
            )
        else:
            # if there are no alerts then copy base5 to base6.
            intermediate_dfs['base6'] = intermediate_dfs['base5']
        # =====================Trigger Garbage collector and free up memory========================
        if not save_outputs_in_local and not save_outputs_in_s3:
            del intermediate_dfs['base5']
            gc.collect()

        intermediate_dfs['base7'] = get_rehosp_features(
            intermediate_dfs['base6'],
            result_dict['patient_rehosps'],
            result_dict['patient_census']
        )
        # =====================Trigger Garbage collector and free up memory========================
        if not save_outputs_in_local and not save_outputs_in_s3:
            del intermediate_dfs['base6']
            gc.collect()

        intermediate_dfs['final'] = get_clean_df(intermediate_dfs['base7'])
        # =====================Trigger Garbage collector and free up memory========================
        if not save_outputs_in_local and not save_outputs_in_s3:
            del intermediate_dfs['base7']
            gc.collect()

        # ========================== downcast to free-up memory ===================================
        intermediate_dfs['final'] = downcast_dtype(intermediate_dfs['final'])

        # =================================TESTING PURPOSE=========================================
        save_intermediate_dfs(
            save_outputs_in_s3,
            s3_location_path_prefix,
            save_outputs_in_local,
            local_folder,
            intermediate_dfs
        )

        log_message(message_type='info', Dataframe_shape=intermediate_dfs['final'].shape)
        log_message(message_type='info', Total_time_taken=(timeit.default_timer() - start_time))
        return intermediate_dfs['final'], result_dict


def save_intermediate_dfs(save_outputs_in_s3, s3_location_path_prefix,
                          save_outputs_in_local, local_folder, df_dict):
    """
    Save intermediate dataframes for testing and comparision
    :param save_outputs_in_s3:
    :param s3_location_path_prefix:
    :param save_outputs_in_local:
    :param local_folder:
    :param df_dict: dictionary of dataframes
    """
    log_message(message_type='info', message='in save_intermediate_dfs')

    def save(path):
        for name, df in df_dict.items():
            log_message(message_type='info', message='saving intermediate output', name=name, path=path)
            df.to_parquet(path + f'/{name}_output.parquet', index=False)

    if save_outputs_in_s3 and (s3_location_path_prefix is not None):
        save(s3_location_path_prefix)
    if save_outputs_in_local and (local_folder is not None):
        save(local_folder)
