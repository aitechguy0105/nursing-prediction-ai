import sys

sys.path.insert(0, '/src')
from clients.base import BaseClient


class Gulfshore(BaseClient):

    def get_prediction_queries(self, prediction_date, facilityid, train_start_date):
        """
        :param prediction_date:
        :param facilityid:
        :param train_start_date:
        :return: List of queries and a name which will be used as filename to save the result of query
        """
        return {
            'patient_vitals': f"""
                        select patientid, facilityid, date, bmi,
                        vitalsdescription, value, diastolicvalue, warnings
                        from view_ods_Patient_weights_vitals
                        where patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and date >= '{train_start_date}'
                        """,
            'master_patient_lookup': f"""
                        select patientid, facilityid, masterpatientid from view_ods_facility_patient
                        where patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        """,
            'patient_census': f"""
                        select patientid, censusdate, facilityid, bedid, beddescription, 
                        roomratetypedescription, payercode, carelevelcode
                        from view_ods_daily_census_v2 where
                        patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and censusdate>='{train_start_date}'
                        """,
            'patient_rehosps': f"""
                        select patientid, facilityid, dateoftransfer, purposeofstay, transferredto,
                        orderedbyid, transferreason, otherreasonfortransfer, planned,
                        hospitaldischargedate, primaryphysicianid 
                        from view_ods_hospital_transfers_transfer_log_v2
                        where patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and dateoftransfer >= '{train_start_date}'
                        """,
            'patient_demographics': f"""
                        select masterpatientid, gender, dateofbirth, education, citizenship, race,
                        religion, state, primarylanguage
                        from view_ods_master_patient
                        where masterpatientid in
                        (select masterpatientid from view_ods_daily_census_v2 a
                        left join view_ods_facility_patient b on a.patientid = b.patientid and a.facilityid = b.facilityid
                        where censusdate = '{prediction_date}' and a.facilityid = {facilityid})
                        """,
            'patient_diagnosis': f"""
                        select patientid, onsetdate, facilityid, diagnosiscode,
                        diagnosisdesc, classification, rank
                        from view_ods_patient_diagnosis
                        where patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and onsetdate >= '{train_start_date}'
                        """,
            'patient_meds': f"""
                        select distinct patientid, facilityid, orderdate,
                        gpiclass, gpisubclassdescription, orderdescription, 
                        pharmacymedicationname, PhysicianOrderID
                        from view_ods_physician_order_list_v2 a
                        inner join view_ods_physician_order_list_med b
                        on a.PhysicianOrderID = b.PhysiciansOrderID
                        where patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and orderdate >= '{train_start_date}'
                        """,
            'patient_orders': f"""
                        select distinct patientid, facilityid, orderdate, ordercategory, ordertype,
                        orderdescription, pharmacymedicationname, diettype, diettexture, dietsupplement
                        from view_ods_physician_order_list_v2
                        where patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and ordercategory in
                        ('Diagnostic', 'Enteral - Feeding', 'Dietary - Diet', 'Dietary - Supplements','Laboratory')
                        and orderdate >= '{train_start_date}'
                        """,
            'patient_alerts': f"""
                        select patientid, facilityid, createddate, stdalertid,
                        alertdescription, a.triggereditemtype, description
                        from [view_ods_cr_alert] a left join view_ods_cr_alert_triggered_item_type b
                        on a.triggereditemtype = b.triggereditemtype
                        where patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid}) and
                        ((a.triggereditemtype is not null))
                        and createddate >= '{train_start_date}'
                        """
        }

    def get_training_queries(self, client, test_end_date, train_start_date):
        """
        Training queries are not tied up to one facility and prediction date.
        So the queries are different from prediction flow
        :param client
        :param test_end_date:
        :param train_start_date:
        :return: List of queries and a name which will be used as filename to save the result of query
        """
        return {
            'master_patient_lookup': f"""
                        select patientid, facilityid, masterpatientid
                        from view_ods_facility_patient
                        where facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        """,
            'patient_census': f"""
                        select patientid, censusdate, facilityid, bedid, beddescription, 
                        roomratetypedescription, payercode, carelevelcode from view_ods_daily_census_v2
                        where censusdate between '{train_start_date}' and '{test_end_date}'
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        """,
            'patient_rehosps': f"""
                        select patientid, facilityid, dateoftransfer, purposeofstay, transferredto,
                        orderedbyid, transferreason, otherreasonfortransfer, planned,
                        hospitaldischargedate, primaryphysicianid 
                        from view_ods_hospital_transfers_transfer_log_v2
                        where dateoftransfer between '{train_start_date}' and '{test_end_date}'
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        """,
            'patient_demographics': f"""
                        select masterpatientid, gender, dateofbirth, education, citizenship, 
                        race, religion, state, primarylanguage from view_ods_master_patient
                        """,
            'patient_diagnosis': f"""
                        select patientid, onsetdate, facilityid, diagnosiscode, diagnosisdesc, classification, rank
                        from view_ods_patient_diagnosis
                        where onsetdate between '{train_start_date}' and '{test_end_date}'
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        """,
            'patient_vitals': f"""
                        select patientid, facilityid, date, bmi, vitalsdescription, value, diastolicvalue, warnings
                        from view_ods_Patient_weights_vitals
                        where date between '{train_start_date}' and '{test_end_date}'
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        and patientid in (select distinct patientid from view_ods_daily_census_v2 
                        where censusdate between '{train_start_date}' and '{test_end_date}')
                        """,
            'patient_meds': f"""
                        select distinct patientid, facilityid, orderdate, gpiclass, 
                        gpisubclassdescription, orderdescription, pharmacymedicationname, PhysicianOrderID
                        from view_ods_physician_order_list_v2 a
                        inner join view_ods_physician_order_list_med b
                        on a.PhysicianOrderID = b.PhysiciansOrderID 
                        where orderdate between '{train_start_date}' and '{test_end_date}';
                        """,
            'patient_orders': f"""
                        select distinct patientid, facilityid, orderdate, ordercategory, 
                        ordertype, orderdescription, pharmacymedicationname, diettype, diettexture, dietsupplement
                        from view_ods_physician_order_list_v2
                        where orderdate between '{train_start_date}' and '{test_end_date}'
                        and ordercategory in ('Diagnostic', 'Enteral - Feeding', 'Dietary - Diet', 'Dietary - Supplements','Laboratory')
                        """,
            'patient_alerts': f"""
                        select patientid, facilityid, createddate, stdalertid, 
                        alertdescription, a.triggereditemtype, description
                        from [view_ods_cr_alert] a left join view_ods_cr_alert_triggered_item_type b
                        on a.triggereditemtype = b.triggereditemtype
                        where createddate between '{train_start_date}' and '{test_end_date}' and 
                        ((a.triggereditemtype is not null))
                        """
        }

    def get_training_dates(self):
        return '2018-01-01', '2019-10-28'