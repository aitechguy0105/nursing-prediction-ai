import sys

sys.path.insert(0, '/src')
from clients.base import BaseClient


class InfinityBenchmark(BaseClient):

    def get_prediction_queries(self, prediction_date, facilityid, train_start_date):
        """
        :param prediction_date:
        :param facilityid:
        :param train_start_date:
        :return: List of queries and a name which will be used as filename to save the result of query
        """
        return {
            'patient_vitals': f"""
                                SELECT b.masterpatientid, a.patientid, a.facilityid, date, bmi, 
                                vitalsdescription, value, diastolicvalue, warnings
                                FROM view_ods_Patient_weights_vitals a
                                INNER JOIN view_ods_facility_patient b ON a.patientid = b.patientid AND a.facilityid = b.facilityid
                                where a.patientid in (select distinct patientid from view_ods_daily_census_v2
                                where censusdate = '{prediction_date}' and facilityid = {facilityid})
                                and Date >= '{train_start_date}'
                                """,
            'patient_census': f"""
                                SELECT b.masterpatientid, a.patientid, a.facilityid, censusdate, 
                                bedid, beddescription, roomratetypedescription, payercode, carelevelcode
                                FROM view_ods_daily_census_v2 a
                                INNER JOIN view_ods_facility_patient b ON a.patientid = b.patientid AND a.facilityid = b.facilityid
                                where a.patientid in (select patientid from view_ods_daily_census_v2
                                where censusdate = '{prediction_date}' and facilityid = {facilityid})
                                """,
            'patient_rehosps': f"""
                                SELECT masterpatientid, a.patientid, a.facilityid, dateoftransfer, 
                                purposeofstay, transferredto, outcome, orderedbyid, 
                                transferreason, otherreasonfortransfer, planned,
                                transferredwithin30daysofadmission, lengthofstay, 
                                hospitaldischargedate, a.primaryphysicianid 
                                FROM view_ods_hospital_transfers_transfer_log_v2 a
                                INNER JOIN view_ods_facility_patient b ON a.patientid = b.patientid AND a.facilityid = b.facilityid
                                where a.patientid in (select patientid from view_ods_daily_census_v2
                                where censusdate = '{prediction_date}' and facilityid = {facilityid})
                                and dateoftransfer >= '{train_start_date}'
                                """,
            'patient_demographics': f"""
                                SELECT masterpatientid, gender, dateofbirth, education, 
                                citizenship, race, religion, state, primarylanguage
                                FROM view_ods_master_patient
                                where masterpatientid in
                                (select masterpatientid from view_ods_daily_census_v2 a
                                left join view_ods_facility_patient b on a.patientid = b.patientid and a.facilityid = b.facilityid
                                where censusdate = '{prediction_date}' and a.facilityid = {facilityid})
                                """,
            'patient_diagnosis': f"""
                                SELECT b.masterpatientid, a.patientid, a.facilityid, onsetdate, 
                                diagnosiscode, diagnosisdesc, classification, rank
                                FROM view_ods_patient_diagnosis a
                                INNER JOIN view_ods_facility_patient b ON a.patientid = b.patientid AND a.facilityid = b.facilityid
                                where a.patientid in (select patientid from view_ods_daily_census_v2
                                where censusdate = '{prediction_date}' and facilityid = {facilityid})
                                and onsetdate >= '{train_start_date}'
                                """,
            'patient_meds': f"""
                                SELECT DISTINCT c.masterpatientid, a.patientid, a.facilityid, 
                                orderdate, gpiclass, gpisubclassdescription, 
                                orderdescription, pharmacymedicationname, physicianorderid
                                FROM view_ods_physician_order_list_v2 a
                                INNER JOIN view_ods_physician_order_list_med b ON a.physicianorderid = b.physiciansorderid 
                                INNER JOIN view_ods_facility_patient c ON a.patientid = c.patientid AND a.facilityid = c.facilityid
                                where a.patientid in (select patientid from view_ods_daily_census_v2
                                where censusdate = '{prediction_date}' and facilityid = {facilityid})
                                and orderdate >= '{train_start_date}'
                                """,
            'patient_progress_notes': f"""
                                SELECT b.masterpatientid, a.patientid, a.facilityid, progressnoteid, progressnotetype, 
                                a.createddate, sectionsequence, section, notetextorder, notetext
                                FROM view_ods_progress_note a
                                INNER JOIN view_ods_facility_patient b ON a.patientid = b.patientid AND a.facilityid = b.facilityid
                                where a.patientid in (select patientid from view_ods_daily_census_v2
                                where censusdate = '{prediction_date}' and facilityid = {facilityid}) and
                                a.createddate >= '{train_start_date}'
                                """,
            'patient_orders': f"""
                                select distinct c.masterpatientid, a.patientid, a.facilityid, orderdate, ordercategory, 
                                ordertype, orderdescription, pharmacymedicationname, diettype, 
                                diettexture, dietsupplement
                                from view_ods_physician_order_list_v2 a
                                INNER JOIN view_ods_facility_patient c ON a.patientid = c.patientid AND a.facilityid = c.facilityid
                                where a.patientid in (select patientid from view_ods_daily_census_v2
                                where censusdate = '{prediction_date}' and facilityid = {facilityid})
                                and ordercategory in ('Diagnostic', 'Enteral - Feeding', 'Dietary - Diet', 'Dietary - Supplements')
                                and orderdate BETWEEN '{train_start_date}' AND '{prediction_date}'
                                """,
            'patient_alerts': f"""
                                select c.masterpatientid, a.patientid, a.facilityid, a.createddate, stdalertid, 
                                alertdescription,  a.triggereditemtype, description
                                from [view_ods_cr_alert] a 
                                INNER JOIN view_ods_facility_patient c ON a.patientid = c.patientid AND a.facilityid = c.facilityid
                                left join view_ods_cr_alert_triggered_item_type b on a.triggereditemtype = b.triggereditemtype
                                where a.patientid in (select patientid from view_ods_daily_census_v2
                                where censusdate = '{prediction_date}' and facilityid = {facilityid}) and
                                (a.triggereditemtype is not null)
                                and createddate BETWEEN '{train_start_date}' AND '{prediction_date}'
                                """,
            'patient_lab_results': f"""
                                select f.masterpatientid, c.patientid, c.facilityid, a.resultdate, b.orderdate, b.specimencollectiondate, 
                                b.ordernumber, a.profiledescription, a.referencerange, a.result, a.abnormalityid, 
                                e.abnormalitydescription, b.reportdesciption, b.severityid, d.severitydescription 
                                from view_ods_result_lab_report_detail a
                                left join view_ods_result_lab_report b on a.LabReportID = b.LabReportID
                                left join view_ods_result_order_source c on b.ResultOrderSourceID = c.ResultOrderSourceID
                                left join view_ods_result_lab_report_severity d on b.SeverityID = d.SeverityID
                                left join view_ods_result_lab_test_abnormality e on a.AbnormalityID = e.AbnormalityID
                                INNER JOIN view_ods_facility_patient f ON c.patientid = f.patientid AND c.facilityid = f.facilityid
                                WHERE a.resultdate BETWEEN '{train_start_date}' AND '{prediction_date}' 
                                AND c.facilityid = {facilityid}
                                """,
        }

    def get_training_queries(self, train_start_date, train_end_date):
        """
        Training queries are not tied up to one facility and prediction date.
        So the queries are different from prediction flow
        :param test_end_date:
        :param train_start_date:
        :return: List of queries and a name which will be used as filename to save the result of query
        """
        return {
            'patient_vitals': f"""
                                SELECT b.masterpatientid, a.patientid, a.facilityid, date, bmi, 
                                vitalsdescription, value, diastolicvalue, warnings
                                FROM view_ods_Patient_weights_vitals a
                                INNER JOIN view_ods_facility_patient b ON a.patientid = b.patientid AND a.facilityid = b.facilityid
                                WHERE Date BETWEEN '{train_start_date}' AND '{train_end_date}'
                                AND a.facilityid IN (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                                AND a.patientid IN (select distinct patientid from view_ods_daily_census_v2 
                                    WHERE censusdate BETWEEN '{train_start_date}' and '{train_end_date}')
                                """,
            'patient_census': f"""
                                SELECT b.masterpatientid, a.patientid, a.facilityid, censusdate, 
                                bedid, beddescription, roomratetypedescription, payercode, carelevelcode
                                FROM view_ods_daily_census_v2 a
                                INNER JOIN view_ods_facility_patient b ON a.patientid = b.patientid AND a.facilityid = b.facilityid
                                WHERE censusdate BETWEEN '{train_start_date}' AND '{train_end_date}'
                                AND a.facilityid IN (SELECT facilityid FROM view_ods_facility WHERE lineofbusiness = 'SNF')
                                """,
            'patient_rehosps': f"""
                                SELECT masterpatientid, a.patientid, a.facilityid, dateoftransfer, 
                                purposeofstay, transferredto, outcome, orderedbyid, 
                                transferreason, otherreasonfortransfer, planned,
                                transferredwithin30daysofadmission, lengthofstay, 
                                hospitaldischargedate, a.primaryphysicianid 
                                FROM view_ods_hospital_transfers_transfer_log_v2 a
                                INNER JOIN view_ods_facility_patient b ON a.patientid = b.patientid AND a.facilityid = b.facilityid
                                WHERE dateoftransfer BETWEEN '{train_start_date}' AND '{train_end_date}'
                                AND a.facilityid IN (SELECT facilityid FROM view_ods_facility WHERE lineofbusiness = 'SNF')
                                """,
            'patient_demographics': f"""
                                SELECT masterpatientid, gender, dateofbirth, education, 
                                citizenship, race, religion, state, primarylanguage
                                FROM view_ods_master_patient
                                """,
            'patient_diagnosis': f"""
                                SELECT b.masterpatientid, a.patientid, a.facilityid, onsetdate, 
                                diagnosiscode, diagnosisdesc, classification, rank
                                FROM view_ods_patient_diagnosis a
                                INNER JOIN view_ods_facility_patient b ON a.patientid = b.patientid AND a.facilityid = b.facilityid
                                WHERE onsetdate BETWEEN '{train_start_date}' and '{train_end_date}'
                                AND a.facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                                """,
            'patient_meds': f"""
                                SELECT DISTINCT c.masterpatientid, a.patientid, a.facilityid, 
                                orderdate, gpiclass, gpisubclassdescription, 
                                orderdescription, pharmacymedicationname, physicianorderid
                                FROM view_ods_physician_order_list_v2 a
                                INNER JOIN view_ods_physician_order_list_med b ON a.physicianorderid = b.physiciansorderid 
                                INNER JOIN view_ods_facility_patient c ON a.patientid = c.patientid AND a.facilityid = c.facilityid
                                WHERE orderdate BETWEEN '{train_start_date}' AND '{train_end_date}'
                                """,
            'patient_progress_notes': f"""
                                SELECT b.masterpatientid, a.patientid, a.facilityid, progressnoteid, progressnotetype, 
                                a.createddate, sectionsequence, section, notetextorder, notetext
                                FROM view_ods_progress_note a
                                INNER JOIN view_ods_facility_patient b ON a.patientid = b.patientid AND a.facilityid = b.facilityid
                                WHERE a.createddate BETWEEN '{train_start_date}' AND '{train_end_date}'
                                """,
            'patient_orders': f"""
                                SELECT DISTINCT c.masterpatientid, a.patientid, a.facilityid, orderdate, ordercategory, 
                                ordertype, orderdescription, pharmacymedicationname, diettype, 
                                diettexture, dietsupplement
                                FROM view_ods_physician_order_list_v2 a
                                INNER JOIN view_ods_facility_patient c ON a.patientid = c.patientid AND a.facilityid = c.facilityid
                                WHERE orderdate BETWEEN '{train_start_date}' AND '{train_end_date}'
                                AND ordercategory IN ('Diagnostic', 'Enteral - Feeding', 
                                                      'Dietary - Diet', 'Dietary - Supplements', 
                                                      'Laboratory')
                                """,
            'patient_alerts': f"""
                                SELECT c.masterpatientid, a.patientid, a.facilityid, a.createddate, stdalertid, 
                                alertdescription,  a.triggereditemtype, description
                                FROM [view_ods_cr_alert] a 
                                INNER JOIN view_ods_facility_patient c ON a.patientid = c.patientid AND a.facilityid = c.facilityid
                                LEFT JOIN view_ods_cr_alert_triggered_item_type b ON a.triggereditemtype = b.triggereditemtype
                                WHERE a.createddate BETWEEN '{train_start_date}' and '{train_end_date}' AND
                                ((a.triggereditemtype is not null))
                                """,
            'patient_lab_results': f"""
                                SELECT f.masterpatientid, c.patientid, c.facilityid, a.resultdate, b.orderdate, b.specimencollectiondate, 
                                    b.ordernumber, a.profiledescription, a.referencerange, a.result, a.abnormalityid, 
                                    e.abnormalitydescription, b.reportdesciption, b.severityid, d.severitydescription 
                                FROM view_ods_result_lab_report_detail a
                                LEFT JOIN view_ods_result_lab_report b ON a.labreportid = b.labreportid
                                LEFT JOIN view_ods_result_order_source c ON b.resultordersourceid = c.resultordersourceid
                                LEFT JOIN view_ods_result_lab_report_severity d ON b.severityid = d.severityid
                                LEFT JOIN view_ods_result_lab_test_abnormality e ON a.abnormalityid = e.abnormalityid
                                INNER JOIN view_ods_facility_patient f ON c.patientid = f.patientid AND c.facilityid = f.facilityid
                                WHERE a.resultdate BETWEEN '{train_start_date}' AND '{train_end_date}'
                                """
        }

    def get_note_embeddings_emar_types(self):
        return [
            'eMAR- Administration Note',
            'eMar - Medication Administration',
            'eMAR-Medication Administration Note',
            'Orders - Administration Note',
            'eMAR- Medication Administration Note'
        ]

    def get_note_embeddings_nan_threshold(self):
        return 0.30

    def get_training_dates(self):
        return '2018-01-01', '2020-03-10'
