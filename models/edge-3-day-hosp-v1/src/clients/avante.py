import sys

sys.path.insert(0, '/src')
from clients.base import BaseClient


class Avante(BaseClient):

    def get_prediction_queries(self, prediction_date, facilityid, train_start_date):
        """
        :param prediction_date:
        :param facilityid:
        :param train_start_date:
        :return: List of queries and a name which will be used as filename to save the result of query
        """
        return {
            'patient_vitals': f"""
                        select clientid as patientid, facilityid, date, bmi, vitalsdescription, value, 
                        diastolicvalue, warnings from view_ods_Patient_weights_vitals
                        where clientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and date BETWEEN '{train_start_date}' AND '{prediction_date}'
                        """,
            'master_patient_lookup': f"""
                        select patientid, facilityid, masterpatientid from view_ods_facility_patient
                        where patientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        """,
            'patient_census': f"""
                        select clientid as patientid, censusdate, facilityid, bedid, 
                        beddescription, roomratetypedescription, payercode, carelevelcode
                        from view_ods_daily_census_v2 where clientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        """,
            'patient_rehosps': f"""
                        select patientid, facilityid, dateoftransfer, purposeofstay, transferredto,
                        orderedbyid, transferreason, otherreasonfortransfer, planned,
                        hospitaldischargedate, primaryphysicianid from view_ods_hospital_transfers_transfer_log_v2
                        where patientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and dateoftransfer between '{train_start_date}' and '{prediction_date}'
                        """,
            'patient_demographics': f"""
                        select masterpatientid, gender, dateofbirth, education, citizenship, race, religion, 
                        state, primarylanguage from view_ods_master_patient where masterpatientid in
                        (select masterpatientid from view_ods_daily_census_v2 a
                        left join view_ods_facility_patient b on a.clientid = b.patientid and a.facilityid = b.facilityid
                        where censusdate = '{prediction_date}' and a.facilityid = {facilityid})
                        """,
            'patient_diagnosis': f"""
                        select patientid, onsetdate, facilityid, diagnosiscode, diagnosisdesc, classification, rank
                        from view_ods_patient_diagnosis where patientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and onsetdate BETWEEN '{train_start_date}' AND '{prediction_date}'
                        """,
            'patient_meds': f"""
                        select distinct patientid, facilityid, orderdate, gpiclass, gpiclassdescription, gpisubclassdescription, orderdescription
                        from view_ods_physician_order_list_v2 a inner join view_ods_physician_order_list_med b
                        on a.PhysicianOrderID = b.PhysiciansOrderID
                        where patientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and orderdate BETWEEN '{train_start_date}' AND '{prediction_date}'
                        """,
            'patient_orders': f"""
                        select distinct patientid, facilityid, orderdate, ordercategory, ordertype,
                        orderdescription, pharmacymedicationname, diettype, diettexture, dietsupplement
                        from view_ods_physician_order_list_v2
                        where patientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and ordercategory in ('Diagnostic', 'Enteral - Feeding', 'Dietary - Diet', 'Dietary - Supplements')
                        and orderdate BETWEEN '{train_start_date}' AND '{prediction_date}'
                        """,
            'patient_alerts': f"""
                        select patientid, facilityid, createddate, stdalertid,
                        alertdescription, a.triggereditemtype, description
                        from [view_ods_cr_alert] a left join view_ods_cr_alert_triggered_item_type b
                        on a.triggereditemtype = b.triggereditemtype
                        where patientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid}) and
                        ((triggereditemid is not null) or (a.triggereditemtype is not null))
                        and createddate BETWEEN '{train_start_date}' AND '{prediction_date}'
                        """,
            'patient_lab_results': f"""
                        select c.patientid, c.facilityid, a.resultdate, a.profiledescription, a.referencerange, 
                        a.result, a.abnormalityid, e.abnormalitydescription, b.reportdesciption, b.severityid, 
                        d.severitydescription from view_ods_result_lab_report_detail a
                        left join view_ods_result_lab_report b on a.LabReportID = b.LabReportID
                        left join view_ods_result_order_source c on b.ResultOrderSourceID = c.ResultOrderSourceID
                        left join view_ods_result_lab_report_severity d on b.SeverityID = d.SeverityID
                        left join view_ods_result_lab_test_abnormality e on a.AbnormalityID = e.AbnormalityID
                        WHERE a.resultdate BETWEEN '{train_start_date}' AND '{prediction_date}' 
                        AND c.facilityid = {facilityid}
                        """,
            'patient_progress_notes': f"""
                        select patientid, facilityid, progressnoteid, progressnotetype, createddate, sectionsequence, 
                        section, notetextorder, notetext from view_ods_progress_note
                        where patientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid}) and
                        createddate BETWEEN '{train_start_date}' AND '{prediction_date}'
                        """
        }

    def get_training_queries(self, client, test_end_date, train_start_date):
        """
        Training queries are not tied up to one facility and prediction date.
        So the queries are different from prediction flow
        :param client:
        :param test_end_date:
        :param train_start_date:
        :return: List of queries and a name which will be used as filename to save the result of query
        """
        return {
            'master_patient_lookup': f'''
                select patientid, facilityid, masterpatientid from view_ods_facility_patient
                where facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                ''',
            'patient_census': f""" 
                        select clientid as patientid, censusdate, facilityid, bedid, beddescription, roomratetypedescription, 
                        payercode, carelevelcode from view_ods_daily_census_v2 
                        where censusdate between '{train_start_date}' and '{test_end_date}' 
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        """,
            'patient_rehosps': f"""
                        select patientid, facilityid, dateoftransfer, purposeofstay, transferredto,
                        orderedbyid, transferreason, otherreasonfortransfer, planned,
                        hospitaldischargedate, primaryphysicianid from view_ods_hospital_transfers_transfer_log_v2
                        where dateoftransfer between '{train_start_date}' and '{test_end_date}'
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        """,
            'patient_demographics': f"""
                        select masterpatientid, gender, dateofbirth, education, citizenship, 
                        race, religion, state, primarylanguage from view_ods_master_patient
                        """,
            'patient_diagnosis': f"""
                        select patientid, onsetdate, facilityid, diagnosiscode, diagnosisdesc, classification, rank
                        from view_ods_patient_diagnosis where onsetdate between '{train_start_date}' and '{test_end_date}'
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        """,
            'patient_vitals': f"""
                        select clientid as patientid, facilityid, date, bmi, vitalsdescription, value, diastolicvalue, warnings
                        from view_ods_Patient_weights_vitals where date between '{train_start_date}' and '{test_end_date}'
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        and clientid in (select distinct clientid from view_ods_daily_census_v2 
                        where censusdate between '{train_start_date}' and '{test_end_date}')
                        """,
            'patient_meds': f"""
                        select distinct patientid, facilityid, orderdate, gpiclass, gpiclassdescription, 
                        gpisubclassdescription, orderdescription, pharmacymedicationname, PhysicianOrderID
                        from view_ods_physician_order_list_v2 a inner join view_ods_physician_order_list_med b
                        on a.PhysicianOrderID = b.PhysiciansOrderID 
                        where orderdate between '{train_start_date}' and '{test_end_date}';
                        """,
            'patient_orders': f"""
                        select distinct patientid, facilityid, orderdate, ordercategory, 
                        ordertype, orderdescription, pharmacymedicationname, diettype, diettexture, dietsupplement
                        from view_ods_physician_order_list_v2 
                        where orderdate between '{train_start_date}' and '{test_end_date}'
                        and ordercategory in ('Diagnostic', 'Enteral - Feeding', 'Dietary - Diet', 'Dietary - Supplements')
                        """,
            'patient_alerts': f"""
                        select patientid, facilityid, createddate, stdalertid, 
                        alertdescription, a.triggereditemtype, description
                        from [view_ods_cr_alert] a left join view_ods_cr_alert_triggered_item_type b
                        on a.triggereditemtype = b.triggereditemtype
                        where createddate between '{train_start_date}' and '{test_end_date}' and 
                        ((a.triggereditemtype is not null))
                        """,
            'patient_lab_results': f'''
                        select c.patientid, c.facilityid, a.resultdate, a.profiledescription, 
                        a.referencerange, a.result, a.abnormalityid, e.abnormalitydescription, b.reportdesciption, 
                        b.severityid, d.severitydescription from view_ods_result_lab_report_detail a
                        left join view_ods_result_lab_report b on a.LabReportID = b.LabReportID
                        left join view_ods_result_order_source c on b.ResultOrderSourceID = c.ResultOrderSourceID
                        left join view_ods_result_lab_report_severity d on b.SeverityID = d.SeverityID
                        left join view_ods_result_lab_test_abnormality e on a.AbnormalityID = e.AbnormalityID
                        WHERE a.resultdate BETWEEN '{train_start_date}' AND '{test_end_date}'
                        ''',
            'patient_progress_notes': f"""
                        select patientid, facilityid, progressnoteid, progressnotetype, createddate, sectionsequence, 
                        section, notetextorder, notetext from view_ods_progress_note
                        where createddate between '{train_start_date}' and '{test_end_date}'
                        """

        }

    def get_note_embeddings_valid_section(self):
        return [
            "eMAR-Medication Administration Note_Note Text",
            "* General NURSING Note_Note Text",
            "* Skilled Nursing Note_Note Text",
            "Weekly Nurses Skin Observation Note_Note Text",
            "Physician Progress Notes_Note Text",
            "Braden Data Tool_Note Text",
            "X Social Service Interview_Note Text",
            "Dietary RD/DTR Data Collection Progress Note_Note Text",
            "z R.T. Shift Note (7am-7pm)_Narrative Note",
            "z R.T. Shift Note (7pm - 7am)_Narrative Note",
            "* Dietary RD/DTR Progress Note_Note Text",
            "* Social Services Note_Note Text",
            "* Activity Note_Note Text",
            "* Physician Progress Note_Note Text",
            "eMar - Shift Level Administration Note_Note Text",
            "* Weekly Wound Documentation_Assessment",
            "* Activities Admission/Readmission NOte_Note Text",
            "* Weight Meeting Note_Note Text",
            "* Incident/Accident Note_Note Text",
            "* Vent/Trach Clinical Observation Note_Comments",
            "* Admission Note_Note Text",
            "MDS Note_Note Text",
            "* Skin / Wound Note_Assessment",
            "* Skin / Wound Note_Plan",
            "* Skin / Wound Note_Intervention",
            "Global Deterioration Scale Note_Note Text",
        ]

    def get_note_embeddings_emar_types(self):
        return ["eMAR-Medication Administration Note"]

    def get_note_embeddings_nan_threshold(self):
        return 0.1

    def get_hospice_payercodes(self):
        return ["HCR", "HST", "HP", "HM", "HSV", "HSP", "HMP", "HM1", "HM2", "HR"]

    def get_training_dates(self):
        return '2017-01-01', '2020-03-10'