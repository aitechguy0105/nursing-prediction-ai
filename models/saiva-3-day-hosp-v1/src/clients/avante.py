import sys

sys.path.insert(0, '/src')
from clients.base import BaseClient


class Avante(BaseClient):

    def get_prediction_queries(self, prediction_date, facilityid, train_start_date):
        """
        :param prediction_date:
        :param facilityid:
        :param train_start_date:
        NOTE: patient_census includes only those rows whose censusactioncode is not in -
        TO(Transfer out of hospital), DE(Deceased Date), RDE (Respite - Deceased Date), RDD (Respite - Discharge Date),
        TP(Therapeutic Leave),L (Leave of Absence/LOA),H (Bed Hold),HI (Hospital Leave- ALL INS),
        TLU (Therapeutic Leave Unpaid), HMU (Hospital Leave Unpaid), HL(Hospital Leave),TL(Therapeutic Leave Medicare),
        PBH(Private Bed HolD), DRA(Discharge Return Anticipated),DRNA(Discharge Return Not Anticipated)
        :return: List of queries and a name which will be used as filename to save the result of query
        - 'patient_room_details' is used only during predictions for reports & ranking
        """
        return {
            'patient_vitals': f"""
                        select clientid as patientid, facilityid, date, bmi, vitalsdescription, value, 
                        diastolicvalue, warnings from view_ods_Patient_weights_vitals
                        where clientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and date BETWEEN '{train_start_date}' AND '{prediction_date}'
                        and strikeoutflag ='N'
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
                        and censusdate>='{train_start_date}'
                        and censusactioncode not in ('DE', 'DRA', 'DRNA', 'H', 'HI', 'HL', 'HMU', 'L', 'PBH', 'RDD', 
                        'RDE', 'TL', 'TLU', 'TO', 'TP')
                        and (payername not like '%hospice%' or payername is null)
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
                        select patientid, onsetdate, facilityid, diagnosiscode, diagnosisdesc, classification, rank, resolveddate, 
                        deleted, struckout
                        from view_ods_patient_diagnosis where patientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and revisiondate BETWEEN '{train_start_date}' AND '{prediction_date}'
                        """,
            'patient_meds': f"""
                        select distinct patientid, facilityid, orderdate, gpiclass, 
                        gpiclassdescription, gpisubclassdescription, orderdescription, a.discontinueddate, a.MAREndDate 
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
                        and ordercategory in ('Diagnostic', 'Enteral - Feeding', 'Dietary - Diet', 'Dietary - Supplements','Laboratory')
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
                        a.result, a.LabReportID, b.severityid,
                        a.units, e.AbnormalityCode, a.abnormalityid, e.abnormalitydescription, b.reportdesciption,  
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
                        """,
            'patient_room_details': f"""
                        select c.clientid as patientid,fp.masterpatientid, c.facilityid, 
                        c.censusdate, fp.initialadmissiondate, c.payername, r.RoomDescription as room, r.RoomID as room_id,
                        r.Floor as floor, r.FloorID as floor_id, u.UnitDescription as unit, u.UnitID as unit_id 
                        from dbo.view_ods_daily_census_v2 c 
                        JOIN dbo.view_ods_facility_patient fp on (fp.facilityid = c.facilityid and fp.patientid = c.clientid)
                        JOIN dbo.view_ods_bed b on (b.BedID = c.bedid) 
                        JOIN dbo.view_ods_room r on (r.RoomID = b.RoomID) 
                        JOIN dbo.view_ods_unit u on (u.UnitID = b.UnitID) 
                        WHERE c.censusdate = '{prediction_date}'
                        and c.facilityid = {facilityid}
                    """
        }

    def get_training_queries(self, test_end_date, train_start_date):
        """
        Training queries are not tied up to one facility and prediction date.
        So the queries are different from prediction flow
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
                        and payername not like '%hospice%'
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
                        select patientid, onsetdate, facilityid, diagnosiscode, diagnosisdesc, classification, rank, resolveddate
                        from view_ods_patient_diagnosis where revisiondate between '{train_start_date}' and '{test_end_date}'
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        """,
            'patient_vitals': f"""
                        select clientid as patientid, facilityid, date, bmi, vitalsdescription, value, diastolicvalue, warnings
                        from view_ods_Patient_weights_vitals where date between '{train_start_date}' and '{test_end_date}'
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        and clientid in (select distinct clientid from view_ods_daily_census_v2 
                        where censusdate between '{train_start_date}' and '{test_end_date}')
                        and strikeoutflag ='N'
                        """,
            'patient_meds': f"""
                        select distinct patientid, facilityid, orderdate, gpiclass, gpiclassdescription, 
                        gpisubclassdescription, orderdescription, pharmacymedicationname, 
                        PhysicianOrderID, a.discontinueddate, a.MAREndDate 
                        from view_ods_physician_order_list_v2 a inner join view_ods_physician_order_list_med b
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
                        """,
            'patient_lab_results': f'''
                        select c.patientid, c.facilityid, a.resultdate, a.profiledescription, 
                        a.referencerange, a.result, a.LabReportID,
                        a.units, e.AbnormalityCode, a.abnormalityid, e.abnormalitydescription, b.reportdesciption, 
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

    def get_training_dates(self):
        return '2018-01-01', '2019-10-28'
