import datetime
import typing
from clients.base import Base
import sys

sys.path.insert(0, '/src')


class Communicare(Base):

    def get_prediction_queries(
        self,
        *,
        prediction_date: datetime.date,
        facilityid: str,
        train_start_date: datetime.date,
        excluded_censusactioncodes: typing.List[str]
    ):
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
        census_action_code = tuple(excluded_censusactioncodes)
        return {
            'patient_census': f"""
                        select clientid as patientid, censusdate, facilityid, bedid, 
                        beddescription, roomratetypedescription, payercode, carelevelcode, censusactioncode, payername 
                        from view_ods_daily_census_v2 where clientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and censusdate>='{train_start_date}'
                        and (lower(payername) NOT LIKE '%hospice%' or payername is null) 
                        and censusactioncode not in {census_action_code}
                        """,
            'patient_vitals': f"""
                        select clientid as patientid, facilityid, date, bmi, vitalsdescription, value, 
                        diastolicvalue, warnings from view_ods_Patient_weights_vitals
                        where clientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and date BETWEEN '{train_start_date}' AND '{prediction_date}'
                        and strikeoutflag ='N'
                        """,
            'patient_admissions': f"""
                        select patientid, facilityid, dateofadmission, admissionstatus, admittedfrom, 
                        primaryphysicianid, PrimaryPhysicianTitle,PrimaryPhysicianFirstName,
                        PrimaryPhysicianLastName,ToFromTypeDescription as to_from_type  
                        from view_ods_hospital_transfers_admission_log
                        where patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and dateofadmission BETWEEN '{train_start_date}' AND '{prediction_date}'
                        """,
            'master_patient_lookup': f"""
                        select patientid, facilityid, masterpatientid from view_ods_facility_patient
                        where patientid in (select clientid from view_ods_daily_census_v2
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
                        select patientid, onsetdate, facilityid, diagnosiscode, diagnosisdesc, classification, rank, resolveddate, 
                        deleted, struckout
                        from view_ods_patient_diagnosis where patientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and revisiondate BETWEEN '{train_start_date}' AND '{prediction_date}'
                        --ADDING CHECK FOR MAX TIMESTAMP--
                        and OnSetDate <= '2038-01-19'
                        """,
            'patient_meds': f"""
                        select distinct patientid, facilityid, orderdate, gpiclass, 
                        gpiclassdescription, gpisubclassdescription, orderdescription, a.discontinueddate, a.MAREndDate 
                        from view_ods_physician_order_list_v2 a inner join view_ods_physician_order_list_med b
                        on a.PhysicianOrderID = b.PhysiciansOrderID
                        where patientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and orderdate BETWEEN '{train_start_date}' AND '{prediction_date}'
                        --ADDING CHECK FOR MAX TIMESTAMP--
                        and (a.discontinueddate <= '2038-01-19' OR a.discontinueddate IS NULL)
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
                        alertdescription, triggereditemtype 
                        from view_ods_cr_alert 
                        where patientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid}) 
                        and (createddate BETWEEN '{train_start_date}' AND '{prediction_date}')
                        and alertdescription is NOT NULL
                        """,
            'patient_progress_notes': f"""
                        select patientid, facilityid, progressnoteid, progressnotetype, createddate, sectionsequence, 
                        section, notetextorder, notetext from view_ods_progress_note
                        where patientid in (select clientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid}) and
                        createddate BETWEEN '{train_start_date}' AND '{prediction_date}'
                        """,
            'patient_lab_results': f"""
                        select c.patientid, c.facilityid, a.resultdate, a.profiledescription, a.referencerange, 
                        a.result, a.abnormalityid, e.abnormalitydescription, b.reportdesciption, b.severityid, 
                        d.severitydescription, a.LabReportID, a.units, e.AbnormalityCode 
                        from view_ods_result_lab_report_detail a
                        left join view_ods_result_lab_report b on a.LabReportID = b.LabReportID
                        left join view_ods_result_order_source c on b.ResultOrderSourceID = c.ResultOrderSourceID
                        left join view_ods_result_lab_report_severity d on b.SeverityID = d.SeverityID
                        left join view_ods_result_lab_test_abnormality e on a.AbnormalityID = e.AbnormalityID
                        WHERE a.resultdate BETWEEN '{train_start_date}' AND '{prediction_date}' 
                        AND c.facilityid = {facilityid}
                        AND c.patientid is not null
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

    def get_training_queries(
        self, 
        *,
        train_start_date: datetime.date, 
        test_end_date: datetime.date, 
        excluded_censusactioncodes: typing.List[str]
    ):
        """
        Training queries are not tied up to one facility and prediction date.
        So the queries are different from prediction flow
        :param test_end_date:
        :param train_start_date:
        :return: List of queries and a name which will be used as filename to save the result of query
        """
        census_action_code = tuple(excluded_censusactioncodes)
        # facilities 1-41: 1, 3, 4, 6, 7, 9, 10, 11, 12, 15, 16, 18, 20, 21, 22, 23, 24, 25, 27, 28, 29, 32, 33, 34, 35, 36, 37, 38, 39, 41
        # facilities 42-75: 42, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75
        # facilities 76-110: 76, 78, 79, 80, 81, 82, 85, 86, 87, 88, 89, 90, 91, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110
        # facilities 111-142: 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142
        facilities = '76, 78, 79, 80, 81, 82, 85, 86, 87, 88, 89, 90, 91, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110'

        return {
            'patient_vitals': f"""
                        select clientid as patientid, facilityid, date, bmi, vitalsdescription, value, diastolicvalue, warnings
                        from view_ods_Patient_weights_vitals
                        where date between '{train_start_date}' and '{test_end_date}'
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        and facilityid in ({facilities})
                        and clientid in (select distinct clientid from view_ods_daily_census_v2 where censusdate between '{train_start_date}' and '{test_end_date}')
                        and strikeoutflag ='N'
                        """,
            'master_patient_lookup': f"""
                        select patientid, facilityid, masterpatientid
                        from view_ods_facility_patient
                        where facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        and facilityid in ({facilities})
                        """,
            'patient_census': f"""
                        select clientid as patientid, censusdate, facilityid, bedid, beddescription, 
                        roomratetypedescription, payercode, carelevelcode, censusactioncode, payername 
                        from view_ods_daily_census_v2
                        where censusdate between '{train_start_date}' and '{test_end_date}'
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        and facilityid in ({facilities})
                        and (lower(payername) NOT LIKE '%hospice%' or payername is null) 
                        and censusactioncode not in {census_action_code}
                        """,
            'patient_rehosps': f"""
                        select patientid, facilityid, dateoftransfer, purposeofstay, transferredto,
                        orderedbyid, transferreason, otherreasonfortransfer, planned,
                        hospitaldischargedate, primaryphysicianid 
                        from view_ods_hospital_transfers_transfer_log_v2
                        where dateoftransfer between '{train_start_date}' and '{test_end_date}'
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        and facilityid in ({facilities})
                        """,
            'patient_admissions': f"""
                        select patientid, facilityid, dateofadmission, admissionstatus, 
                        admittedfrom, primaryphysicianid, ToFromTypeDescription as to_from_type  
                        from view_ods_hospital_transfers_admission_log
                        where dateofadmission between '{train_start_date}' and '{test_end_date}'
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        and facilityid in ({facilities})
                        """,
            'patient_demographics': f"""
                        select masterpatientid, gender, dateofbirth, education, citizenship, 
                        race, religion, state, primarylanguage from view_ods_master_patient
                        """,
            'patient_diagnosis': f"""
                        select patientid, onsetdate, facilityid, diagnosiscode, diagnosisdesc, classification, rank, 
                        resolveddate
                        from view_ods_patient_diagnosis
                        where revisiondate between '{train_start_date}' and '{test_end_date}'
                        and facilityid in (select facilityid from view_ods_facility where lineofbusiness = 'SNF')
                        and facilityid in ({facilities})
                        """,
            'patient_meds': f"""
                        select distinct patientid, facilityid, orderdate, gpiclass, 
                        gpisubclassdescription, orderdescription, pharmacymedicationname, 
                        PhysicianOrderID, a.discontinueddate, a.MAREndDate
                        from view_ods_physician_order_list_v2 a
                        inner join view_ods_physician_order_list_med b
                        on a.PhysicianOrderID = b.PhysiciansOrderID 
                        where orderdate between '{train_start_date}' and '{test_end_date}'
                        and facilityid in ({facilities})
                        """,
            'patient_orders': f"""
                        select distinct patientid, facilityid, orderdate, ordercategory, 
                        ordertype, orderdescription, pharmacymedicationname, diettype, diettexture, dietsupplement
                        from view_ods_physician_order_list_v2
                        where orderdate between '{train_start_date}' and '{test_end_date}'
                        and facilityid in ({facilities})
                        and ordercategory in ('Diagnostic', 'Enteral - Feeding', 'Dietary - Diet', 'Dietary - Supplements','Laboratory')
                        """,
            'patient_alerts': f"""
                        select patientid, facilityid, createddate, stdalertid, 
                        alertdescription, triggereditemtype
                        from view_ods_cr_alert
                        where createddate between '{train_start_date}' and '{test_end_date}'
                        and facilityid in ({facilities})
                        and alertdescription is NOT NULL
                        """,
            'patient_progress_notes': f"""
                        select patientid, facilityid, progressnoteid, progressnotetype, 
                        createddate, sectionsequence, section, notetextorder, notetext
                        from view_ods_progress_note
                        where createddate between '{train_start_date}' and '{test_end_date}'
                        and facilityid in ({facilities})
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
                        and c.facilityid in ({facilities})
                        ''',
        }