import sys
import datetime

sys.path.insert(0, '/src')
from clients.base import Base


class Excels(Base):

    def get_prediction_queries(
        self,
        *,
        prediction_date: datetime.date,
        facilityid: str,
        train_start_date: datetime.date,
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
        return {
            'patient_census': f"""
                         select patientid, CAST(censusdate as datetime) as censusdate, facilityid, null as bedid, '' as beddescription, 
                         '' as roomratetypedescription, '' as payercode, '' as carelevelcode, '' as censusactioncode, 
                         payer as payername 
                         from view_ods_daily_census_v2 where patientid in (select patientid from view_ods_daily_census_v2
                         where censusdate = '{prediction_date}' and facilityid = {facilityid})
                         and censusdate BETWEEN '{train_start_date}' AND '{prediction_date}'
                         and discharge_date is null
                         """,
            'patient_vitals': f"""
                         select patientid, facilityid, date, bmi, vital_description as vitalsdescription, value,
                         diastolic_value as diastolicvalue, '' as warnings from view_ods_Patient_weights_vitals
                         where patientid in (select patientid from view_ods_daily_census_v2
                         where censusdate = '{prediction_date}' and facilityid = {facilityid})
                         and date BETWEEN '{train_start_date}' AND '{prediction_date}'
                         """,
            'patient_admissions': f"""
                         select patientid, facilityid, dateofadmission, '' as admissionstatus, 
                         '' as admittedfrom, '' as primaryphysicianid, '' as to_from_type 
                         from view_ods_bed
                         where patientid in (select patientid from view_ods_daily_census_v2
                         where censusdate = '{prediction_date}' and facilityid = {facilityid})
                         and dateofadmission BETWEEN '{train_start_date}' AND '{prediction_date}'
                         """,
            'master_patient_lookup': f"""
                                 select distinct patientid, facilityid, patientid as masterpatientid from view_ods_daily_census_v2
                                 where patientid in (select patientid from view_ods_daily_census_v2
                                 where censusdate = '{prediction_date}' and facilityid = {facilityid})
                                 """,
            'patient_rehosps': f"""
                         select patientid, facilityid, dateoftransfer, '' as purposeofstay, '' as transferredto,
                         '' as orderedbyid, '' as transferreason, '' as otherreasonfortransfer, '' as planned,
                         '' as hospitaldischargedate, '' as primaryphysicianid 
                         from view_ods_hospital_transfers_transfer_log_v2
                         where patientid in (select patientid from view_ods_daily_census_v2
                         where censusdate = '{prediction_date}' and facilityid = {facilityid})
                         and dateoftransfer between '{train_start_date}' and '{prediction_date}'
                         """,
            'patient_demographics': f"""
                         select patientid as masterpatientid, gender, dateofbirth, '' as education, 
                         '' as citizenship, race, religion, '' as state, language as primarylanguage 
                         from view_ods_master_patient where patientid in
                         (select patientid from view_ods_daily_census_v2
                         where censusdate = '{prediction_date}' and facilityid = {facilityid})
                         """,
            'patient_diagnosis': f"""
                         select patientid, diagnosed_date as onsetdate, facilityid, icd10_code as diagnosiscode,
                         diagnosis_desc as diagnosisdesc, clinical_category as classification, 
                         category as rank, '' as resolveddate 
                         from view_ods_patient_diagnosis where patientid in (select patientid from view_ods_daily_census_v2
                         where censusdate = '{prediction_date}' and facilityid = {facilityid})
                         and diagnosed_date BETWEEN '{train_start_date}' AND '{prediction_date}'
                         """,
            'patient_progress_notes': f"""
                         select patientid, facilityid, note_id as progressnoteid, note_type as progressnotetype,
                         created_date as createddate, '' as sectionsequence, '' as section, CAST(null as INTEGER) as notetextorder, 
                         note_text as notetext
                         from view_ods_progress_note
                         where patientid in (select patientid from view_ods_daily_census_v2
                         where censusdate = '{prediction_date}' and facilityid = {facilityid}) and
                         created_date BETWEEN '{train_start_date}' AND '{prediction_date}'
                         """,
            'patient_room_details': f"""
                                         select pc.patientid, pc.patientid as masterpatientid, pc.facilityid, 
                                         CAST(pc.censusdate as datetime) as censusdate, ad.payer as payername, ad.room, NULL as room_id,
                                         NULL as floor, NULL as floor_id, ad.unit, NULL as unit_id 
                                         from view_ods_daily_census_v2 pc
                                         JOIN view_ods_bed ad on (
                                             pc.facilityid = ad.facilityid AND pc.patientid = ad.patientid
                                             AND ad.dateofadmission IN (
                                                 SELECT MAX(adm.dateofadmission) FROM view_ods_bed AS adm GROUP BY adm.patientid
                                             )
                                         )
                                         WHERE pc.censusdate = '{prediction_date}'
                                         and pc.facilityid = {facilityid}
                                     """,
            'patient_orders': f"""
                        select  patientid, facilityid, start_date as orderdate, order_category as ordercategory, order_type as ordertype,
                        order_description as orderdescription
                        from view_ods_physician_order_list_v2
                        where patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and order_category in ('Enteral', 'Dietary','Labs')
                        and start_date BETWEEN '{train_start_date}' AND '{prediction_date}'
                        """,
            'patient_meds': f"""
                        select a.patientid, a.facilityid, b.start_date as orderdate, '' as gpiclass, '' as gpiclassdescription, 
                        '' as gpisubclassdescription, b.order_description as orderdescription, b.end_date as discontinueddate 
                        from view_ods_physician_order_list_v2 a inner join view_ods_physician_order_list_med b
                        on a.order_id = b.order_id
                        where a.patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and a.start_date >= '{train_start_date}'
                        """,
        }

    
    def get_training_queries(
        self, 
        *,
        train_start_date: datetime.date, 
        test_end_date: datetime.date, 
    ):
        """
        Training queries are not tied up to one facility and prediction date.
        So the queries are different from prediction flow
        :param test_end_date:
        :param train_start_date:
        :return: List of queries and a name which will be used as filename to save the result of query
        """
        return {
            'patient_vitals': f"""
                                 select patientid, facilityid, date, bmi, vital_description as vitalsdescription, value,
                                 diastolic_value as diastolicvalue, '' as warnings from view_ods_Patient_weights_vitals
                                 where date BETWEEN '{train_start_date}' AND '{test_end_date}'
                                 """,
            'patient_admissions': f"""
                                 select patientid, facilityid, dateofadmission, '' as admissionstatus, 
                                 '' as admittedfrom, '' as primaryphysicianid, '' as to_from_type 
                                 from view_ods_bed
                                 where dateofadmission BETWEEN '{train_start_date}' AND '{test_end_date}'
                                 """,
            'master_patient_lookup': f"""
                                         select distinct patientid, facilityid, patientid as masterpatientid from view_ods_daily_census_v2 
                                         """,
            'patient_census': f"""
                                 select patientid, CAST(censusdate as datetime) as censusdate, facilityid, null as bedid, '' as beddescription, 
                                 '' as roomratetypedescription, '' as payercode, '' as carelevelcode, '' as censusactioncode, 
                                 payer as payername 
                                 from view_ods_daily_census_v2 where censusdate BETWEEN '{train_start_date}' AND '{test_end_date}'
                                 """,
            'patient_rehosps': f"""
                                 select patientid, facilityid, dateoftransfer, '' as purposeofstay, '' as transferredto,
                                 '' as orderedbyid, '' as transferreason, '' as otherreasonfortransfer, '' as planned,
                                 '' as hospitaldischargedate, '' as primaryphysicianid 
                                 from view_ods_hospital_transfers_transfer_log_v2
                                 where dateoftransfer between '{train_start_date}' and '{test_end_date}'
                                 """,
            'patient_orders': f"""
                        select  patientid, facilityid, order_category as ordercategory, order_type as ordertype,
                        order_description as orderdescription
                        from view_ods_physician_order_list_v2
                        where start_date between '{train_start_date}' and '{test_end_date}'
                        and order_category in ('Enteral', 'Dietary','Labs')
                        """,
            
            'patient_demographics': f"""
                                 select patientid as masterpatientid, gender, dateofbirth, '' as education, 
                                 '' as citizenship, race, religion, '' as state, language as primarylanguage 
                                 from view_ods_master_patient
                                 """,
            'patient_diagnosis': f"""
                                 select patientid, diagnosed_date as onsetdate, facilityid, icd10_code as diagnosiscode,
                                 diagnosis_desc as diagnosisdesc, clinical_category as classification, 
                                 category as rank, '' as resolveddate
                                 from view_ods_patient_diagnosis where diagnosed_date BETWEEN '{train_start_date}' AND '{test_end_date}'
                                 """,
            'patient_progress_notes': f"""
                                 select patientid, facilityid, note_id as progressnoteid, note_type as progressnotetype,
                                 created_date as createddate, '' as sectionsequence, '' as section, '' as notetextorder, 
                                 note_text as notetext
                                 from view_ods_progress_note
                                 where created_date BETWEEN '{train_start_date}' AND '{test_end_date}'
                                 """
        }
