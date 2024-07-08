import abc
import typing

import six

from saiva.model.shared.constants import INVALID_ACTIONTYPE

@six.add_metaclass(abc.ABCMeta)
class BaseClientMatrixcare(object):

    def __init__(self, facilities: typing.Optional[typing.List[int]]=None):
        self.facilities = "SELECT FacilityID FROM view_ods_facility where status='Active'"
        self.note_embeddings_nan_threshold = 0.1

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
            'patient_census': f"""
                         select patientid, patientid as masterpatientid, CAST(censusdate as datetime) as censusdate, facilityid, null as bedid, '' as beddescription, 
                         '' as roomratetypedescription, '' as payercode, '' as payertype, '' as carelevelcode, '' as censusactioncode, 
                         payer as payername , '' as day_since_bed_change
                         from view_ods_daily_census_v2 where patientid in (select patientid from view_ods_daily_census_v2
                         where censusdate = '{prediction_date}' and facilityid = {facilityid})
                         and censusdate BETWEEN '{train_start_date}' AND '{prediction_date}'
                         and discharge_date is null
                         and (lower(payer) not like '%hospice%' or payer is NULL)
                         and (status is NULL or status in ('In House'))  
                         """,
            'patient_vitals': f"""
                         select patientid, patientid as masterpatientid, facilityid, date, vital_description as vitalsdescription, value,
                         diastolic_value as diastolicvalue, '' as warnings from view_ods_Patient_weights_vitals
                         where patientid in (select patientid from view_ods_daily_census_v2
                         where censusdate = '{prediction_date}' and facilityid = {facilityid})
                         and date BETWEEN '{train_start_date}' AND '{prediction_date}'
                         """,
            'patient_admissions': f"""
                        select vob.patientid, vob.patientid as masterpatientid, vob.facilityid, vob.dateofadmission, '' as admissionstatus, 
                        '' as admittedfrom, '' as primaryphysicianid, vomp.primary_physician as primaryphysicianfirstname, 
                        '' as to_from_type 
                        from view_ods_bed vob
                        left join view_ods_master_patient vomp 
                        on vob.patientid = vomp.patientid 
                        and vob.facilityid = vomp.facilityid 
                        where vob.patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate = '{prediction_date}' and facilityid = {facilityid})
                        and (vob.dateofadmission BETWEEN '{train_start_date}' AND '{prediction_date}')
                        and vob.census_type NOT IN ('Pre-Admission', 'Bed Change')
                        """,
            'master_patient_lookup': f"""
                                 select distinct patientid, facilityid, patientid as masterpatientid, '' as allergies from view_ods_daily_census_v2
                                 where patientid in (select patientid from view_ods_daily_census_v2
                                 where censusdate = '{prediction_date}' and facilityid = {facilityid})
                                 """,
            'patient_rehosps': f"""
                         select patientid, patientid as masterpatientid, facilityid, dateoftransfer, '' as purposeofstay, '' as transferredto,
                         '' as orderedbyid, '' as transferreason, '' as otherreasonfortransfer, '' as planned,
                         '' as hospitaldischargedate, '' as primaryphysicianid 
                         from view_ods_hospital_transfers_transfer_log_v2
                         where patientid in (select patientid from view_ods_daily_census_v2
                         where censusdate = '{prediction_date}' and facilityid = {facilityid})
                         and dateoftransfer between '{train_start_date}' and '{prediction_date}'
                         """,
            'patient_demographics': f"""
                         select patientid as masterpatientid, gender, dateofbirth, '' as education, 
                         '' as citizenship, race, religion, '' as state, language as primarylanguage, '' as maritalstatus 
                         from view_ods_master_patient where patientid in
                         (select patientid from view_ods_daily_census_v2
                         where censusdate = '{prediction_date}' and facilityid = {facilityid})
                         and (deleted='N' or deleted is NULL)
                         """,
            'patient_diagnosis': f"""
                         select patientid, patientid as masterpatientid, diagnosed_date as onsetdate, facilityid, icd10_code as diagnosiscode,
                         diagnosis_desc as diagnosisdesc, clinical_category as classification, 
                         category as rank, '' as resolveddate 
                         from view_ods_patient_diagnosis where patientid in (select patientid from view_ods_daily_census_v2
                         where censusdate = '{prediction_date}' and facilityid = {facilityid})
                         and diagnosed_date BETWEEN '{train_start_date}' AND '{prediction_date}'
                         """,
            'patient_progress_notes': f"""
                         select patientid, patientid as masterpatientid, facilityid, note_id as progressnoteid, note_type as progressnotetype,
                         created_date as createddate, '' as sectionsequence, '' as section, CAST(null as INTEGER) as notetextorder, 
                         note_text as notetext, '' as highrisk, '' as showon24hr, '' as showonshift
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
            'patient_adt':f"""
                                WITH t1 AS (
                                    SELECT
                                        facilityid,
                                        patientid,
                                        dateoftransfer AS begineffectivedate,
                                        dateoftransfer,
                                        census_type AS actiontype,
                                        'No' AS planned 
                                    FROM dbo.view_ods_hospital_transfers_transfer_log_v2 vohttlv 
                                    UNION ALL
                                    SELECT
                                        facilityid,
                                        patientid,
                                        dateofadmission AS begineffectivedate,
                                        NULL AS dateoftransfer,
                                        census_type AS actiontype,
                                        'No' AS planned
                                    FROM view_ods_bed vob
                                )
                                SELECT
                                    facilityid,
                                    patientid,
                                    patientid as masterpatientid,
                                    t1.begineffectivedate,
                                    CASE
                                        WHEN actiontype LIKE 'Discharge%'
                                        OR actiontype LIKE 'Death%'
                                        OR actiontype LIKE 'Expired%' -- Basically if discharge or death. Not sure in MatrixCare actiontype patterns
                                        THEN t1.begineffectivedate
                                        ELSE next_begineffectivedates.begineffectivedate
                                    END AS endeffectivedate,
                                    dateoftransfer,
                                    actiontype,
                                    planned
                                FROM t1
                                OUTER APPLY (
                                    SELECT TOP 1 begineffectivedate
                                    FROM t1 AS t_next
                                    WHERE t1.patientid = t_next.patientid
                                    AND t1.facilityid = t_next.facilityid
                                    AND t1.begineffectivedate < t_next.begineffectivedate
                                    ORDER BY t_next.begineffectivedate ASC
                                ) next_begineffectivedates
                                where facilityid in ({self.facilities})
                                ORDER BY patientid, begineffectivedate
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
            'patient_vitals': f"""
                                select patientid, facilityid, date, vital_description as vitalsdescription, value,
                                diastolic_value as diastolicvalue, '' as warnings from view_ods_Patient_weights_vitals
                                where date BETWEEN '{train_start_date}' AND '{test_end_date}'
                                and facilityid in ({self.facilities})
                                and patientid in (select distinct patientid from view_ods_daily_census_v2
                                where censusdate between '{train_start_date}' and '{test_end_date}')
                                 """,
            'patient_admissions': f"""
                                 select patientid, facilityid, dateofadmission, '' as admissionstatus, 
                                 '' as admittedfrom, '' as primaryphysicianid, '' as to_from_type 
                                 from view_ods_bed
                                 where facilityid in ({self.facilities})
                                 and census_type NOT IN ('Pre-Admission', 'Bed Change')
                                 """,
            'master_patient_lookup': f"""
                                         select distinct patientid, facilityid, patientid as masterpatientid, '' as allergies from view_ods_daily_census_v2 WHERE facilityid IN ({self.facilities})
                                         """,
            'patient_census': f"""
                                 select patientid, patientid as masterpatientid, CAST(censusdate as datetime) as censusdate, facilityid, null as bedid, '' as beddescription, 
                                 '' as roomratetypedescription, '' as payercode,'' as payertype, '' as carelevelcode, '' as censusactioncode, 
                                 payer as payername , '' as day_since_bed_change
                                 from view_ods_daily_census_v2 where censusdate BETWEEN '{train_start_date}' AND '{test_end_date}'
                                 and (lower(payer) not like '%hospice%' or payer is NULL)
                                 """,
            'patient_rehosps': f"""
                                 select patientid, facilityid, dateoftransfer, '' as purposeofstay, '' as transferredto,
                                 '' as orderedbyid, '' as transferreason, '' as otherreasonfortransfer, 'No' as planned,
                                 '' as hospitaldischargedate, '' as primaryphysicianid 
                                 from view_ods_hospital_transfers_transfer_log_v2
                                 where dateoftransfer between '{train_start_date}' and '{test_end_date}'
                                 and facilityid in ({self.facilities})
                                 """,
            
            'patient_demographics': f"""
                                 select patientid as masterpatientid, gender, dateofbirth, '' as education, 
                                 '' as citizenship, race, religion, '' as state, language as primarylanguage, 
                                 '' as maritalstatus
                                 from view_ods_master_patient
                                 where (deleted='N' or deleted is NULL)
                                 """,
            'patient_diagnosis': f"""
                                 select patientid, diagnosed_date as onsetdate, facilityid, icd10_code as diagnosiscode,
                                 diagnosis_desc as diagnosisdesc, clinical_category as classification, 
                                 category as rank, '' as resolveddate
                                 from view_ods_patient_diagnosis where diagnosed_date BETWEEN '{train_start_date}' AND '{test_end_date}'
                                 and facilityid in ({self.facilities})
                                 and diagnosed_date <= current_timestamp
                                 """,
            'patient_progress_notes': f"""
                                 select patientid, facilityid, note_id as progressnoteid, note_type as progressnotetype,
                                 created_date as createddate, '' as sectionsequence, '' as section, '' as notetextorder, 
                                 note_text as notetext,  '' as highrisk, '' as showon24hr, '' as showonshift
                                 from view_ods_progress_note
                                 where created_date BETWEEN '{train_start_date}' AND '{test_end_date}'
                                 and facilityid in ({self.facilities})
                                 """,
            'patient_adt':f"""
                                WITH t1 AS (
                                    SELECT
                                        facilityid,
                                        patientid,
                                        dateoftransfer AS begineffectivedate,
                                        dateoftransfer,
                                        census_type AS actiontype,
                                        'No' AS planned 
                                    FROM dbo.view_ods_hospital_transfers_transfer_log_v2 vohttlv 
                                    UNION ALL
                                    SELECT
                                        facilityid,
                                        patientid,
                                        dateofadmission AS begineffectivedate,
                                        NULL AS dateoftransfer,
                                        census_type AS actiontype,
                                        'No' AS planned
                                    FROM view_ods_bed vob
                                )
                                SELECT
                                    facilityid,
                                    patientid,
                                    t1.begineffectivedate,
                                    CASE
                                        WHEN actiontype LIKE 'Discharge%'
                                        OR actiontype LIKE 'Death%'
                                        OR actiontype LIKE 'Expired%' -- Basically if discharge or death. Not sure in MatrixCare actiontype patterns
                                        THEN t1.begineffectivedate
                                        ELSE next_begineffectivedates.begineffectivedate
                                    END AS endeffectivedate,
                                    dateoftransfer,
                                    actiontype,
                                    planned
                                FROM t1
                                OUTER APPLY (
                                    SELECT TOP 1 begineffectivedate
                                    FROM t1 AS t_next
                                    WHERE t1.patientid = t_next.patientid
                                    AND t1.facilityid = t_next.facilityid
                                    AND t1.begineffectivedate < t_next.begineffectivedate
                                    ORDER BY t_next.begineffectivedate ASC
                                ) next_begineffectivedates
                                where facilityid in ({self.facilities})
                                ORDER BY patientid, begineffectivedate
                                """
    }

    def get_note_embeddings_nan_threshold(self):
        return self.note_embeddings_nan_threshold

    def validate_dataset(self, facilityid, dataset_name, dataset_len):
        # The base implementation makes sure the dataset always has data in it
        assert dataset_len != 0, f'''{dataset_name} , Empty Dataset!'''

    def get_experiment_dates(self):
        return {
            'train_start_date': None,
            'train_end_date': None,
            'validation_start_date': None,
            'validation_end_date': None,
            'test_start_date': None,
            'test_end_date': None,
        }
    
    def get_excluded_censusactioncodes(self):
        return []

    def get_excluded_actiontype(self):
        """
        Removing residents that are not present in the facility
        """
        return INVALID_ACTIONTYPE