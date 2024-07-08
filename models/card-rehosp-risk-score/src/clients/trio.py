import sys

sys.path.insert(0, '/src')
from clients.base import BaseClient


class Trio(BaseClient):

    def get_prediction_queries(self, start_date, end_date, facilityid):
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
            'master_patient_lookup': f"""
                        select patientid, facilityid, masterpatientid from view_ods_facility_patient
                        where patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate BETWEEN '{start_date}' and '{end_date}' and facilityid = {facilityid})
                        """,
            'patient_census': f"""
                        select patientid, censusdate, facilityid, bedid, 
                        beddescription, roomratetypedescription, payercode, carelevelcode
                        from view_ods_daily_census_v2 where censusdate BETWEEN '{start_date}' and '{end_date}'
                        and facilityid = {facilityid}
                        and censusactioncode not in ('DE', 'DRA', 'DRNA', 'H', 'HI', 'HL', 'HMU', 'L', 'PBH', 'RDD', 
                        'RDE', 'TL', 'TLU', 'TO', 'TP')
                        and (payername not like '%hospice%' or payername is null)
                        """,
            'patient_diagnosis': f"""
                        select patientid, onsetdate, facilityid, diagnosiscode, diagnosisdesc, classification, rank, resolveddate, 
                        deleted, struckout
                        from view_ods_patient_diagnosis where patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate BETWEEN '{start_date}' and '{end_date}' and facilityid = {facilityid})
                        and struckout='N' and deleted='N' and resolveddate is null and onsetdate < '{end_date}'
                        """,
            'patient_alerts': f"""
                        select patientid, facilityid, createddate, stdalertid,
                        alertdescription, triggereditemtype 
                        from view_ods_cr_alert 
                        where patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate BETWEEN '{start_date}' and '{end_date}' and facilityid = {facilityid}) 
                        and Deleted = 'N' and createddate < '{end_date}'
                        """,
            'patient_admissions': f"""
                                select patientid, facilityid, dateofadmission, admissionstatus, admittedfrom, primaryphysicianid 
                                from view_ods_hospital_transfers_admission_log
                                where patientid in (select patientid from view_ods_daily_census_v2
                                where censusdate BETWEEN '{start_date}' and '{end_date}' and facilityid = {facilityid})
                                """,
            'patient_rehosps': f"""
                        select patientid, facilityid, dateoftransfer, purposeofstay, transferredto,
                        orderedbyid, transferreason, otherreasonfortransfer, planned,
                        hospitaldischargedate, primaryphysicianid from view_ods_hospital_transfers_transfer_log_v2
                        where patientid in (select patientid from view_ods_daily_census_v2
                        where censusdate BETWEEN '{start_date}' and '{end_date}' and facilityid = {facilityid})
                        and dateoftransfer between '{start_date}' and '{end_date}'
                        """,
        }