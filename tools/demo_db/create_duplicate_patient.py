"""
Creates a duplicate patient and duplicates all reacords for the patient in the relevant tables
- cd demo_db
- python create_duplicate_patient.py --patient_id 2796411 
This is the stratgey used to create the duplicate patient: New ids are generated by incrementing the previous max id for each table
- Create a temp table "temp_patient" and insert into that the current patientid, masterpatientid, mrn and the new values for those fields
- Create duplicate rows in all tables with the relevant patientid and use the temp_patient table to find the new patientid,
  masterpatientid & mrn
- Create a table duplicate_patient_log to keep track of all duplicate patients created in the database
- In places where there are foreign key references (e.g view_ods_physician_order_list_med & view_ods_physician_order_list_v2),
  use temp table to maintain referential integrity
- Drop the temp tables on completion of the script
"""
import os
import json
import sys
import fire
import boto3
import pandas as pd
from eliot import start_action, log_message, to_file
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.sql import text

to_file(sys.stdout)

region_name = "us-east-1"

def main(patient_id, env='dev', client='saiva_demo'):

    with start_action(action_type="get_secrets"):
        # Get database Passwords from AWS SecretsManager
        session = boto3.session.Session()
        secrets_client = session.client(
            service_name="secretsmanager", region_name=region_name
        )
        client_db_info = json.loads(
            secrets_client.get_secret_value(SecretId=f"{env}-sqlserver")[
                "SecretString"
            ]
        )

    with start_action(action_type="generate_db_urls"):
        client_url = URL(
            drivername="mssql+pyodbc",
            username=client_db_info["username"],
            password=client_db_info["password"],
            host=client_db_info["host"],
            port=client_db_info["port"],
            database=client,
            query={"driver": "ODBC Driver 17 for SQL Server"},
        )

    # Establish connection with client mysql 
    with start_action(
        action_type="connect_to_databases",
        client_url=repr(client_url)
    ):
        engine = create_engine(client_url, echo=False)

    with engine.connect().execution_options(autocommit=True) as con:
        # Create temp table for operations
        with start_action(action_type="create_duplicate_patient"):
            # This is equivalent to the MySQL command “CREATE TABLE IF NOT EXISTS” 
            # Reference https://docs.microsoft.com/en-us/sql/relational-databases/system-catalog-views/sys-objects-transact-sql?view=sql-server-ver15
            # i.e., checks whether duplicate_patient_log exists before creating the table
            # The purpose of duplicate_patient_log is to maintain a history of all duplicate patients created
            statement = text("""IF NOT EXISTS (
	                SELECT * FROM sysobjects WHERE name = 'duplicate_patient_log' AND xtype = 'U'
                ) CREATE TABLE duplicate_patient_log (
	            masterpatientid int NOT NULL, patientid int NOT NULL, patientmrn int NOT NULL, newmasterpatientid int NOT NULL, newpatientid int NOT NULL, newpatientmrn int NOT NULL)"""
            )
            con.execute(statement)

            statement = text("""CREATE TABLE temp_patient (
            	masterpatientid int NOT NULL,
            	patientid int NOT NULL,
            	patientmrn int NOT NULL,
            	newmasterpatientid int NOT NULL,
            	newpatientid int NOT NULL,
            	newpatientmrn int NOT NULL)"""
            )
            con.execute(statement)

            # Insert row into temp table
            statement = text("""WITH max_values AS (
                	SELECT
                		CASE WHEN max(masterpatientid) > 10000000 THEN max(masterpatientid) ELSE 10000000
                		END as maxmasterpatientid,
                		CASE WHEN max(patientid) > 10000000 THEN max(patientid) ELSE 10000000
                		END as maxpatientid,
                		max(CAST(patientmrn AS Int)) as maxpatientmrn
                	FROM
                		view_ods_facility_patient
                )
                INSERT INTO temp_patient
                SELECT
                	masterpatientid,
                	patientid,
                	patientmrn,
                	maxmasterpatientid+1,
                	maxpatientid+1,
                	CAST (maxpatientmrn+1 as varchar(35)) 
                FROM
                	view_ods_facility_patient, max_values
                WHERE patientid = :patient_id ;"""
            )
            con.execute(statement, patient_id=patient_id)

            statement = text("""INSERT INTO duplicate_patient_log SELECT * FROM temp_patient""")
            con.execute(statement, patient_id=patient_id)

            # Duplicate row in masterpatient table with new masterpatientid
            statement = text("""INSERT INTO view_ods_master_patient(
            	masterpatientid, gender, dateofbirth, education, citizenship, race, religion, [state], 
                primarylanguage, deleted
            )
            SELECT
            	tp.newmasterpatientid AS masterpatientid,
            	mp.gender,
            	mp.dateofbirth,
            	mp.education,
            	mp.citizenship,
            	mp.race,
            	mp.religion,
            	mp. [state],
            	mp.primarylanguage,
            	mp.deleted
            FROM
            	view_ods_master_patient mp
            	JOIN temp_patient tp ON mp.masterpatientid = tp.masterpatientid;"""
            )
            con.execute(statement)
            
            # Duplicate row in facilitypatient table with new patientid, masterpatientid and mrn
            statement = text("""INSERT INTO view_ods_facility_patient(
            	patientid, facilityid, masterpatientid, FirstName, LastName, patientmrn, PatientIDNumber, 
                originaladmissiondate, recentadmissiondate, initialadmissiondate, MasterPatientDeleted, PatientDeleted
            )
            SELECT
            	tp.newpatientid AS patientid,
            	fp.facilityid,
            	tp.newmasterpatientid AS masterpatientid,
            	concat(fp.FirstName, '-Dup'), 
            	concat(fp.LastName, '-Dup'), 
            	tp.newpatientmrn as patientmrn,
            	tp.newpatientmrn AS PatientIDNumber,
            	fp.originaladmissiondate,
            	fp.recentadmissiondate,
            	fp.initialadmissiondate,
            	fp.MasterPatientDeleted,
            	fp.PatientDeleted
            FROM
            	view_ods_facility_patient fp
            	JOIN temp_patient tp ON fp.patientid = tp.patientid;"""
            )
            con.execute(statement)

            # Duplicate row in admissionlog table with new patientid, masterpatientid and mrn
            statement = text("""INSERT into view_ods_hospital_transfers_admission_log(
            	FacilityID, PatientID, DateOfAdmission, AdmissionStatus, AdmittedFrom, PrimaryPhysicianID, 
                PrimaryPhysicianTitle, PrimaryPhysicianFirstName, PrimaryPhysicianLastName, PrimaryPhysicianProfession, 
                MedicalRecordNumber, HospitalDischargeDate, ToFromTypeDescription, PayerID, PayerType, PayerTypeDescription, 
                AdmissionInEffectiveDate, AdmittedWithinLast30Days, TransferredWithin30DaysOfAdmission
            ) SELECT 
            al.FacilityID,
            tp.newpatientid,
            al.DateOfAdmission,
            al.AdmissionStatus,
            al.AdmittedFrom,
            al.PrimaryPhysicianID,
            al.PrimaryPhysicianTitle,
            al.PrimaryPhysicianFirstName,
            al.PrimaryPhysicianLastName,
            al.PrimaryPhysicianProfession,
            tp.newpatientmrn,
            al.HospitalDischargeDate,
            al.ToFromTypeDescription,
            al.PayerID,
            al.PayerType,
            al.PayerTypeDescription,
            al.AdmissionInEffectiveDate,
            al.AdmittedWithinLast30Days,
            al.TransferredWithin30DaysOfAdmission
            FROM view_ods_hospital_transfers_admission_log al JOIN temp_patient tp on al.PatientID = tp.patientid;"""
            )
            con.execute(statement)
            
            # Duplicate row in admissionlog table with new patientid, masterpatientid and mrn
            statement = text("""INSERT into view_ods_hospital_transfers_transfer_log_v2(
            	FacilityID, PatientID, dateoftransfer, orderedbyid, transferreason, PrimaryPhysicianID, 
                PrimaryPhysicianTitle, PrimaryPhysicianFirstName, PrimaryPhysicianLastName, PrimaryPhysicianProfession, 
                otherreasonfortransfer, planned, ToFromType, PayerID, PayerType, PayerDescription, 
                outcome, LastAdmissionDate, lengthofstay, transferredwithin30daysofadmission, OriginalAdmissionDate,
                HospitalDischargeDate, PurposeOfStayID, PurposeOfStay, TransferredTo
            ) SELECT 
            tl.FacilityID,
            tp.newpatientid,
            tl.dateoftransfer,
            tl.orderedbyid,
            tl.transferreason,
            tl.PrimaryPhysicianID,
            tl.PrimaryPhysicianTitle,
            tl.PrimaryPhysicianFirstName,
            tl.PrimaryPhysicianLastName,
            tl.PrimaryPhysicianProfession,
            tl.otherreasonfortransfer,
            tl.planned,
            tl.ToFromType,
            tl.PayerID,
            tl.PayerType,
            tl.PayerDescription,
            tl.outcome,
            tl.LastAdmissionDate,
            tl.lengthofstay,
            tl.transferredwithin30daysofadmission,
            tl.OriginalAdmissionDate,
            tl.HospitalDischargeDate,
            tl.PurposeOfStayID,
            tl.PurposeOfStay,
            tl.TransferredTo
            FROM view_ods_hospital_transfers_transfer_log_v2 tl JOIN temp_patient tp on tl.PatientID = tp.patientid;"""
            )
            con.execute(statement)

            # Duplicate row in cr_alert table with new patientid
            statement = text("""WITH max_value AS (
            	SELECT
            		CASE WHEN max(alertid) > 10000000 THEN max(alertid)
            		ELSE 10000000
            		END as maxalertid
            		FROM
            			view_ods_cr_alert
            )
            INSERT into view_ods_cr_alert(
            	alertid, patientid, facilityid, createddate, stdalertid, alertdescription, triggereditemtype, ResolvedDate, Deleted
            ) SELECT 
            ROW_NUMBER() OVER (ORDER BY alertid) + max_value.maxalertid as alertid,
            tp.newpatientid,
            ca.facilityid,
            ca.createddate,
            ca.stdalertid,
            ca.alertdescription,
            ca.triggereditemtype,
            ca.ResolvedDate,
            ca.Deleted
                FROM view_ods_cr_alert ca join max_value on 1=1 JOIN temp_patient tp on ca.PatientID = tp.patientid;"""
            )
            con.execute(statement)

            # Duplicate row in daily_census table with new patientid
            statement = text("""INSERT into view_ods_daily_census_v2(
                patientid, censusdate, facilityid, bedid, beddescription, roomratetypedescription, 
                payercode, payername, carelevelcode, censusactioncode, CensusStatusID, CensusStatusCode, DeletedFlag
            ) SELECT 
            tp.newpatientid,
            dc.censusdate,
            dc.facilityid,
            dc.bedid,
            dc.beddescription,
            dc.roomratetypedescription,
            dc.payercode,
            dc.payername,
            dc.carelevelcode,
            dc.censusactioncode,
            dc.CensusStatusID,
            dc.CensusStatusCode,
            dc.DeletedFlag
                FROM view_ods_daily_census_v2 dc JOIN temp_patient tp on dc.PatientID = tp.patientid;"""
            )
            con.execute(statement)

            # Duplicate row in view_ods_physician_order_list_v2 & view_ods_physician_order_list_med tables with new patientid
            statement = text("""CREATE TABLE temp_physician_order_list (
            	physicianorderid int NOT NULL,
            	patientid int NOT NULL,
            	newphysicianorderid int NOT NULL,
            	newpatientid int NOT NULL)"""
            )
            con.execute(statement)

            # ---
            statement = text("""WITH max_values AS (
            	SELECT
            		CASE WHEN max(physicianorderid) > 10000000 THEN max(physicianorderid) ELSE 10000000
            		END AS maxphysicianorderid
            	FROM
            		view_ods_physician_order_list_v2
            )
            INSERT INTO temp_physician_order_list(
            	physicianorderid, patientid, newphysicianorderid, newpatientid
            )
            SELECT
                ol.physicianorderid,
                ol.patientid,
            	ROW_NUMBER() OVER (ORDER BY physicianorderid) + max_values.maxphysicianorderid AS newphysicianorderid,
            	tp.newpatientid
            FROM
            	view_ods_physician_order_list_v2 ol 
                JOIN max_values ON 1=1
            	JOIN temp_patient tp ON ol.patientid = tp.patientid;"""
            )
            con.execute(statement)

            statement = text("""INSERT INTO view_ods_physician_order_list_v2(
            	physicianorderid, patientid, facilityid, orderdate, ordercategory, ordertype, orderdescription, pharmacymedicationname,
                diettype, diettexture, dietsupplement, discontinueddate, MARStartDate, MAREndDate, OrderHoldStartDate,
                OrderHoldEndDate, OrderCreatedDate, OrderRevisionDate
            )
            SELECT
                tol.newphysicianorderid,
                tol.newpatientid,
                ol.facilityid,
                ol.orderdate,
                ol.ordercategory,
                ol.ordertype,
                ol.orderdescription,
                ol.pharmacymedicationname,
                ol.diettype,
                ol.diettexture,
                ol.dietsupplement,
                ol.discontinueddate,
                ol.MARStartDate,
                ol.MAREndDate,
                ol.OrderHoldStartDate,
                ol.OrderHoldEndDate,
                ol.OrderCreatedDate,
                ol.OrderRevisionDate
            FROM
            	view_ods_physician_order_list_v2 ol
            	JOIN temp_physician_order_list tol ON ol.physicianorderid = tol.physicianorderid;"""
            )
            con.execute(statement)

            statement = text("""INSERT INTO view_ods_physician_order_list_med(
            	PhysiciansOrderID, GPIClass, GPIClassDescription, GPISubClass, GPISubClassDescription
            )
            SELECT
                tol.newphysicianorderid,
                ol.GPIClass,
                ol.GPIClassDescription,
                ol.GPISubClass,
                ol.GPISubClassDescription
            FROM
            	view_ods_physician_order_list_med ol
            	JOIN temp_physician_order_list tol ON ol.PhysiciansOrderID = tol.physicianorderid;"""
            )
            con.execute(statement)
            
            statement = text("""drop table temp_physician_order_list""")
            con.execute(statement)

            # Duplicate row in view_ods_patient_diagnosis table with new patientid
            statement = text("""INSERT into view_ods_patient_diagnosis(
            	clientdiagnosisid, patientid, onsetdate, facilityid, diagnosiscode, diagnosisdesc, 
                classification, rank, resolveddate, struckout, deleted, revisiondate
            ) SELECT 
            pd.clientdiagnosisid,
            tp.newpatientid,
            pd.onsetdate,
            pd.facilityid,
            pd.diagnosiscode,
            pd.diagnosisdesc,
            pd.classification,
            pd.rank,
            pd.resolveddate,
            pd.struckout,
            pd.deleted,
            pd.revisiondate
                FROM view_ods_patient_diagnosis pd JOIN temp_patient tp on pd.patientid = tp.patientid;"""
            )
            con.execute(statement)

            # Duplicate row in view_ods_patient_diagnosis table with new patientid
            statement = text("""INSERT into view_ods_progress_note(
            	patientid, facilityid, progressnoteid, progressnotetype, createddate, sectionsequence, 
                section, notetextorder, notetext, deleted
            ) SELECT 
            tp.newpatientid,
            pn.facilityid,
            pn.progressnoteid,
            pn.progressnotetype,
            pn.createddate,
            pn.sectionsequence,
            pn.section,
            pn.notetextorder,
            pn.notetext,
            pn.deleted
                FROM view_ods_progress_note pn JOIN temp_patient tp on pn.patientid = tp.patientid;"""
            )
            con.execute(statement)

            # Duplicate row in view_ods_Patient_weights_vitals table with new patientid
            statement = text("""INSERT into view_ods_Patient_weights_vitals(
            	patientid, facilityid, date, bmi, vitalsdescription, value, diastolicvalue, 
                warnings, StrikeOutFlag, Cleared
            ) SELECT 
            tp.newpatientid,
            pv.facilityid,
            pv.date,
            pv.bmi,
            pv.vitalsdescription,
            pv.value,
            pv.diastolicvalue,
            pv.warnings,
            pv.StrikeOutFlag,
            pv.Cleared
                FROM view_ods_Patient_weights_vitals pv JOIN temp_patient tp on pv.patientid = tp.patientid;"""
            )
            con.execute(statement)
        
    with engine.connect() as con:
        statement = text("""select newpatientid, newpatientmrn, newmasterpatientid from temp_patient""")
        result = con.execute(statement)
        row = result.fetchone()
        log_message(
            message_type="new_patient", 
            new_patientid=row['newpatientid'], 
            new_patientmrn=row['newpatientmrn'], 
            new_masterpatientid=row['newmasterpatientid']
        )

    with engine.connect().execution_options(autocommit=True) as con:
        statement = text("""drop table temp_patient""")
        con.execute(statement)

if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes.
    fire.Fire(main)