"""
Creates a demo database from another database in sql server and annonymizes the data
cd demo_db
python create_demo_db.py --from_env prod --to_env dev --client palmgarden
To populate saivadb with the new demo organization, turn on the dag for saiva_demo in the relavant org
"""
import os
import json
import fire
import boto3
import pandas as pd
from eliot import start_action, start_task, to_file, log_message
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.sql import text


region_name = "us-east-1"

def main(from_env, to_env, client, demo_db_name='saiva_demo'):
    cmd = (f"python ../db_backup_restore.py --from_env {from_env} " + 
           f"--to_env {to_env} " +
           f"--client {client} " +
           f"--to_db {demo_db_name} ")
    ret_val = os.system(cmd)
    assert ret_val == 0, f'db_backup_restore.py failed'

    with start_action(action_type="get_secrets"):
        # Get database Passwords from AWS SecretsManager
        session = boto3.session.Session()
        secrets_client = session.client(
            service_name="secretsmanager", region_name=region_name
        )
        client_db_info = json.loads(
            secrets_client.get_secret_value(SecretId=f"{to_env}-sqlserver")[
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
            database=demo_db_name,
            query={"driver": "ODBC Driver 17 for SQL Server"},
        )

    # Establish connection with client mysql & postgres database
    with start_action(
        action_type="connect_to_databases",
        client_url=repr(client_url)
    ):
        engine = create_engine(client_url, echo=False)

    with engine.connect().execution_options(autocommit=True) as con:
        # Update MRN to random values
        statement = text("""UPDATE view_ods_facility_patient set patientmrn = ABS(CHECKSUM(NEWID()))""")
        con.execute(statement)
        
        # Add new column id
        statement = text("""ALTER table view_ods_facility_patient add id UniqueIdentifier NOT NULL default newid()""")
        con.execute(statement)

        # Shuffle first names
        statement = """
        WITH 
        first_names AS (
            SELECT row_number() over (order by NEWID()) n, FirstName AS NewFirstName FROM view_ods_facility_patient
        ),  
        ids AS (
            SELECT row_number() over (order by NEWID()) n, id as ref_id FROM view_ods_facility_patient
        )
        UPDATE view_ods_facility_patient 
        SET FirstName = first_names.NewFirstName 
        FROM first_names JOIN ids ON first_names.n = ids.n 
        WHERE id=ids.ref_id"""
        
        con.execute(statement)

        # Shuffle last names
        statement = """
        WITH 
        last_names AS (
            SELECT row_number() over (order by NEWID()) n, LastName AS NewLastName FROM view_ods_facility_patient
        ),  
        ids AS (
            SELECT row_number() over (order by NEWID()) n, id as ref_id FROM view_ods_facility_patient
        )
        UPDATE view_ods_facility_patient 
        SET LastName = last_names.NewLastName 
        FROM last_names JOIN ids ON last_names.n = ids.n 
        WHERE id=ids.ref_id"""
        con.execute(statement)

        # Mask Facility names to Facility #
        statement = text(
            """UPDATE UpdateTarget SET  FacilityName = concat('Facility #', RowNum) FROM 
            (SELECT  X.FacilityName, ROW_NUMBER() OVER(ORDER BY NewID()) AS RowNum FROM view_ods_facility x) AS UpdateTarget"""
        )
        con.execute(statement)

        # Setting facility names to demo-friendly names
        # Create and populate temp_facilities table 

        statement = text(
            """create table temp_facilities(n INTEGER not NULL, name VARCHAR(255) NOT NULL)"""
        )
        con.execute(statement)

        statement = text(
            """insert into temp_facilities(name, n) VALUES
                ('Harbour Quay Health and Rehab', 1),
                ('Blue Ridge Health and Rehab', 2),
                ('Seminole Health and Rehab', 3),
                ('Graceland Health and Rehab', 4),
                ('Old Saybrook Health and Rehab', 5),
                ('East Port Health and Rehab', 6),
                ('Cumberland Health and Rehab', 7),
                ('Fort McHenry Health and Rehab', 8),
                ('Lincoln City Health and Rehab', 9),
                ('Sunnyvale Health and Rehab', 10),
                ('Gold Beach Health and Rehab', 11),
                ('Casper Health and Rehab', 12),
                ('Aurora Health and Rehab', 13),
                ('Taos Health and Rehab', 14),
                ('Twin Falls Health and Rehab', 15),
                ('Sun Valley Health and Rehab', 16);"""
            )
        con.execute(statement)

        statement = text(
            """WITH f AS (
                SELECT ROW_NUMBER() OVER (ORDER BY facilityid) AS n, facilityname FROM view_ods_facility
                WHERE MRNIdentifier IS NOT NULL AND lineofbusiness = 'SNF'
            ),
            tf AS (
                SELECT name, facilityname FROM f JOIN temp_facilities tf ON f.n = tf.n
            )
            UPDATE tf SET tf.facilityname = tf.name;"""
        )
        con.execute(statement)

        statement = text(
            """drop TABLE temp_facilities;"""
        )
        con.execute(statement)

        # Mask Unit names to one among (North, South, East, West) by using modulus 4 of unit id
        statement = text(
            """WITH NewUnit AS(SELECT * FROM (VALUES(0,'North'),(1,'South'),(2,'East'),(3,'West'))AS X(id,name)),
            TempUnit AS(SELECT ut.UnitDescription,ut.FacilityID,nut.name FROM view_ods_unit ut JOIN NewUnit nut ON(ut.UnitID%4)=nut.id)
            UPDATE TempUnit SET TempUnit.UnitDescription=TempUnit.name;"""
        )
        con.execute(statement)

        # To generate sequence in the format 1-99A, 1-99B upto 1-99Z
        # Not being used now since trio doesn't have view_ods_room
        sequence_generator = """WITH x AS (SELECT n FROM (VALUES (0),(1),(2),(3),(4),(5),(6),(7),(8),(9)) v(n)),
        sequence as (SELECT ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) as sequence_number
        FROM x ones, x tens),
        letters as (select CHAR(sequence_number) as letter from SEQUENCE where sequence_number BETWEEN 65 and 90)
        SELECT concat(d.sequence_number, l.letter) from letters l CROSS JOIN SEQUENCE d where d.sequence_number < 100"""

        # Trio doesn't have view_ods_floor
        # # Mask Floor names to Floor #
        # statement = text(
        #     """UPDATE UpdateTarget
        #     SET  FloorDesc = concat('Floor #', RowNum)
        #     FROM
        #     (
        #         SELECT  X.FloorDesc, ROW_NUMBER() OVER(PARTITION BY FacilityID ORDER BY NewID()) AS RowNum
        #         FROM    view_ods_floor x
        #     ) AS UpdateTarget"""
        # )
        # con.execute(statement)

        # Trio doesn't have view_ods_room
        # # Mask Room names to Room #
        # statement = text(
        #     """UPDATE UpdateTarget
        #     SET  RoomDescription = concat('Room #', RowNum)
        #     FROM
        #     (
        #         SELECT  X.RoomDescription, ROW_NUMBER() OVER(PARTITION BY UnitID, FloorID ORDER BY NewID()) AS RowNum
        #         FROM    view_ods_room x
        #     ) AS UpdateTarget"""
        # )
        # con.execute(statement)

        # Mask floors as 1st Floor and 2nd Floor
        statement = text(
            """WITH NewFloor AS (
                	SELECT * FROM (VALUES(0,'1st Floor'), (1, '2nd Floor')) AS X (id, name)),
                TempFloor AS (
                	SELECT
                		bed.[Floor],
                		nf.name
                	FROM
                		view_ods_bed bed
                		JOIN NewFloor nf ON (bed.FloorId % 2) = nf.id
                )
                UPDATE
                	TempFloor
                SET
                	TempFloor.[Floor] = TempFloor.name;"""
        )
        con.execute(statement)

        # Shift view_ods_hospital_transfers_transfer_log_v2 dates 
        statement = text(
            """UPDATE view_ods_hospital_transfers_transfer_log_v2 
            SET DateOfTransfer = DATEADD(day, -1 * (PatientID % 90), DateOfTransfer),
            LastAdmissionDate = DATEADD(day, -1 * (PatientID % 90), LastAdmissionDate),
            OriginalAdmissionDate = DATEADD(day, -1 * (PatientID % 90), OriginalAdmissionDate),
            HospitalDischargeDate = DATEADD(day, -1 * (PatientID % 90), HospitalDischargeDate)"""
        )
        con.execute(statement)

        # Shift view_ods_hospital_transfers_admission_log dates 
        statement = text(
            """UPDATE view_ods_hospital_transfers_admission_log 
            SET AdmissionInEffectiveDate = DATEADD(day, -1 * (PatientID % 90), AdmissionInEffectiveDate),
            DateOfAdmission = DATEADD(day, -1 * (PatientID % 90), DateOfAdmission),
            HospitalDischargeDate = DATEADD(day, -1 * (PatientID % 90), HospitalDischargeDate)"""
        )
        con.execute(statement)

        # Shift view_ods_facility_patient dates 
        statement = text(
            """UPDATE view_ods_facility_patient 
            SET originaladmissiondate = DATEADD(day, -1 * (PatientID % 90), originaladmissiondate),
            recentadmissiondate = DATEADD(day, -1 * (PatientID % 90), recentadmissiondate),
            initialadmissiondate = DATEADD(day, -1 * (PatientID % 90), initialadmissiondate)"""
        )
        con.execute(statement)

        # Shift view_ods_daily_census_v2 dates 
        statement = text(
            """UPDATE view_ods_daily_census_v2 
            SET censusdate = DATEADD(day, -1 * (patientid % 90), censusdate)"""
        )
        con.execute(statement)

        # Shift view_ods_master_patient dates 
        statement = text(
            """UPDATE view_ods_master_patient 
            SET dateofbirth = DATEADD(day, -1 * (masterpatientid % 90), dateofbirth)"""
        )
        con.execute(statement)

        # Shift view_ods_patient_diagnosis dates 
        statement = text(
            """UPDATE view_ods_patient_diagnosis 
            SET OnSetDate = DATEADD(day, -1 * (PatientID % 90), OnSetDate),
            ResolvedDate = DATEADD(day, -1 * (PatientID % 90), ResolvedDate)"""
        )
        con.execute(statement)

        # Shift view_ods_Patient_weights_vitals dates 
        statement = text(
            """UPDATE view_ods_Patient_weights_vitals 
            SET date = DATEADD(day, -1 * (patientid % 90), date)"""
        )
        con.execute(statement)

        # Shift view_ods_physician_order_list_v2 dates 
        statement = text(
            """UPDATE view_ods_physician_order_list_v2 
            SET orderdate = DATEADD(day, -1 * (PatientID % 90), orderdate)"""
        )
        con.execute(statement)

        # Shift view_ods_physician_order_list_v2 dates 
        statement = text(
            """UPDATE view_ods_physician_order_list_v2 
            SET orderdate = DATEADD(day, -1 * (PatientID % 90), orderdate)"""
        )
        con.execute(statement)

        # # Shift view_ods_result_lab_report dates 
        # statement = text(
        #     """UPDATE view_ods_result_lab_report 
        #     SET CreatedDate = DATEADD(day, -1 * (PatientID % 90), CreatedDate)"""
        # )
        # con.execute(statement)

        # Shift view_ods_cr_alert dates 
        statement = text(
            """UPDATE view_ods_cr_alert 
            SET createddate = DATEADD(day, -1 * (PatientID % 90), createddate)"""
        )
        con.execute(statement)

        # Shift view_ods_progress_note dates 
        statement = text(
            """UPDATE view_ods_progress_note 
            SET createddate = DATEADD(day, -1 * (PatientID % 90), createddate)"""
        )
        con.execute(statement)

        # Create table for storing hospital names
        statement = text(
            """CREATE TABLE hospitals (
                hospitalid UniqueIdentifier NOT NULL default newid(), 
                name VARCHAR(255) NOT NULL
            )"""
        )
        con.execute(statement)

        # Insert into hospitals from admissions table
        statement = text(
            """INSERT INTO hospitals (name)
            SELECT DISTINCT TransferredTo from view_ods_hospital_transfers_transfer_log_v2 where TransferredTo is not null
            """
        )
        con.execute(statement)

        # Create table for storing physician names
        statement = text(
            """CREATE TABLE physicians (
                physicianid int NOT NULL PRIMARY KEY,
                firstname VARCHAR(255) NOT NULL,
                lastname VARCHAR(255) NOT NULL
            )"""
        )
        con.execute(statement)

        # Insert into hospitals from admissions table
        statement = text(
            """INSERT INTO physicians (physicianid, firstname, lastname)
            SELECT DISTINCT PrimaryPhysicianID, PrimaryPhysicianFirstName, PrimaryPhysicianLastName 
            FROM view_ods_hospital_transfers_transfer_log_v2 where PrimaryPhysicianID is not null
            UNION 
            SELECT DISTINCT PrimaryPhysicianID, PrimaryPhysicianFirstName, PrimaryPhysicianLastName 
            FROM view_ods_hospital_transfers_admission_log where PrimaryPhysicianID is not null
            """
        )
        con.execute(statement)

        # Create table for storing sample hospital names
        statement = text(
            """CREATE TABLE sample_hospitals (
                hospitalid UniqueIdentifier NOT NULL default newid(), 
                name VARCHAR(255) NOT NULL
            )"""
        )
        con.execute(statement)
        
        hospitals = pd.read_csv('Hospitals.csv')['NAME'].tolist()[:1000]  # Max number of rows allowed is 1000
        hospitals_str = ', '.join(["('" + x.replace("'", "''") + "')" for x in hospitals])
        statement = f"""INSERT INTO sample_hospitals(name) values {hospitals_str}"""
        con.execute(statement)

        
        # Create table for storing sample person names
        statement = text(
            """CREATE TABLE sample_person_names (
                personid UniqueIdentifier NOT NULL default newid(), 
                firstname VARCHAR(255) NOT NULL,
                lastname VARCHAR(255) NOT NULL
            )"""
        )
        con.execute(statement)

        persons = pd.read_csv('PersonNames.csv')
        persons_str = ', '.join([f"('{x['first_name']}', '{x['last_name']}')" for idx, x in persons.iterrows()])
        statement = f"""INSERT INTO sample_person_names(firstname, lastname) values {persons_str}"""
        con.execute(statement)

        # Shuffle hospital names
        statement = """
        WITH 
        original_names AS (
            SELECT row_number() over (order by name) n, name AS originalname FROM hospitals
        ),  
        new_names AS (
            SELECT row_number() over (order by name) n, name as newname FROM sample_hospitals
        )
        UPDATE view_ods_hospital_transfers_transfer_log_v2 
        SET TransferredTo = new_names.newname 
        FROM original_names JOIN new_names ON original_names.n = new_names.n 
        WHERE TransferredTo=original_names.originalname"""
        
        con.execute(statement)

        # Shuffle physician names in transfers
        statement = """
        WITH 
        original_names AS (
            SELECT row_number() over (order by physicianid) n, physicianid, firstname, lastname FROM physicians
        ),  
        new_names AS (
            SELECT row_number() over (order by firstname, lastname) n, firstname as newfirstname, lastname as newlastname FROM sample_person_names
        )
        UPDATE view_ods_hospital_transfers_transfer_log_v2 
        SET PrimaryPhysicianFirstName = new_names.newfirstname, PrimaryPhysicianLastName = new_names.newlastname
        FROM original_names JOIN new_names ON original_names.n = new_names.n 
        WHERE PrimaryPhysicianID=original_names.physicianid"""
        con.execute(statement)

        # Shuffle physician names in admissions
        statement = """
        WITH 
        original_names AS (
            SELECT row_number() over (order by physicianid) n, physicianid, firstname, lastname FROM physicians
        ),  
        new_names AS (
            SELECT row_number() over (order by firstname, lastname) n, firstname as newfirstname, lastname as newlastname FROM sample_person_names
        )
        UPDATE view_ods_hospital_transfers_admission_log 
        SET PrimaryPhysicianFirstName = new_names.newfirstname, PrimaryPhysicianLastName = new_names.newlastname
        FROM original_names JOIN new_names ON original_names.n = new_names.n 
        WHERE PrimaryPhysicianID=original_names.physicianid"""
        con.execute(statement)


if __name__ == "__main__":
    # fire lets us create easy CLI's around functions/classes.
    fire.Fire(main)