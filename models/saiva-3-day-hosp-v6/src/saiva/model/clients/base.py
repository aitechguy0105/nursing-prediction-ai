import abc
import typing

import six
from sqlalchemy.engine import Engine
from eliot import log_message

from saiva.model.shared.database import check_multiple_tables_exist, check_table_exists
from saiva.model.shared.constants import INVALID_ACTIONTYPE


@six.add_metaclass(abc.ABCMeta)
class Base(object):

    def __init__(
            self,
            facilities: typing.Optional[typing.List[int]] = None,
            engine: typing.Optional[Engine] = None,
    ):

        if facilities is None:
            self.facilities = "SELECT FacilityID FROM view_ods_facility WHERE LineOfBusiness = 'SNF' AND Deleted='N'"
        else:
            self.facilities = ",".join(str(facility_id) for facility_id in facilities)
        self.note_embeddings_nan_threshold = 0.1

        if (
            engine is not None
            and all(check_multiple_tables_exist(engine, ['wv_vitals', 'wv_std_vitals', 'wv_vitals_exception']))
        ):
            log_message(
                message_type="info",
                message=(
                    "detected presence of wv_vitals, wv_std_vitals, wv_vitals_exception tables in the database"
                    " using an optimized query istead of view_ods_Patient_weights_vitals."
                )
            )

            # modyfied query for view_ods_Patient_weights_vitals
            self.custom_vitals_query_prefix = """
                WITH custom_vitals AS (
                    SELECT wv_vitals.client_id      AS ClientId,
                        wv_vitals.fac_id            AS FacilityId,
                        wv_vitals.[date]            AS [Date],
                        wv_vitals.[value]           AS [Value],
                        wv_vitals.dystolic_value    AS DiastolicValue,
                        wv_vitals.strikeout_flag    AS StrikeOutFlag,
                        wv_std_vitals.description   AS VitalsDescription,
                        wv_vitals_exception.[description] + IsNull(
                            CASE
                                When wv_vitals_exception.std_vitals_id IN (1, 3) THEN --Weights , Blood Pressure
                                ISNULL(
                                    ' [ ' + CAST(
                                        CAST(wv_vitals_exception.[value] AS DECIMAL(9, 1)) AS VARCHAR(11)
                                    ) + '%',
                                    ''
                                ) + IsNull(
                                    ', ' + CAST(CAST(diff_value AS DECIMAL(9, 1)) AS VARCHAR(11)),
                                    ''
                                ) + ' ]; '
                                else NULL
                            End,
                            ''
                        ) AS Warnings
                    FROM wv_vitals WITH (NOLOCK)
                        LEFT JOIN wv_std_vitals WITH (NOLOCK) ON wv_vitals.std_vitals_id = wv_std_vitals.std_vitals_id
                        LEFT JOIN wv_vitals_exception WITH (NOLOCK) on wv_vitals.vitals_id = wv_vitals_exception.vitals_id
                    WHERE wv_vitals.baseline IS NULL
                )
                """
        else:
            self.custom_vitals_query_prefix = """
                WITH custom_vitals AS (
                    SELECT * FROM view_ods_Patient_weights_vitals
                )
                """

        if (
            engine is not None
            and all(check_multiple_tables_exist(engine, ['contact', 'common_code']))
        ):
            log_message(
                message_type="info",
                message=(
                    "detected presence of contact, common_code tables in the database"
                    " using 'contact' table for physician_names."
                )
            )
            self.physician_names = """
                WITH physician_names AS (
                    SELECT DISTINCT
                        c.contact_id AS primaryphysicianid,
                        cc.item_description AS primaryphysiciantitle,
                        c.first_name AS primaryphysicianfirstname,
                        c.last_name AS primaryphysicianlastname
                    FROM contact c
                    LEFT JOIN common_code cc
                        ON cc.item_id = c.title_id
                    WHERE c.deleted = 'N'
                )
                """
        else:
            log_message(
                message_type="info",
                message=(
                    "Tables contact, common_code not present in the database"
                    " using 'view_ods_hospital_transfers_admission_log' view for physician_names."
                )
            )
            self.physician_names = """
                WITH physician_names AS (
                    SELECT DISTINCT
                        primaryphysicianid AS primaryphysicianid,
                        primaryphysiciantitle AS primaryphysiciantitle,
                        primaryphysicianfirstname AS primaryphysicianfirstname,
                        primaryphysicianlastname AS primaryphysicianlastname
                    FROM view_ods_hospital_transfers_admission_log
                )
                """

        # Use these attributes in the inherited classes to set specific queries for the given client
        # Use {key: None} if you want to exclude query
        self.client_specific_prediction_queries = dict()
        self.client_specific_training_queries = dict()
        self.include_mds_query = (engine is not None and check_table_exists(engine, 'view_ods_as_assessment'))

    def get_prediction_queries(self, *, prediction_date, facilityid, train_start_date):
        """
        :param prediction_date:
        :param facilityid:
        :param train_start_date:
        NOTE: patient_census includes only those rows whose actiontype is not in -
        TO(Transfer out of hospital), DE(Deceased Date), RDE (Respite - Deceased Date), RDD (Respite - Discharge Date),
        TP(Therapeutic Leave),L (Leave of Absence/LOA),H (Bed Hold),HI (Hospital Leave- ALL INS),
        TLU (Therapeutic Leave Unpaid), HMU (Hospital Leave Unpaid), HL(Hospital Leave),TL(Therapeutic Leave Medicare),
        PBH(Private Bed HolD), DRA(Discharge Return Anticipated),DRNA(Discharge Return Not Anticipated)
        :return: List of queries and a name which will be used as filename to save the result of query
        - 'patient_room_details' is used only during predictions for reports & ranking
        """
        invalid_actiontype = tuple(self.get_excluded_actiontype())

        MPIDs = f"""
                (SELECT t2.masterpatientid FROM
                (SELECT patientid FROM view_ods_patient_census
                    WHERE begineffectivedate < '{prediction_date}'
                        AND ((endeffectivedate IS NULL) OR (endeffectivedate >= '{prediction_date}'))
                        AND facilityid = {facilityid}) t1
                JOIN (SELECT DISTINCT patientid, masterpatientid FROM view_ods_facility_patient WHERE facilityid={facilityid} and
                ((patientdeleted is NULL or patientdeleted='N') and 
                (masterpatientdeleted  is NULL or masterpatientdeleted='N'))
                ) t2
                ON t1.patientid = t2.patientid)
                """

        queries = {
            'master_patient_lookup': f"""
                    {self.physician_names}
                    SELECT DISTINCT patientid, facilityid, masterpatientid, allergies, pn.primaryphysicianid,
                    pn.primaryphysiciantitle, pn.primaryphysicianfirstname, pn.primaryphysicianlastname  
                    FROM(
                        SELECT patientid, facilityid, masterpatientid, allergies, primaryphysicianid
                        FROM view_ods_facility_patient
                        WHERE masterpatientid IN {MPIDs}
                            and ((patientdeleted is NULL or patientdeleted='N')
                            and (masterpatientdeleted  is NULL or masterpatientdeleted='N'))) fp
                    LEFT JOIN physician_names pn
                    ON fp.primaryphysicianid = pn.primaryphysicianid
                    """,
            'patient_vitals': f"""
                    {self.custom_vitals_query_prefix}
                    SELECT b.patientid, b.facilityid, b.masterpatientid, date, vitalsdescription, value, 
                        diastolicvalue, warnings
                    FROM (SELECT clientid AS patientid, facilityid, date, vitalsdescription, value, 
                            diastolicvalue, warnings
                          FROM custom_vitals
                          WHERE date >= '{train_start_date}' AND date<='{prediction_date}'
                            AND strikeoutflag ='N'
                            AND vitalsdescription != 'Height'
                            AND vitalsdescription is not NULL) a
                          JOIN (SELECT DISTINCT patientid, facilityid, masterpatientid FROM view_ods_facility_patient) b
                          ON a.facilityid = b.facilityid AND a.patientid = b.patientid
                    WHERE b.masterpatientid IN {MPIDs} AND b.facilityid IN ({self.facilities})
                    """,
            'patient_admissions': f"""
                {self.physician_names}
                SELECT
                    b.masterpatientid,
                    a.patientid,
                    a.facilityid,
                    a.dateofadmission,
                    a.admissionstatus,
                    a.admittedfrom,
                    b.primaryphysicianid,
                    a.to_from_type,
                    b.primaryphysiciantitle,
                    b.primaryphysicianfirstname,
                    b.primaryphysicianlastname
                FROM (
                    SELECT
                        patientid,
                        facilityid,
                        dateofadmission,
                        admissionstatus,
                        admittedfrom,
                        primaryphysiciantitle,
                        primaryphysicianfirstname,
                        primaryphysicianlastname,
                        ToFromTypeDescription as to_from_type
                    FROM view_ods_hospital_transfers_admission_log
                ) a
                JOIN (
                    SELECT DISTINCT
                        patientid,
                        facilityid,
                        masterpatientid,
                        fp.primaryphysicianid,
                        pn.primaryphysiciantitle,
                        pn.primaryphysicianfirstname,
                        pn.primaryphysicianlastname
                    FROM view_ods_facility_patient fp
                    LEFT JOIN physician_names pn
                        ON fp.primaryphysicianid = pn.primaryphysicianid
                    WHERE masterpatientid IN {MPIDs} AND facilityid IN ({self.facilities})
                ) b
                ON a.facilityid = b.facilityid AND a.patientid = b.patientid
                """,
            'patient_census': f"""
                    SELECT t1.facilityid, t1.patientid, t1.masterpatientid, t1.begineffectivedate, t1.endeffectivedate,
                        t1.bedid, t3.beddesc as beddescription, t2.shortdescription as censusactioncode,
                        t4.payername, t4.payercode, t1.carelevelcode, t2.actiontype, t4.payertype
                    FROM(SELECT a.patientid, a.facilityid, b.masterpatientid, bedid, begineffectivedate, 
                            endeffectivedate, carelevelcode, actioncodeid, payerid, outpatientstatus
                        FROM(SELECT patientid, facilityid, bedid, begineffectivedate,endeffectivedate, 
                                    carelevelcode, actioncodeid, payerid, outpatientstatus
                                FROM view_ods_patient_census
                                WHERE Convert(date, begineffectivedate) <= '{prediction_date}') a
                                JOIN (SELECT DISTINCT patientid, facilityid, masterpatientid FROM view_ods_facility_patient) b
                                ON a.facilityid = b.facilityid AND a.patientid = b.patientid
                        WHERE b.masterpatientid IN {MPIDs}
                            AND actioncodeid IS NOT NULL
                            AND b.facilityid in ({self.facilities})) t1
                    LEFT JOIN (
                        SELECT itemid, shortdescription, actiontype
                        FROM view_ods_census_codes
                        WHERE codetype = 'ACT' AND deleted = 'N') t2
                        ON t1.actioncodeid = t2.itemid
                    LEFT JOIN view_ods_bed t3
                        ON t1.facilityid=t3.facilityid AND t1.bedid=t3.bedid
                    LEFT JOIN view_ods_payer t4
                        ON t1.payerid = t4.payerid
                    WHERE (outpatientstatus != 'A' OR outpatientstatus IS NULL)
                        AND (t4.payertype != 'Outpatient' OR t4.payertype IS NULL) 
                        """,   
            'patient_rehosps': f"""
                    SELECT a.patientid, a.facilityid, b.masterpatientid, dateoftransfer, purposeofstay, transferredto, 
                        orderedbyid, transferreason, otherreasonfortransfer, planned,hospitaldischargedate, 
                            primaryphysicianid
                    FROM (SELECT patientid, facilityid, dateoftransfer, purposeofstay, transferredto, orderedbyid, 
                            transferreason, otherreasonfortransfer, planned,hospitaldischargedate, primaryphysicianid 
                            from view_ods_hospital_transfers_transfer_log_v2) a
                        JOIN (SELECT DISTINCT patientid, facilityid, masterpatientid FROM view_ods_facility_patient) b
                        ON a.facilityid = b.facilityid AND a.patientid = b.patientid
                    WHERE b.masterpatientid IN {MPIDs} AND b.facilityid IN ({self.facilities})
                        """,
            'patient_demographics': f"""
                    SELECT masterpatientid, gender, dateofbirth, education, citizenship, race, religion,
                        state, primarylanguage, maritalstatus 
                    FROM view_ods_master_patient 
                    WHERE masterpatientid IN {MPIDs} AND deleted != 'Y'
                    """,
            'patient_diagnosis': f"""
                    SELECT b.patientid, b.facilityid, b.masterpatientid, onsetdate, diagnosiscode, diagnosisdesc, 
                        classification, rank, resolveddate, deleted, struckout, createddate
                    FROM (SELECT patientid, facilityid, onsetdate, diagnosiscode, diagnosisdesc, classification,
                            rank, resolveddate, deleted, struckout, createddate
                          FROM view_ods_patient_diagnosis
                          WHERE revisiondate >= '{train_start_date}' AND revisiondate <= '{prediction_date}'
                            AND onsetdate <= CURRENT_TIMESTAMP) a
                    JOIN (SELECT DISTINCT patientid, facilityid, masterpatientid FROM view_ods_facility_patient) b
                    ON a.facilityid = b.facilityid AND a.patientid = b.patientid
                    WHERE b.masterpatientid IN {MPIDs} AND b.facilityid IN ({self.facilities})
                    """,
            'patient_meds': f"""
                    SELECT d.patientid, d.facilityid, d.masterpatientid, orderdate, gpiclass, gpiclassdescription,
                        gpisubclassdescription, orderdescription, discontinueddate, MAREndDate, ordercreateddate, pharmacymedicationname
                    FROM (SELECT DISTINCT patientid, facilityid, orderdate, gpiclass, gpiclassdescription, 
                            gpisubclassdescription, orderdescription, a.discontinueddate, a.MAREndDate, ordercreateddate, pharmacymedicationname
                        FROM view_ods_physician_order_list_v2 a 
                        INNER JOIN view_ods_physician_order_list_med b
                        ON a.PhysicianOrderID = b.PhysiciansOrderID
                        WHERE orderdate >= '{train_start_date}' AND orderdate <= '{prediction_date}') c
                    JOIN (SELECT DISTINCT patientid, facilityid, masterpatientid FROM view_ods_facility_patient) d
                    ON c.facilityid = d.facilityid AND c.patientid = d.patientid
                    WHERE d.masterpatientid IN {MPIDs} AND d.facilityid IN ({self.facilities})
                    """,          
            'patient_orders': f"""
                    SELECT b.patientid, b.facilityid, b.masterpatientid, orderdate, ordercategory, ordertype, 
                        orderdescription, pharmacymedicationname, diettype, diettexture, dietsupplement, 
                        fluidconsistency, ordercreateddate, discontinueddate
                    FROM (SELECT DISTINCT patientid, facilityid, orderdate, ordercategory, ordertype, orderdescription, 
                                pharmacymedicationname, diettype, diettexture, dietsupplement, fluidconsistency, ordercreateddate, discontinueddate
                                from view_ods_physician_order_list_v2
                            WHERE ordercategory in ('Diagnostic', 'Enteral - Feeding', 'Dietary - Diet',
                                'Dietary - Supplements','Laboratory', 'Other')
                                AND orderdate >='{train_start_date}' AND orderdate <='{prediction_date}') a
                    JOIN (SELECT DISTINCT patientid, facilityid, masterpatientid FROM view_ods_facility_patient) b
                    ON a.facilityid = b.facilityid AND a.patientid = b.patientid
                    WHERE b.masterpatientid IN {MPIDs} AND b.facilityid IN ({self.facilities})
                    """, 
            'patient_alerts': f"""
                    SELECT b.patientid, b.facilityid, b.masterpatientid, createddate, stdalertid, alertdescription,
                        triggereditemtype, stdalerttypeid
                    FROM (SELECT patientid, facilityid, createddate, t1.stdalertid, alertdescription, triggereditemtype,
                             stdalerttypeid
                          FROM view_ods_cr_alert t1
                            LEFT JOIN (SELECT stdalertid, stdalerttypeid FROM view_ods_cr_std_alert) t2
                            ON t1.stdalertid=t2.stdalertid 
                          WHERE createddate >= '{train_start_date}' AND createddate <='{prediction_date}'
                            AND deleted = 'N'
                            AND alertdescription is NOT NULL) a
                    JOIN (SELECT DISTINCT patientid, facilityid, masterpatientid FROM view_ods_facility_patient) b
                    ON a.facilityid = b.facilityid AND a.patientid = b.patientid
                    WHERE b.masterpatientid IN {MPIDs} AND b.facilityid IN ({self.facilities})
                    """, 
            'patient_lab_results': f"""
                    SELECT g.patientid, g.facilityid, g.masterpatientid, resultdate, profiledescription, referencerange,
                            result, LabReportID, severityid, MasterLabReportID, VersionNumber, LabTestID,
                            units, AbnormalityCode, abnormalityid, abnormalitydescription, reportdesciption,
                            severitydescription, createddate 
                    FROM (SELECT c.patientid, c.facilityid, a.resultdate, a.profiledescription, a.referencerange,
                            a.result, a.LabReportID, b.severityid,b.MasterLabReportID, b.VersionNumber, a.LabTestID,
                            a.units, e.AbnormalityCode, a.abnormalityid, e.abnormalitydescription, b.reportdesciption,
                            d.severitydescription, b.createddate
                          FROM view_ods_result_lab_report_detail a
                          LEFT JOIN view_ods_result_lab_report b ON a.LabReportID = b.LabReportID
                          LEFT JOIN view_ods_result_order_source c ON b.ResultOrderSourceID = c.ResultOrderSourceID
                          LEFT JOIN view_ods_result_lab_report_severity d ON b.SeverityID = d.SeverityID
                          LEFT JOIN view_ods_result_lab_test_abnormality e ON a.AbnormalityID = e.AbnormalityID
                          WHERE a.resultdate >= '{train_start_date}' AND a.resultdate <= '{prediction_date}') f
                    JOIN (SELECT DISTINCT patientid, facilityid, masterpatientid FROM view_ods_facility_patient) g
                    ON f.facilityid = g.facilityid AND f.patientid = g.patientid
                    WHERE g.masterpatientid IN {MPIDs} AND g.facilityid IN ({self.facilities})
                    """, 
            'patient_progress_notes': f"""
                    SELECT b.patientid, b.facilityid, b.masterpatientid, progressnoteid, progressnotetype, 
                        createddate, effectivedate, sectionsequence, section, notetextorder, notetext, highrisk, showon24hr, 
                        showonshift
                    FROM (SELECT patientid, facilityid, progressnoteid, progressnotetype, createddate, effectivedate,
                            sectionsequence, section, notetextorder, notetext, highrisk, showon24hr, showonshift
                          FROM view_ods_progress_note
                          WHERE createddate >= '{train_start_date}' AND createddate <='{prediction_date}') a
                    JOIN (SELECT DISTINCT patientid, facilityid, masterpatientid FROM view_ods_facility_patient) b
                    ON a.facilityid = b.facilityid AND a.patientid = b.patientid
                    WHERE b.masterpatientid IN {MPIDs} AND b.facilityid IN ({self.facilities})
                    """,
            'patient_room_details': f"""
                    SELECT c.patientid AS patientid,
                           fp.masterpatientid,
                           c.facilityid,
                           '{prediction_date}' AS censusdate,
                           fp.initialadmissiondate,
                           p.payername,
                           r.RoomDescription AS room,
                           r.RoomID AS room_id,
                           r.Floor AS floor,
                           r.FloorID AS floor_id,
                           u.UnitDescription AS unit,
                           u.UnitID AS unit_id
                    FROM (
                        SELECT patientid,
                               facilityid, 
                               bedid, 
                               payerid
                        FROM view_ods_patient_census 
                        WHERE begineffectivedate < '{prediction_date}'
                          AND ((endeffectivedate IS NULL) OR (Convert(date, endeffectivedate) >= '{prediction_date}'))
                    ) c
                    JOIN (
                        SELECT DISTINCT facilityid,
                                        patientid,
                                        masterpatientid,
                                        initialadmissiondate
                        FROM view_ods_facility_patient
                        WHERE ((patientdeleted is NULL or patientdeleted='N') and 
                        (masterpatientdeleted  is NULL or masterpatientdeleted='N'))
                    ) fp
                      ON fp.facilityid = c.facilityid AND fp.patientid = c.patientid
                    JOIN view_ods_bed b
                      ON b.BedID = c.bedid
                    JOIN view_ods_room r
                      ON r.RoomID = b.RoomID
                    JOIN dbo.view_ods_unit u
                      ON u.UnitID = b.UnitID
                    LEFT JOIN view_ods_payer p
                      ON c.payerid = p.payerid
                    WHERE c.facilityid = {facilityid}
                    """,
            'patient_immunizations': f"""
                    SELECT b.patientid, b.facilityid, b.masterpatientid, immunizationdate, immunizationdesc, createddate
                    FROM (SELECT facilityid, clientid AS patientid,immunizationdate, immunizationdesc, createddate
                          FROM view_ods_patient_immunizations
                          WHERE immunizationdate >= '{train_start_date}' AND immunizationdate <='{prediction_date}'
                            AND consent not in ('Not Eligible', 'TBD' , 'Refused')
                            AND deleted = 'N') a
                          JOIN (SELECT DISTINCT patientid, facilityid, masterpatientid FROM view_ods_facility_patient) b
                          ON a.facilityid = b.facilityid AND a.patientid = b.patientid
                    WHERE b.masterpatientid IN {MPIDs} AND b.facilityid IN ({self.facilities})
                    """,
            'patient_risks': f"""
                    SELECT b.patientid, b.facilityid, b.masterpatientid, incidentdate, description, typeid, createddate
                    FROM (SELECT a.facilityid, patientid, incidentdate, b.description, typeid, a.createddate
                        FROM view_ods_inc_incident a
                        LEFT join view_ods_inc_std_pick_list_item b
                        ON a.typeid = b.picklistitemid
                        WHERE isdeleted='N' 
                            AND incidentdate >= '{train_start_date}' AND incidentdate <= '{prediction_date}') a
                    JOIN (SELECT DISTINCT patientid, facilityid, masterpatientid FROM view_ods_facility_patient) b
                    ON a.facilityid = b.facilityid AND a.patientid = b.patientid
                    WHERE b.masterpatientid IN {MPIDs} AND b.facilityid IN ({self.facilities})
                    """,
            'patient_assessments':f"""
                    SELECT c.patientid, c.facilityid, c.masterpatientid, assessmentdate, description, createddate
                    FROM (SELECT b.facilityid, b.patientid, b.assessmentdate, a.description, b.createddate
                          FROM view_ods_std_assessment a
                          LEFT JOIN view_ods_assessment b on a.stdassessid = b.stdassessid
                          WHERE a.Status = 'A' AND a.deleted = 'N'
                             AND b.patientid != -9999
                             AND b.deleted = 'N'
                             AND b.assessmentdate >= '{train_start_date}' AND b.assessmentdate <= '{prediction_date}') b
                          JOIN (SELECT DISTINCT patientid, facilityid, masterpatientid FROM view_ods_facility_patient) c
                          ON b.facilityid = c.facilityid AND b.patientid = c.patientid
                    WHERE c.masterpatientid IN {MPIDs} AND c.facilityid IN ({self.facilities})
                    """,            
            'patient_adt':f"""
                    SELECT b.patientid, b.facilityid, b.masterpatientid, actiontype, outcome,
                            longdescriptionaction, admitdischargelocationtype, begineffectivedate,
                            endeffectivedate, transferreason, dateoftransfer, planned
                    FROM (SELECT facilityid, patientid, act.actiontype, hta.outcome,
                                act.longdescriptionaction, ein.admitdischargelocationtype, ein.begineffectivedate,
                                ein.endeffectivedate, hta.transferreason,  hta.dateoftransfer,  hta.planned
                        FROM view_ods_patient_census ein
                        LEFT JOIN
                            (SELECT LongDescription AS LongDescriptionAction, ActionType,
                                ShortDescription AS ShortDescriptionAction, ItemID 
                                FROM view_ods_census_codes) act
                        ON act.ItemID = ein.ActionCodeID
                        LEFT JOIN
                            (SELECT LongDescription AS LongDescriptionStatus, ItemID,
                                ShortDescription AS ShortDescriptionStatus
                                FROM view_ods_census_codes) sta
                        ON sta.ItemID = ein.StatusCodeID
                        LEFT JOIN
                            (SELECT DISTINCT MasterPatientID, FacilityID AS FacilityIDFac, PatientID AS PatientIDFac
                                FROM view_ods_facility_patient) as fac
                        ON fac.PatientIDFac = ein.PatientID
                            AND fac.FacilityIDFac = ein.FacilityID
                        LEFT JOIN
                            (SELECT FacilityID as FacilityIDhtt, PatientID as PatientIDhtt, CensusID as CensusIDhatt
                                FROM view_ods_hospital_transfers_admission_log) htt
                        ON htt.FacilityIDhtt = ein.FacilityID
                            AND htt.PatientIDhtt = ein.PatientID
                            AND htt.CensusIDhatt = ein.CensusID
                        LEFT JOIN
                            (SELECT FacilityID as FacilityIDhta, PatientID as PatientIDhta, CensusID as CensusIDhata,
                                dateoftransfer, planned, TransferReason, Outcome
                                FROM view_ods_hospital_transfers_transfer_log_v2) hta
                        ON hta.FacilityIDhta = ein.FacilityID
                            AND hta.PatientIDhta = ein.PatientID
                            AND hta.CensusIDhata = ein.CensusID
                        WHERE act.actiontype != 'Internal Transfer') a
                    JOIN (SELECT DISTINCT patientid, facilityid, masterpatientid FROM view_ods_facility_patient) b
                    ON a.facilityid = b.facilityid AND a.patientid = b.patientid
                    WHERE b.masterpatientid IN {MPIDs} AND b.facilityid IN ({self.facilities})
                    """,
        }
        if self.include_mds_query:
            queries['patient_mds'] = f"""
                SELECT
                    b.patientid,
                    b.facilityid,
                    b.masterpatientid,
                    voaa.AssessmentID,
                    voaa.AssessTypeCode AS AssessmentTypeKey,
                    voaa.AssessDate AS AssessmentDate,
                    voaa.AdlScore,
                    voaa.LockedDate,
                    voaa.IncorrectAssessmentID,
                    voaa.CmiFed AS MedicareCMI,
                    voaa.CmiCodeFed AS MedicareRUG
                FROM view_ods_as_assessment voaa
                JOIN (
                    SELECT DISTINCT patientid, facilityid, masterpatientid FROM view_ods_facility_patient
                    WHERE masterpatientid IN {MPIDs} AND facilityid IN ({self.facilities})
                ) b
                ON voaa.facilityid = b.facilityid AND voaa.ResidentID = b.patientid
                WHERE voaa.IsDeleted = 'N'
                AND voaa.AssessDate >= DATEADD(DAY, -100, '{train_start_date}')
                AND voaa.AssessDate <= '{prediction_date}'
                AND voaa.StdAssessmentID = 11
                AND voaa.LockedDate IS NOT NULL
                """

        queries.update(self.client_specific_prediction_queries)
        return queries

    def get_training_queries(self, *, test_end_date, train_start_date):
        """
        Training queries are not tied up to one facility and prediction date.
        So the queries are different from prediction flow
        :param test_end_date:
        :param train_start_date:
        :return: List of queries and a name which will be used as filename to save the result of query
        """
        invalid_actiontype = tuple(self.get_excluded_actiontype())
        
        PIDs = f"""
                SELECT DISTINCT patientid
                FROM (SELECT facilityid ,patientid, actioncodeid, begineffectivedate, endeffectivedate 
                    FROM view_ods_patient_census) a
                LEFT JOIN (SELECT itemid, shortdescription, actiontype 
                    FROM view_ods_census_codes WHERE codetype = 'ACT' AND deleted = 'N') b
                ON a.actioncodeid = b.itemid
                WHERE ((Convert(date, begineffectivedate) <= '{test_end_date}' 
                            AND Convert(date, endeffectivedate) > '{train_start_date}' 
                            AND Convert(date, begineffectivedate) < Convert(date, endeffectivedate))
                        OR (Convert(date, begineffectivedate) <= '{test_end_date}'  
                            AND endeffectivedate IS NULL))
                    AND facilityid IN ({self.facilities})
                    AND actioncodeid IS NOT NULL 
                    AND actiontype NOT IN {invalid_actiontype}
                """

        queries = {
            'master_patient_lookup': f'''
                SELECT DISTINCT patientid, facilityid, masterpatientid, allergies
                FROM view_ods_facility_patient
                WHERE facilityid IN ({self.facilities}) and ((patientdeleted is NULL or patientdeleted='N') and 
                (masterpatientdeleted  is NULL or masterpatientdeleted='N'))
                ''',
            'patient_census': f"""
                SELECT t1.patientid, t1.begineffectivedate, t1.endeffectivedate, t1.createddate,
                    t1.facilityid, t1.bedid, t3.beddesc as beddescription,
                    t2.shortdescription as censusactioncode, t4.payername,
                    t4.payercode, t1.carelevelcode, t2.actiontype, t4.payertype
                FROM (
                    SELECT patientid, facilityid, bedid, begineffectivedate,endeffectivedate, 
                        createddate, carelevelcode, actioncodeid, payerid, outpatientstatus
                    FROM view_ods_patient_census
                    WHERE Convert(date, begineffectivedate) <= '{test_end_date}' 
                        AND facilityid IN ({self.facilities})
                        AND actioncodeid IS NOT NULL) t1
                LEFT JOIN (
                    SELECT itemid, shortdescription, actiontype
                    FROM view_ods_census_codes
                    WHERE codetype = 'ACT' AND deleted = 'N') t2
                    ON t1.actioncodeid = t2.itemid
                LEFT JOIN view_ods_bed t3
                    ON t1.facilityid=t3.facilityid AND t1.bedid=t3.bedid
                LEFT JOIN view_ods_payer t4
                    ON t1.payerid = t4.payerid
                WHERE (outpatientstatus != 'A' OR outpatientstatus IS NULL) 
                    AND (t4.payertype != 'Outpatient' OR t4.payertype IS NULL) 
                """,
            'patient_rehosps': f"""
                        select patientid, facilityid, dateoftransfer, purposeofstay, transferredto,
                        orderedbyid, transferreason, otherreasonfortransfer, planned, lengthofstay, payerdescription,
                        hospitaldischargedate, primaryphysicianid from view_ods_hospital_transfers_transfer_log_v2
                        where facilityid in ({self.facilities})
                        """,
            'patient_admissions': f"""
                        {self.physician_names}
                        select a.patientid, a.facilityid, a.dateofadmission, a.admissionstatus, a.admittedfrom, 
                        b.primaryphysicianid, b.primaryphysiciantitle, b.primaryphysicianfirstname, 
                        b.primaryphysicianlastname, a.tofromtypedescription as to_from_type
                        from view_ods_hospital_transfers_admission_log a
                        join (SELECT DISTINCT patientid, facilityid, masterpatientid, fp.primaryphysicianid,
                                    pn.primaryphysiciantitle, pn.primaryphysicianfirstname, 
                                    pn.primaryphysicianlastname
                                FROM view_ods_facility_patient fp
                                LEFT JOIN physician_names pn
                                ON fp.primaryphysicianid = pn.primaryphysicianid) b 
                        on a.facilityid = b.facilityid and a.patientid = b.patientid
                        where a.facilityid in ({self.facilities})
                        """,
            'patient_demographics': f"""
                        select masterpatientid, gender, dateofbirth, education, citizenship,race, religion, 
                            state, primarylanguage, maritalstatus 
                        from view_ods_master_patient
                        where deleted != 'Y'
                        """,
            'patient_diagnosis': f"""
                        select patientid, onsetdate, facilityid, diagnosiscode, diagnosisdesc, classification,
                            rank, resolveddate, createddate
                        from view_ods_patient_diagnosis where revisiondate between '{train_start_date}' and '{test_end_date}'
                        and facilityid in ({self.facilities})
                        and onsetdate <= current_timestamp
                        """,
            'patient_vitals': f"""
                        {self.custom_vitals_query_prefix}
                        select clientid as patientid, facilityid, date, vitalsdescription, value, diastolicvalue, warnings
                        from custom_vitals where date between '{train_start_date}' and '{test_end_date}'
                        and facilityid in ({self.facilities})
                        and clientid in ({PIDs})
                        and strikeoutflag ='N'
                        and vitalsdescription != 'Height'
                        and vitalsdescription is not NULL
                        """,
            'patient_meds': f"""
                        select distinct patientid, facilityid, orderdate, gpiclass, gpiclassdescription,
                        gpisubclassdescription, orderdescription, pharmacymedicationname,
                        PhysicianOrderID, a.discontinueddate, a.MAREndDate, ordercreateddate
                        from view_ods_physician_order_list_v2 a inner join view_ods_physician_order_list_med b
                        on a.PhysicianOrderID = b.PhysiciansOrderID
                        where orderdate between '{train_start_date}' and '{test_end_date}'
                        and facilityid in ({self.facilities})
                        """,
            'patient_orders': f"""
                        select distinct patientid, facilityid, orderdate, ordercategory, fluidconsistency,
                        ordertype, orderdescription, pharmacymedicationname, diettype, diettexture, dietsupplement, ordercreateddate
                        from view_ods_physician_order_list_v2
                        where orderdate between '{train_start_date}' and '{test_end_date}'
                        and facilityid in ({self.facilities})
                        and ordercategory in ('Diagnostic', 'Enteral - Feeding', 'Dietary - Diet',
                        'Dietary - Supplements','Laboratory', 'Other')
                        """,
            'patient_alerts': f"""
                        SELECT patientid, facilityid, createddate, a.stdalertid, alertdescription,triggereditemtype, 
                            stdalerttypeid, stdalerttypedescription
                        FROM view_ods_cr_alert a
                        LEFT JOIN (SELECT stdalertid, stdalerttypeid, stdalerttypedescription 
                            FROM view_ods_cr_std_alert) b
                        ON a.stdalertid=b.stdalertid 
                        WHERE createddate >= '{train_start_date}' AND createddate <= '{test_end_date}'
                        AND facilityid in ({self.facilities})
                        AND deleted = 'N'
                        AND alertdescription is NOT NULL
                        """,
            'patient_lab_results': f'''
                        select c.patientid, c.facilityid, a.resultdate, a.profiledescription,
                        a.referencerange, a.result, a.LabReportID,b.MasterLabReportID, b.VersionNumber, a.LabTestID,
                        a.units, e.AbnormalityCode, a.abnormalityid, e.abnormalitydescription, b.reportdesciption,
                        b.severityid, d.severitydescription, b.createddate
                        from view_ods_result_lab_report_detail a
                        left join view_ods_result_lab_report b on a.LabReportID = b.LabReportID
                        left join view_ods_result_order_source c on b.ResultOrderSourceID = c.ResultOrderSourceID
                        left join view_ods_result_lab_report_severity d on b.SeverityID = d.SeverityID
                        left join view_ods_result_lab_test_abnormality e on a.AbnormalityID = e.AbnormalityID
                        WHERE a.resultdate BETWEEN '{train_start_date}' AND '{test_end_date}'
                        and c.facilityid in ({self.facilities})
                        ''',
            'patient_progress_notes': f"""
                        select patientid, facilityid, progressnoteid, progressnotetype, createddate, effectivedate, sectionsequence,
                        section, notetextorder, notetext, highrisk, showon24hr, showonshift
                        from view_ods_progress_note
                        where createddate between '{train_start_date}' and '{test_end_date}'
                        and facilityid in ({self.facilities})
                        """,
            'patient_immunizations': f"""
                        select facilityid, clientid AS patientid,immunizationdate, immunizationdesc, createddate
                        from view_ods_patient_immunizations
                        where immunizationdate between '{train_start_date}' and '{test_end_date}'
                            and facilityid in ({self.facilities})
                            and consent not in ('Not Eligible', 'TBD' , 'Refused')
                            and deleted = 'N'
                        """,
            'patient_risks': f"""
                        select a.facilityid, patientid, incidentdate, a.createddate, b.description, typeid
                        from view_ods_inc_incident a
                        left join view_ods_inc_std_pick_list_item b
                        on a.typeid = b.picklistitemid
                        where isdeleted='N' and incidentdate between '{train_start_date}' and '{test_end_date}'
                             and a.facilityid in ({self.facilities})
                        """,
            'patient_assessments':f"""
                        select b.facilityid, b.patientid, b.assessmentdate, a.description, b.createddate
                        from view_ods_std_assessment a
                        left join view_ods_assessment b on a.stdassessid = b.stdassessid
                        where a.Status = 'A' and a.deleted = 'N'
                            and b.patientid != -9999
                            and b.deleted = 'N'
                            and b.assessmentdate between '{train_start_date}' and '{test_end_date}'
                            and b.facilityid in ({self.facilities})
                        """,
            'patient_adt':f"""
                        SELECT facilityid, patientid, act.actiontype, hta.outcome, act.longdescriptionaction,
                            ein.admitdischargelocationtype, ein.begineffectivedate, ein.endeffectivedate, 
                            hta.dateoftransfer, hta.planned, hta.transferreason
                        FROM view_ods_patient_census ein
                        LEFT JOIN
                            (SELECT LongDescription AS LongDescriptionAction, ActionType,
                                ShortDescription AS ShortDescriptionAction, ItemID
                              FROM view_ods_census_codes) act
                        ON act.ItemID = ein.ActionCodeID
                        LEFT JOIN
                            (SELECT LongDescription AS LongDescriptionStatus, ItemID,
                                ShortDescription AS ShortDescriptionStatus
                               FROM view_ods_census_codes) sta
                        ON sta.ItemID = ein.StatusCodeID
                        LEFT JOIN
                            (SELECT DISTINCT MasterPatientID, FacilityID AS FacilityIDFac, PatientID AS PatientIDFac
                               FROM view_ods_facility_patient
                               WHERE ((patientdeleted is NULL or patientdeleted='N') and 
                               (masterpatientdeleted  is NULL or masterpatientdeleted='N'))) as fac
                        ON fac.PatientIDFac = ein.PatientID
                            AND fac.FacilityIDFac = ein.FacilityID
                        LEFT JOIN
                            (SELECT FacilityID as FacilityIDhtt, PatientID as PatientIDhtt, CensusID as CensusIDhatt
                               FROM view_ods_hospital_transfers_admission_log) htt
                        ON htt.FacilityIDhtt = ein.FacilityID
                            AND htt.PatientIDhtt = ein.PatientID
                            AND htt.CensusIDhatt = ein.CensusID
                        LEFT JOIN
                            (SELECT FacilityID as FacilityIDhta, PatientID as PatientIDhta, CensusID as CensusIDhata,
                               dateoftransfer, planned, TransferReason, Outcome
                               FROM view_ods_hospital_transfers_transfer_log_v2) hta
                        ON hta.FacilityIDhta = ein.FacilityID
                            AND hta.PatientIDhta = ein.PatientID
                            AND hta.CensusIDhata = ein.CensusID
                        WHERE ein.FacilityID in ({self.facilities})
                            AND act.actiontype != 'Internal Transfer'
                        """,
        }
        if self.include_mds_query:
            queries['patient_mds'] = f"""
                SELECT 
                    voaa.assessmentid,
                    voaa.facilityid,
                    voaa.ResidentID AS patientid,
                    voaa.AssessTypeCode AS assessmenttypekey,
                    voaa.AssessDate AS assessmentdate,
                    voaa.adlscore,
                    voaa.lockeddate,
                    voaa.incorrectassessmentid,
                    voaa.CmiFed AS medicarecmi,
                    voaa.CmiCodeFed AS medicarerug
                FROM view_ods_as_assessment voaa
                WHERE voaa.IsDeleted = 'N'
                AND voaa.AssessDate >= DATEADD(DAY, -100, '{train_start_date}')
                AND voaa.AssessDate <= '{test_end_date}'
                AND voaa.StdAssessmentID = 11
                AND voaa.LockedDate IS NOT NULL
                """

        queries.update(self.client_specific_training_queries)
        return queries

    def client_specific_transformations(self, result_dict):
        return result_dict

    def get_note_embeddings_valid_section(self):
        return []

    def get_note_embeddings_emar_types(self):
        return []

    def get_note_embeddings_nan_threshold(self):
        return self.note_embeddings_nan_threshold

    def get_hospice_payercodes(self):
        return []

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
