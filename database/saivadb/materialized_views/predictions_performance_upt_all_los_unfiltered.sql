CREATE MATERIALIZED VIEW public.predictions_performance_upt_all_los_unfiltered AS (
    WITH deaths AS (
        SELECT DISTINCT 
            ed.facility_id AS facilityid,
            ed.patient_id AS patientid,
            ed.patient_first_name AS patientfirstname,
            ed.patient_last_name AS patientlastname,
            ed.event_date AS dateoftransfer,
            NULL::text AS purposeofstay,
            NULL::text AS transferredto,
            'Death'::text AS transferreason,
            NULL::text AS otherreasonfortransfer,
            NULL::text AS planned,
            'Death'::text AS outcome,
            ed.medical_record_number AS medicalrecordnumber,
            NULL::text AS tofromtype,
            ed.payer_type AS payertype,
            ed.payer_description AS payerdescription,
            p.title AS primaryphysiciantitle,
            p.first_name AS primaryphysicianfirstname,
            p.last_name AS primaryphysicianlastname,
            NULL::text AS primaryphysicianprofession,
            ed.original_admission_date AS originaladmissiondate,
            ed.last_admission_date AS lastadmissiondate,
            NULL::timestamp without time zone AS hospitaldischargedate,
            ed.length_of_stay AS lengthofstay,
                CASE
                    WHEN ed.length_of_stay <= 30 THEN 1
                    ELSE 0
                END AS transferredwithin30daysofadmission,
            ed.client
        FROM event_death ed
            LEFT JOIN physician p ON ed.client::text = p.client::text AND ed.primary_physician_id = p.physician_id
        WHERE ed.on_premise AND ed.event_date >= '2020-01-01 00:00:00'::timestamp without time zone AND (lower(ed.payer_description::text) !~~ '%hospice%'::text OR ed.payer_description IS NULL)
    ), rehosps AS (
        SELECT DISTINCT 
            hospital_transfers.facilityid,
            hospital_transfers.patientid,
            hospital_transfers.patientfirstname,
            hospital_transfers.patientlastname,
            hospital_transfers.dateoftransfer,
            hospital_transfers.purposeofstay,
            hospital_transfers.transferredto,
            hospital_transfers.transferreason,
            hospital_transfers.otherreasonfortransfer,
            hospital_transfers.planned,
            hospital_transfers.outcome,
            hospital_transfers.medicalrecordnumber,
            hospital_transfers.tofromtype,
            hospital_transfers.payertype,
            hospital_transfers.payerdescription,
            hospital_transfers.primaryphysiciantitle,
            hospital_transfers.primaryphysicianfirstname,
            hospital_transfers.primaryphysicianlastname,
            hospital_transfers.primaryphysicianprofession,
            hospital_transfers.originaladmissiondate,
            hospital_transfers.lastadmissiondate,
            hospital_transfers.hospitaldischargedate,
            hospital_transfers.lengthofstay,
            hospital_transfers.transferredwithin30daysofadmission,
            hospital_transfers.client
        FROM hospital_transfers
        WHERE hospital_transfers.dateoftransfer >= '2020-01-01 00:00:00'::timestamp without time zone AND (lower(hospital_transfers.payerdescription) !~~ '%hospice%'::text OR hospital_transfers.payerdescription IS NULL)
    ), upt AS (
        SELECT 
            deaths.facilityid,
            deaths.patientid,
            deaths.patientfirstname,
            deaths.patientlastname,
            deaths.dateoftransfer,
            deaths.purposeofstay,
            deaths.transferredto,
            deaths.transferreason,
            deaths.otherreasonfortransfer,
            deaths.planned,
            deaths.outcome,
            deaths.medicalrecordnumber,
            deaths.tofromtype,
            deaths.payertype,
            deaths.payerdescription,
            deaths.primaryphysiciantitle,
            deaths.primaryphysicianfirstname,
            deaths.primaryphysicianlastname,
            deaths.primaryphysicianprofession,
            deaths.originaladmissiondate,
            deaths.lastadmissiondate,
            deaths.hospitaldischargedate,
            deaths.lengthofstay,
            deaths.transferredwithin30daysofadmission,
            deaths.client
        FROM deaths
    UNION
        SELECT 
            rehosps.facilityid,
            rehosps.patientid,
            rehosps.patientfirstname,
            rehosps.patientlastname,
            rehosps.dateoftransfer,
            rehosps.purposeofstay,
            rehosps.transferredto,
            rehosps.transferreason,
            rehosps.otherreasonfortransfer,
            rehosps.planned,
            rehosps.outcome,
            rehosps.medicalrecordnumber,
            rehosps.tofromtype,
            rehosps.payertype,
            rehosps.payerdescription,
            rehosps.primaryphysiciantitle,
            rehosps.primaryphysicianfirstname,
            rehosps.primaryphysicianlastname,
            rehosps.primaryphysicianprofession,
            rehosps.originaladmissiondate,
            rehosps.lastadmissiondate,
            rehosps.hospitaldischargedate,
            rehosps.lengthofstay,
            rehosps.transferredwithin30daysofadmission,
            rehosps.client
        FROM rehosps
    )
    SELECT DISTINCT 
        upt.facilityid,
        upt.patientid,
        upt.patientfirstname,
        upt.patientlastname,
        upt.dateoftransfer,
        upt.purposeofstay,
        upt.transferredto,
        upt.transferreason,
        upt.otherreasonfortransfer,
        upt.planned,
        upt.outcome,
        upt.medicalrecordnumber,
        upt.tofromtype,
        upt.payertype,
        upt.payerdescription,
        upt.primaryphysiciantitle,
        upt.primaryphysicianfirstname,
        upt.primaryphysicianlastname,
        upt.primaryphysicianprofession,
        upt.originaladmissiondate,
        upt.lastadmissiondate,
        upt.hospitaldischargedate,
        upt.lengthofstay,
        upt.transferredwithin30daysofadmission,
        upt.client,
        fa.facilityname,
        fp.masterpatientid,
        dp.modelid,
        dp.group_rank,
        dp.group_id,
        dp.show_in_report,
        dp.censusdate,
        dp.ml_model_org_config_id,
        fp.patientmrn,
        fp.firstname,
        fp.lastname,
        dp.published,
        dp.experiment_group,
        fa.is_active
    FROM
        upt
    LEFT JOIN facility_patient fp 
        ON upt.client::text = fp.client::text
        AND upt.facilityid = fp.facilityid
        AND upt.patientid = fp.patientid
    LEFT JOIN daily_predictions dp 
        ON upt.client::text = dp.client::text
        AND upt.facilityid = dp.facilityid
        AND(date(upt.dateoftransfer) - date(dp.censusdate)) <= 3
        AND date(dp.censusdate) <= date(upt.dateoftransfer)
        AND fp.masterpatientid = dp.masterpatientid
        AND fp.ml_model_org_config_id = dp.ml_model_org_config_id
    LEFT JOIN facility fa 
        ON fa.facilityid = upt.facilityid
        AND fa.client::text = upt.client::text
        AND fa.ml_model_org_config_id = dp.ml_model_org_config_id
);