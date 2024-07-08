CREATE MATERIALIZED VIEW public.predictions_performance_upt_all_los AS (
    SELECT
        facilityid,
        patientid,
        patientfirstname,
        patientlastname,
        dateoftransfer,
        purposeofstay,
        transferredto,
        transferreason,
        otherreasonfortransfer,
        planned,
        outcome,
        medicalrecordnumber,
        tofromtype,
        payertype,
        payerdescription,
        primaryphysiciantitle,
        primaryphysicianfirstname,
        primaryphysicianlastname,
        primaryphysicianprofession,
        originaladmissiondate,
        lastadmissiondate,
        hospitaldischargedate,
        lengthofstay,
        transferredwithin30daysofadmission,
        client,
        facilityname,
        masterpatientid,
        modelid,
        group_rank,
        group_id,
        show_in_report,
        censusdate,
        ml_model_org_config_id,
        patientmrn,
        firstname,
        lastname
    FROM
        predictions_performance_upt_all_los_unfiltered
    WHERE
        is_active = TRUE
        AND payertype IS DISTINCT FROM 'Outpatient'
        AND (published IS NULL OR published = TRUE)
        AND (experiment_group IS NULL OR experiment_group = TRUE)
        AND (lengthofstay IS NULL OR lengthofstay > 0 ::double PRECISION)
);

CREATE INDEX predictions_performance_upt_all_los_client_org_config_id_idx 
    ON public.predictions_performance_upt_all_los (
        client,
        ml_model_org_config_id
    );

CREATE INDEX predictions_performance_upt_all_los_facility_idx 
    ON public.predictions_performance_upt_all_los (
        facilityid
    );

CREATE INDEX predictions_performance_upt_all_los_dateoftransfer_idx 
    ON public.predictions_performance_upt_all_los (
        dateoftransfer
    );

CREATE INDEX predictions_performance_upt_all_los_pred_perf_report_idx 
    ON public.predictions_performance_upt_all_los (
        client,
        ml_model_org_config_id,
        facilityid,
        dateoftransfer
    );

CREATE INDEX predictions_performance_upt_all_los_hit_miss_report_idx 
    ON public.predictions_performance_upt_all_los (
        client,
        ml_model_org_config_id,
        facilityid,
        dateoftransfer,
        masterpatientid
    );
