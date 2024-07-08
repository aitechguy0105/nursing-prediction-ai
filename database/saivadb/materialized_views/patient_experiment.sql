-- Not in use

CREATE MATERIALIZED VIEW public.patient_experiment
TABLESPACE pg_default
AS SELECT p.client,
    p.facilityid,
    fp.patientid,
    bool_and(p.experiment_group) AS experiment_group
   FROM daily_predictions p
     JOIN facility_patient fp ON p.facilityid = fp.facilityid AND p.masterpatientid = fp.masterpatientid AND p.client::text = fp.client::text
  GROUP BY p.client, p.facilityid, fp.patientid
WITH DATA;

-- Permissions

ALTER TABLE public.patient_experiment OWNER TO saivaadmin;
GRANT ALL ON TABLE public.patient_experiment TO saivaadmin;
GRANT SELECT ON TABLE public.patient_experiment TO quicksight_ro;
