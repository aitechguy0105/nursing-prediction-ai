-- View: public.daily_prediction_performance

-- DROP MATERIALIZED VIEW public.daily_prediction_performance;

CREATE MATERIALIZED VIEW public.daily_prediction_performance
TABLESPACE pg_default
AS SELECT a.client,
    a.masterpatientid,
    a.facilityid,
    a.bedid,
    a.censusdate,
    a.predictionvalue,
    a.predictionrank,
    a.modelid,
    b.dayspredictionvalid,
    min(
        CASE
            WHEN t.dateoftransfer >= a.censusdate THEN t.dateoftransfer
            ELSE NULL::timestamp without time zone
        END) AS dateoftransfer
    FROM daily_predictions a
     LEFT JOIN model_metadata b ON a.modelid = b.modelid
     LEFT JOIN ( SELECT DISTINCT facility_patient.masterpatientid,
            facility_patient.facilityid,
            facility_patient.patientid,
            facility_patient.client
           FROM facility_patient) c ON a.masterpatientid = c.masterpatientid AND a.facilityid = c.facilityid AND a.client::text = c.client::text
     LEFT JOIN ( SELECT fp.masterpatientid,
            ht.dateoftransfer,
            fp.facilityid,
            fp.client
           FROM hospital_transfers ht
             JOIN facility_patient fp ON ht.patientid = fp.patientid AND ht.facilityid = fp.facilityid AND ht.client::text = fp.client::text) t ON c.masterpatientid = t.masterpatientid AND c.facilityid = t.facilityid AND c.client::text = t.client::text
  GROUP BY a.masterpatientid, a.facilityid, a.bedid, a.censusdate, a.predictionvalue, a.predictionrank, a.modelid, b.dayspredictionvalid, a.client
WITH DATA;

-- Permissions

ALTER TABLE public.daily_prediction_performance OWNER TO saivaadmin;
GRANT ALL ON TABLE public.daily_prediction_performance TO saivaadmin;
GRANT SELECT ON TABLE public.daily_prediction_performance TO quicksight_ro;
