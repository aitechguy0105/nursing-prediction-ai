-- Used for AWS QuickSight

CREATE OR REPLACE VIEW public.transfer_rate_monthly
AS WITH pc AS (
         SELECT date_trunc('month'::text, x.censusdate)::date AS censusdate,
            x.facilityid,
            count(DISTINCT x.patientid) AS unique_patients,
            x.client
           FROM patient_census x
          GROUP BY (date_trunc('month'::text, x.censusdate)::date), x.facilityid, x.client
        ), tc AS (
         SELECT date_trunc('month'::text, hospital_transfers.dateoftransfer)::date AS dateoftransfer,
            hospital_transfers.facilityid,
            count(1) AS transfers,
            hospital_transfers.client
           FROM hospital_transfers
          GROUP BY (date_trunc('month'::text, hospital_transfers.dateoftransfer)::date), hospital_transfers.facilityid, hospital_transfers.client
        )
 SELECT pc.censusdate,
    pc.client,
    f.facilityname,
    pc.facilityid,
    pc.unique_patients,
    tc.transfers,
    tc.transfers::double precision / pc.unique_patients::double precision AS transfer_rate
   FROM pc
     JOIN tc ON pc.facilityid = tc.facilityid AND pc.censusdate = tc.dateoftransfer AND pc.client = tc.client::text
     JOIN facility f ON pc.facilityid = f.facilityid AND pc.client = f.client::text
  ORDER BY pc.censusdate, pc.facilityid;

-- Permissions

ALTER TABLE public.transfer_rate_monthly OWNER TO saivaadmin;
GRANT ALL ON TABLE public.transfer_rate_monthly TO saivaadmin;
GRANT SELECT ON TABLE public.transfer_rate_monthly TO quicksight_ro;
