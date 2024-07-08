-- Drop table

-- DROP TABLE public.patient_census;

CREATE TABLE public.patient_census (
	patientid int8 NULL,
	facilityid int8 NULL,
	censusdate timestamp NULL,
	client text NULL
);

-- Permissions

ALTER TABLE public.patient_census OWNER TO saivaadmin;
GRANT ALL ON TABLE public.patient_census TO saivaadmin;
GRANT SELECT ON TABLE public.patient_census TO quicksight_ro;
