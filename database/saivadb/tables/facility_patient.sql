-- Drop table

-- DROP TABLE public.facility_patient;

CREATE TABLE public.facility_patient (
	patientid int4 NOT NULL,
	facilityid int4 NOT NULL,
	masterpatientid int4 NOT NULL,
	patientmrn varchar(50) NULL,
	createdat timestamptz NOT NULL DEFAULT now(),
	updatedat timestamptz NOT NULL DEFAULT now(),
	client varchar(200) NOT NULL DEFAULT 'unspecified'::character varying,
	originaladmissiondate timestamp NULL,
	recentadmissiondate timestamp NULL,
	initialadmissiondate timestamp NULL,
	CONSTRAINT facility_patient_patientid_key UNIQUE (client, patientid)
);

-- Table Triggers

-- DROP TRIGGER set_timestamp ON public.facility_patient;

create trigger set_timestamp before
update
    on
    public.facility_patient for each row execute procedure trigger_set_timestamp();

-- Permissions

ALTER TABLE public.facility_patient OWNER TO saivaadmin;
GRANT ALL ON TABLE public.facility_patient TO saivaadmin;
GRANT SELECT ON TABLE public.facility_patient TO quicksight_ro;
