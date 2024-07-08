-- Drop table

-- DROP TABLE public.facility;

CREATE TABLE public.facility (
	facilityid int4 NOT NULL,
	facilityname varchar(375) NULL,
	lineofbusiness varchar(10) NULL,
	client varchar(200) NOT NULL DEFAULT 'unspecified'::character varying,
	CONSTRAINT facility_facilityid_key UNIQUE (client, facilityid)
);

-- Permissions

ALTER TABLE public.facility OWNER TO saivaadmin;
GRANT ALL ON TABLE public.facility TO saivaadmin;
GRANT SELECT ON TABLE public.facility TO quicksight_ro;
