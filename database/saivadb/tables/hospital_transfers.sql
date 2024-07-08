-- Drop table

-- DROP TABLE public.hospital_transfers;

CREATE TABLE public.hospital_transfers (
	patientid int4 NOT NULL,
	facilityid int4 NOT NULL,
	dateoftransfer timestamp NOT NULL,
	transferreason varchar(100) NULL,
	client varchar(200) NOT NULL DEFAULT 'unspecified'::character varying,
	purposeofstay varchar NULL,
	transferredto varchar NULL,
	otherreasonfortransfer varchar NULL,
	planned varchar NULL,
	outcome varchar NULL,
	payertype varchar NULL,
	lengthofstay float8 NULL,
	transferredwithin30daysofadmission int4 NULL,
	CONSTRAINT hospital_transfers_transfer_log_v2_patientid_dateoftransfer_key UNIQUE (client, patientid, dateoftransfer)
);
CREATE INDEX fki_hospital_transfers_facilityid_fkey ON public.hospital_transfers USING btree (facilityid);
CREATE INDEX idx_hospital_transfers_patientid_dateoftransfer ON public.hospital_transfers USING btree (patientid, dateoftransfer);

-- Permissions

ALTER TABLE public.hospital_transfers OWNER TO saivaadmin;
GRANT ALL ON TABLE public.hospital_transfers TO saivaadmin;
GRANT SELECT ON TABLE public.hospital_transfers TO quicksight_ro;
