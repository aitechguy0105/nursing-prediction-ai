-- Drop table

-- DROP TABLE public.model_metadata;

CREATE TABLE public.model_metadata (
	modelid bpchar(300) NOT NULL,
	dayspredictionvalid int8 NOT NULL,
	predictiontask varchar(100) NOT NULL,
	modeldescription varchar(1000) NULL,
	prospectivedatestart date NULL,
	CONSTRAINT model_metadata_modeldescription_key UNIQUE (modeldescription),
	CONSTRAINT model_metadata_pkey PRIMARY KEY (modelid)
);

-- Permissions

ALTER TABLE public.model_metadata OWNER TO saivaadmin;
GRANT ALL ON TABLE public.model_metadata TO saivaadmin;
GRANT SELECT ON TABLE public.model_metadata TO quicksight_ro;
