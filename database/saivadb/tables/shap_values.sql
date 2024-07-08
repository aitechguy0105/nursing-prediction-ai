-- Drop table

-- DROP TABLE public.shap_values;

CREATE TABLE public.shap_values (
	censusdate timestamp NOT NULL,
	masterpatientid int4 NOT NULL,
	facilityid int4 NOT NULL,
	client varchar(200) NOT NULL,
	modelid varchar(300) NOT NULL,
	feature varchar NOT NULL,
	feature_value varchar NOT NULL,
	feature_type varchar NOT NULL,
	human_readable_name varchar NULL,
	attribution_score float8 NULL,
	sum_attribution_score float8 NULL,
	attribution_percent float8 NULL,
	attribution_rank float8 NULL,
	mapping_status varchar NULL,
	CONSTRAINT shap_values_pk PRIMARY KEY (censusdate, masterpatientid, client, facilityid, feature)
);

-- Permissions

ALTER TABLE public.shap_values OWNER TO saivaadmin;
GRANT ALL ON TABLE public.shap_values TO saivaadmin;
GRANT SELECT ON TABLE public.shap_values TO quicksight_ro;
