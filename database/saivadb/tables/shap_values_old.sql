-- Drop table

-- DROP TABLE public.shap_values_old;

CREATE TABLE public.shap_values_old (
	censusdate timestamp NULL,
	masterpatientid int4 NULL,
	facilityid int4 NULL,
	client varchar(200) NULL,
	modelid varchar(300) NULL,
	feature varchar NULL,
	human_readable_name varchar NULL,
	shap_value float8 NULL,
	shap_rank int4 NULL
);

-- Permissions

ALTER TABLE public.shap_values_old OWNER TO saivaadmin;
GRANT ALL ON TABLE public.shap_values_old TO saivaadmin;
GRANT SELECT ON TABLE public.shap_values_old TO quicksight_ro;
