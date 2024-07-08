-- Table: public.daily_predictions

-- DROP TABLE public.daily_predictions;
CREATE TABLE public.daily_predictions (
	masterpatientid int4 NOT NULL,
	facilityid int4 NOT NULL,
	bedid int4 NULL,
	censusdate timestamp NOT NULL,
	predictionvalue numeric(10,8) NOT NULL,
	predictionrank int8 NOT NULL,
	modelid bpchar(300) NOT NULL,
	createdat timestamptz NOT NULL DEFAULT now(),
	updatedat timestamptz NOT NULL DEFAULT now(),
	client varchar(200) NOT NULL DEFAULT 'unspecified'::character varying,
	experiment_group bool NULL DEFAULT true,
	experiment_group_rank int8 NULL,
	published bool NULL DEFAULT true,
	CONSTRAINT daily_predictions_pkey PRIMARY KEY (client, modelid, facilityid, masterpatientid, censusdate),
	CONSTRAINT daily_predictions_modelid_fkey FOREIGN KEY (modelid) REFERENCES model_metadata(modelid)
);


-- Index: fki_daily_predictions_facilityid_fkey

-- DROP INDEX public.fki_daily_predictions_facilityid_fkey;

CREATE INDEX fki_daily_predictions_facilityid_fkey ON public.daily_predictions USING btree (facilityid);

-- Index: idx_daily_predictions_censusdate

-- DROP INDEX public.idx_daily_predictions_censusdate;

CREATE INDEX idx_daily_predictions_censusdate ON public.daily_predictions USING btree (censusdate);

-- Index: idx_daily_predictions_facilityid_censusdate_modelid_predictionv

-- DROP INDEX public.idx_daily_predictions_facilityid_censusdate_modelid_predictionv;

CREATE INDEX idx_daily_predictions_facilityid_censusdate_modelid_predictionv ON public.daily_predictions USING btree (facilityid, censusdate, modelid, predictionvalue DESC);


-- Table Triggers

-- DROP TRIGGER set_timestamp ON public.daily_predictions;

create trigger set_timestamp before
update
    on
    public.daily_predictions for each row execute procedure trigger_set_timestamp();

-- Permissions

ALTER TABLE public.daily_predictions OWNER TO saivaadmin;
GRANT ALL ON TABLE public.daily_predictions TO saivaadmin;
GRANT SELECT ON TABLE public.daily_predictions TO quicksight_ro;
