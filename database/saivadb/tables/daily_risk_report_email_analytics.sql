-- Drop table

-- DROP TABLE public.daily_risk_report_email_analytics;

CREATE EXTENSION IF NOT EXISTS citext;

CREATE TABLE public.daily_risk_report_email_analytics (
    source_tracking_id TEXT PRIMARY KEY,
	user_email CITEXT NOT NULL,
    client varchar(200) NOT NULL DEFAULT 'unspecified'::character varying,
	facility_id int4 NOT NULL,
    facility_name TEXT NOT NULL,
    report_date timestamp NOT NULL,
    open_count INT NOT NULL DEFAULT 0 CHECK (open_count >= 0),
    sent_at timestamptz NOT NULL DEFAULT now(),
	last_opened_at timestamptz NULL
);

-- Permissions

ALTER TABLE public.daily_risk_report_email_analytics OWNER TO saivaadmin;
GRANT ALL ON TABLE public.daily_risk_report_email_analytics TO saivaadmin;
GRANT SELECT ON TABLE public.daily_risk_report_email_analytics TO quicksight_ro;
