-- Drop table

-- DROP TABLE public.assessment_baseline;

CREATE TABLE public.assessment_baseline (
	"ResponseID" int8 NULL,
	"StdQuestionKey" text NULL,
	"ResponseValue" text NULL,
	"StdAssessID" int8 NULL,
	"AssessmentID" int8 NULL,
	"ItemValue" text NULL,
	"CreatedDate" timestamp NULL,
	"RevisionDate" timestamp NULL,
	"LockedDate" timestamp NULL,
	"ControlType" text NULL,
	"PatientID" int8 NULL,
	"FacilityID" int8 NULL,
	client varchar NULL
);

-- Permissions

ALTER TABLE public.assessment_baseline OWNER TO saivaadmin;
GRANT ALL ON TABLE public.assessment_baseline TO saivaadmin;
GRANT SELECT ON TABLE public.assessment_baseline TO quicksight_ro;
