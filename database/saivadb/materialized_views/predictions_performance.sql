-- Used for creating the materialized view used in the Predictions
-- performance tab in mySaiva. The view is refreshed in the monitor-asg DAG

-- For dropping the materialized view:
-- DROP MATERIALIZED VIEW public.predictions_performance;

CREATE MATERIALIZED VIEW predictions_performance AS (
	SELECT
		ht.id,
		ht.facilityid,
		ht.patientid,
		ht.patientfirstname,
		ht.patientlastname,
		ht.dateoftransfer,
		ht.purposeofstay,
		ht.transferredto,
		ht.transferreason,
		ht.otherreasonfortransfer,
		ht.planned,
		ht.outcome,
		ht.medicalrecordnumber,
		ht.tofromtype,
		ht.payertype,
		ht.payerdescription,
		ht.primaryphysiciantitle,
		ht.primaryphysicianfirstname,
		ht.primaryphysicianlastname,
		ht.primaryphysicianprofession,
		ht.originaladmissiondate,
		ht.lastadmissiondate,
		ht.hospitaldischargedate,
		ht.lengthofstay,
		ht.transferredwithin30daysofadmission,
		ht.client,
		fa.facilityname,
		fp.masterpatientid,
		dp.modelid,
		dp.group_rank,
		dp.show_in_report,
		fp.patientmrn,
		fp.firstname,
		fp.lastname
	FROM
		hospital_transfers ht
	LEFT JOIN facility_patient fp ON ht.client = fp.client ::text
		AND ht.facilityid = fp.facilityid
		AND ht.patientid = fp.patientid
	LEFT JOIN daily_predictions dp ON ht.client = dp.client ::text
		AND ht.facilityid = dp.facilityid
		AND(date(ht.dateoftransfer) - date(dp.censusdate)) <= 3
		AND date(dp.censusdate) <= date(ht.dateoftransfer)
		AND fp.masterpatientid = dp.masterpatientid
	LEFT JOIN facility fa ON fa.facilityid = ht.facilityid
		AND fa.client ::text = ht.client
WHERE (dp.published = TRUE
	OR dp.published IS NULL)
AND ht.dateoftransfer >= '2020-01-01 00:00:00' ::timestamp without time zone
AND (ht.lengthofstay is NULL or ht.lengthofstay <= 30::double precision)
AND(dp.experiment_group = TRUE
	OR dp.experiment_group IS NULL)
AND(lower(ht.payerdescription) !~~ '%hospice%'::text
	OR ht.payerdescription IS NULL)
AND fa.is_active = TRUE)