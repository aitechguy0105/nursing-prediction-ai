WITH unranked_census AS (
	SELECT
		c.censusdate,
		c.facilityid,
		c.{patiendid_col} as patientid,
		min(c.bedid) as bedid
	FROM
		view_ods_daily_census_v2 c
	WHERE
		censusdate >= '{start_date}'
		AND (payername LIKE '%hospice%'OR  {censusactioncode_filter})
	GROUP BY
		c.censusdate,
		c.facilityid,
		c.{patiendid_col}
),
distinct_facility_patient AS (
	select distinct patientid, facilityid, masterpatientid from view_ods_facility_patient
)
SELECT
	'{client}' AS client,
	c.censusdate,
	c.facilityid,
	fp.masterpatientid,
	c.bedid,
	'FALSE' AS show_in_report,
	'TRUE' AS experiment_group,
	NULL AS predictionvalue,
	NULL AS predictionrank,
	NULL AS modelid,
	NULL AS published,
	NULL AS group_rank,
	NULL AS group_level,
	NULL AS group_id,
	NULL AS censusactioncode,
	NULL AS payername,
	NULL AS payercode,
	NULL AS admissionstatus,
	NULL AS to_from_type,
	getdate() as createdat,
	getdate() as updatedat
FROM
	unranked_census c
	JOIN distinct_facility_patient fp ON c.facilityid = fp.facilityid
		AND c.patientid = fp.patientid