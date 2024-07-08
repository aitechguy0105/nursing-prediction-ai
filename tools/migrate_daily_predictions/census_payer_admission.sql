WITH census AS (
	SELECT
		censusdate,
		facilityid,
		{patiendid_col} as patientid,
		min(censusactioncode) as censusactioncode,
		min(payername) as payername,
		min(payercode) as payercode
	FROM
		view_ods_daily_census_v2
	WHERE
		censusdate > '{start_date}'
	GROUP BY
		censusdate,
		facilityid,
		{patiendid_col}
),
census_admissions as (
SELECT
	c.censusdate,
	c.facilityid,
	c.patientid,
	c.censusactioncode,
	c.payername,
	c.payercode,
	admissions.AdmissionStatus as admissionstatus,
	admissions.ToFromTypeDescription as to_from_type
FROM
	census c
	OUTER APPLY (
		SELECT
			TOP 1 a.AdmissionStatus, a.ToFromTypeDescription
		FROM
			view_ods_hospital_transfers_admission_log a
		WHERE
			 c.facilityid = a.facilityid and c.patientid = a.patientid and c.censusdate >= a.DateOfAdmission order by DateOfAdmission DESC) admissions 
),
distinct_facility_patient AS (
	select distinct patientid, facilityid, masterpatientid from view_ods_facility_patient
)		
SELECT ca.*, '{client}' as client, fp.masterpatientid from census_admissions ca JOIN distinct_facility_patient fp on ca.facilityid = fp.facilityid and ca.patientid = fp.patientid  order by ca.censusdate, ca.facilityid, ca.patientid