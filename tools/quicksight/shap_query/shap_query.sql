with rh as
(select fp.patientid, fp.masterpatientid, fp.patientmrn, fp.lastname, fp.firstname,
dp.censusdate, ht.dateoftransfer, dp.experiment_group_rank,
dp.experiment_group, dp.modelid,
ht.client, ht.facilityid, fa.facilityname, ht.planned,
ht.transferreason, ht.otherreasonfortransfer, ht.outcome, ht.transferredto,
ht.lengthofstay, dp.published, 
sv.feature, sv.feature_type, sv.human_readable_name, sv.attribution_rank
from public.hospital_transfers ht
left join public.facility_patient fp
on ht.client = fp.client
and ht.facilityid = fp.facilityid
and ht.patientid = fp.patientid
left join daily_predictions dp
on ht.client = dp.client
and ht.facilityid = dp.facilityid
and (date(ht.dateoftransfer) - date(dp.censusdate)) <= 3
-- and (date(ht.dateoftransfer) - date(dp.censusdate)) != 0
and date(dp.censusdate) <= date(ht.dateoftransfer)
and fp.masterpatientid = dp.masterpatientid
left join facility fa
on fa.facilityid = ht.facilityid
and fa.client = ht.client
left join shap_values sv
on sv.client = ht.client 
and sv.censusdate = dp.censusdate 
and sv.facilityid = ht.facilityid 
and sv.masterpatientid = dp.masterpatientid
and sv.attribution_rank <=5
where
dp.published = True
and dp.experiment_group = True
order by ht.facilityid, ht.dateoftransfer asc, dp.censusdate asc)
SELECT rh.client, rh.facilityid, rh.facilityname, rh.experiment_group, rh.modelid,
rh.patientid, rh.masterpatientid, rh.patientmrn, rh.lastname, rh.firstname, rh.censusdate, rh.dateoftransfer,
rh.planned, rh.transferreason, rh.otherreasonfortransfer, rh.outcome,
rh.transferredto, rh.lengthofstay, rh.feature, rh.feature_type, rh.human_readable_name, rh.attribution_rank,
min(experiment_group_rank) as best_exp_rank
FROM rh
GROUP BY rh.client, rh.facilityid, rh.facilityname, rh.experiment_group, rh.modelid,
rh.patientid, rh.masterpatientid, rh.patientmrn, rh.lastname, rh.firstname, rh.censusdate, rh.dateoftransfer,
rh.planned, rh.transferreason, rh.otherreasonfortransfer, rh.outcome,
rh.transferredto, rh.lengthofstay, rh.feature, rh.feature_type, 
rh.human_readable_name, rh.attribution_rank
HAVING min(experiment_group_rank) <= 1000
order by client, facilityid, censusdate, dateoftransfer,masterpatientid, attribution_rank