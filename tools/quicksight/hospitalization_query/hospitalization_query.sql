select ht.patientid,fp.patientmrn,fp.firstname,fp.lastname, fa.facilityname, ht.facilityid, ht.dateoftransfer, 
ht.transferreason, ht.client, ht.purposeofstay, ht.transferredto, 
ht.otherreasonfortransfer, ht.planned, ht.outcome, ht.payertype, 
ht.lengthofstay, ht.transferredwithin30daysofadmission,ht.payerdescription
from hospital_transfers ht
left join public.facility_patient fp
on ht.client = fp.client
and ht.facilityid = fp.facilityid
and ht.patientid = fp.patientid
inner join public.facility fa
on ht.facilityid=fa.facilityid
and ht.client = fa.client
where
ht.dateoftransfer >= '2020-01-01'