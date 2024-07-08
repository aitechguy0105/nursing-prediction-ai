with rh as (
    select ht.*,
           fa.facilityname,
           fp.masterpatientid,
           dp.modelid,
           dp.group_rank,
           dp.show_in_report,
           fp.patientmrn,
           fp.firstname,
           fp.lastname
    from public.hospital_transfers ht
             left join public.facility_patient fp
                       on ht.client = fp.client
                           and ht.facilityid = fp.facilityid
                           and ht.patientid = fp.patientid
             left join daily_predictions dp
                       on ht.client = dp.client
                           and ht.facilityid = dp.facilityid
                           and (date(ht.dateoftransfer) - date(dp.censusdate)) <= 3
						   and (date(ht.dateoftransfer) - date(dp.censusdate)) != 0
                           and date(dp.censusdate) <= date(ht.dateoftransfer)
                           and fp.masterpatientid = dp.masterpatientid
             left join facility fa
                       on fa.facilityid = ht.facilityid
                           and fa.client = ht.client
    where (dp.published = True or dp.published is null)
      and ht.dateoftransfer >= '2020-01-01 00:00:00'
      and (dp.experiment_group = True or dp.experiment_group is null)
      and (lower(ht.payerdescription) NOT LIKE '%hospice%' or ht.payerdescription is null)
)
SELECT rh.client,
       rh.facilityid,
       rh.facilityname,
       rh.modelid,
       rh.patientid,
       rh.masterpatientid,
       rh.patientmrn,
       rh.lastname,
       rh.firstname,
       rh.dateoftransfer,
       rh.planned,
       rh.transferreason,
       rh.otherreasonfortransfer,
       rh.outcome,
       rh.transferredto,
       rh.lengthofstay,
       rh.payerdescription,
       min(group_rank)            as best_exp_rank,
       bool_or(rh.show_in_report) as show_in_report,
       -- count of how many predictions were made for that day (the number of rows that were grouped)
       -- have to special case for when we made no predictions because there would be still be 1 row
       (CASE
            WHEN bool_or(rh.show_in_report) IS NULL
                THEN 0
            ELSE count(*)
           END
           )                      as num_predictions
FROM rh
GROUP BY rh.client, rh.facilityid, rh.facilityname, rh.modelid,
         rh.patientid, rh.masterpatientid, rh.patientmrn, rh.lastname, rh.firstname, rh.dateoftransfer,
         rh.planned, rh.transferreason, rh.otherreasonfortransfer, rh.outcome,
         rh.transferredto, rh.lengthofstay, rh.payerdescription