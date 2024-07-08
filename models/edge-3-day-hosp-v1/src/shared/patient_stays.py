"""
Create a row for masterpatientid per stay duration.
Based on census date, create stay duration rows
Further based on transfers from rehops table, break down
the stay duration into multiple rows.
"""

import datetime

import numpy as np
import pandas as pd


def _get_patient_stays_from_census(daily_census):
    """
    Create a row for masterpatientid per stay duration
    """
    pids, fids, start_dates, end_dates = [], [], [], []
    masterpids = []
    grouped_daily_census = daily_census.groupby(['facilityid', 'masterpatientid'])
    for group_idx, group in grouped_daily_census:

        if len(group) == 1:
            # If only one row in a group exists
            pids.append(group.patientid.iloc[0])
            masterpids.append(group.masterpatientid.iloc[0])
            fids.append(group.facilityid.iloc[0])
            start_dates.append(group.censusdate.iloc[0])
            end_dates.append(group.censusdate.iloc[0])
        else:
            group = group.sort_values(by='censusdate')
            group_dates = group.censusdate.to_list()
            # Difference bw each date to the date of joining the SNIF
            deltas_in_days = np.array([(group_dates_i - group_dates[0]).days for group_dates_i in group_dates])
            # a[n+1] - a[n]
            diffs = np.diff(deltas_in_days)
            # Return the indices of the elements that are greater than 1
            boundaries = np.nonzero(diffs > 1)[0]
            start_idx = 0
            # If we dont have subsiquent dates then we will have multiple boundaries
            # ie. patient going out of SNIF and comming back again
            # create a row per stay duration
            for end_idx in boundaries:
                # Adding the first date index
                end_idx += 1

                pids.append(group.patientid.iloc[0])
                masterpids.append(group.masterpatientid.iloc[0])
                fids.append(group.facilityid.iloc[0])
                group_dates_for_stay = group_dates[start_idx:end_idx]
                start_dates.append(group_dates_for_stay[0])
                end_dates.append(group_dates_for_stay[-1])

                start_idx = end_idx

            # Last entry for the last boundary to as of now
            pids.append(group.patientid.iloc[0])
            masterpids.append(group.masterpatientid.iloc[0])
            fids.append(group.facilityid.iloc[0])
            group_dates_for_stay = group_dates[start_idx:]
            start_dates.append(group_dates_for_stay[0])
            end_dates.append(group_dates_for_stay[-1])

    patient_stays = pd.DataFrame({'masterpatientid': masterpids,
                                  'patientid': pids,
                                  'facilityid': fids,
                                  'startdate': start_dates,
                                  'enddate': end_dates})
    return patient_stays


def _split_stay_on_date(stay, split_date):
    # TODO - more accurate to see if there is another transfer log entry
    #        that tells us whether we bounced back day after split_date?  
    first_part = stay.copy()
    first_part['enddate'] = split_date
    second_part = stay.copy()
    second_part['startdate'] = split_date
    return first_part, second_part


def _split_stays_by_transfers(patient_stays, transfers):
    new_patient_stays = pd.DataFrame(columns=patient_stays.columns)
    num_splits = 0
    for idx, stay in patient_stays.iterrows():
        if idx % 1000 == 0:
            print(f"Working on stay {idx} / {patient_stays.shape[0]}, {num_splits} splits so far")

        patientid = stay.masterpatientid
        stay_start_date = stay.startdate
        # Add 1 day b/c transfer date is often day after last daily census date.  
        stay_end_date = (stay.enddate + pd.to_timedelta("1 days"))

        # Get transfer events with transfer dates during this stay. 
        transfers_for_pid = transfers[transfers.masterpatientid == patientid]
        mask = (transfers_for_pid.transferdate >= stay_start_date) & \
               (transfers_for_pid.transferdate <= stay_end_date)
        transfer_events = transfers_for_pid[mask]
        transfer_events = transfer_events.sort_values(['transferdate'])

        if len(transfer_events) == 0:
            # No transfers happened during this stay so keep as is. 
            new_patient_stays = new_patient_stays.append(stay.copy(), ignore_index=True)

        elif len(transfer_events) == 1:
            # If only one transfer happened during the stay duration
            transfer_event = transfer_events.iloc[0]

            if transfer_event.transferdate < stay.enddate.date():
                # Split this stay on the transfer date.  
                first_part, second_part = _split_stay_on_date(stay,
                                                              transfer_event.transferdate)
                new_patient_stays = new_patient_stays.append(first_part, ignore_index=True)
                new_patient_stays = new_patient_stays.append(second_part, ignore_index=True)
                num_splits += 1

            else:
                # Adjust the stay end date to match the transfer date
                new_stay = stay.copy()
                new_stay['enddate'] = transfer_event.transferdate
                new_patient_stays = new_patient_stays.append(new_stay, ignore_index=True)

        else:
            # Recursively split the daily census stay using transfer events.  
            # Note - all transfer events have transfer dates <= stay_end_date
            # so base case for recursion WILL trigger b/c we'll run out of 
            # transfer events.  
            this_stay = stay
            transfer_matched_end_p = False

            for transfer_idx, transfer_event in transfer_events.iterrows():
                # We are at last transfer event, and it lines up with end of stay...  
                if transfer_event.transferdate == stay_end_date:
                    transfer_matched_end_p = True
                    break

                # We have a transfer in interior of stay.  
                first_part, second_part = _split_stay_on_date(this_stay,
                                                              transfer_event.transferdate)
                new_patient_stays = new_patient_stays.append(first_part, ignore_index=True)

                # Every second_part might have multiple transfer date in between them.
                # Transfers are sorted in ascending
                num_splits += 1
                this_stay = second_part

            if transfer_matched_end_p:
                second_part['enddate'] = transfer_event.transferdate
            new_patient_stays = new_patient_stays.append(second_part, ignore_index=True)

            num_splits += 1

    print(f'Did {num_splits} splits')
    return new_patient_stays


def get_patient_stays(census, transfers):
    '''
       This mostly just splits census stays based on transfers.  There are other
       edge cases to clean up but this is the most important one. 
    '''

    # Remove duplicates from transfers.  Note - assumes two transfers
    # on same day are an error. 
    print(f'Removing duplicates from transfers...')
    transfers['transferdate'] = transfers.dateoftransfer.dt.date
    key_cols = ['masterpatientid', 'facilityid', 'transferdate']
    transfers = transfers.drop_duplicates(key_cols, keep='first')

    # Add 'enddate' column to transfers... 
    transfers.loc[:, 'enddate'] = transfers.dateoftransfer.dt.date
    print(f'Processing census into proposed stays')
    patient_stays = _get_patient_stays_from_census(census)

    print(f'Splitting stays by transfers...')
    patient_stays = _split_stays_by_transfers(patient_stays, transfers)

    # Convert startdate and enddate into dates (no times)
    new_starts, new_ends = [], []
    for idx, stay in patient_stays.iterrows():
        if type(stay.startdate) != datetime.date:
            new_starts.append(stay.startdate.date())
        else:
            new_starts.append(stay.startdate)
        if type(stay.enddate) != datetime.date:
            new_ends.append(stay.enddate.date())
        else:
            new_ends.append(stay.enddate)
    patient_stays['startdate'] = new_starts
    patient_stays['enddate'] = new_ends

    key_cols = ['masterpatientid', 'facilityid', 'startdate', 'enddate']
    patient_stays = patient_stays.drop_duplicates(key_cols, keep='first').reset_index(drop=True)

    patient_stays = patient_stays.merge(
        transfers,
        how='left',
        left_on=['masterpatientid', 'facilityid', 'enddate'],
        right_on=['masterpatientid', 'facilityid', 'transferdate'],
        suffixes=['', '_y']
    )
    patient_stays = patient_stays.drop(columns=['patientid_y', 'enddate_y'])

    length_of_stay = (patient_stays.enddate - patient_stays.startdate).dt.days.to_numpy()
    patient_stays['lengthofstay'] = length_of_stay

    return patient_stays
