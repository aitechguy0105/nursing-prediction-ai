import pandas as pd
import numpy as np
import datetime
import os

from multiprocessing import Pool


def _stayFeaturizeWorker(job_data):
    N = job_data[0]
    td = job_data[1]
    group = job_data[2].copy()
    group = group.sort_values("startdate")
    group = group.reset_index(drop=True)

    night_start = datetime.time(18, 0, 0)
    night_end = datetime.time(7, 0, 0)

    rows = []
    num_prior_stays_l, num_prior_rehosps_l = [], []
    prev_stay_rehosp_l, prev_stay_night_rehosp_l = [], []
    prev_stay_los_l, days_between_stays_l = [], []
    for idx, stay in group.iterrows():
        rows.append(stay.RowNumber)
        min_date = stay.startdate - td
        # This is critical line - we only look at stays that ended prior to
        # the current stay, but within td days of the start of the current
        # stay.  This should prevent any leakage from current stay.
        sel = (group.enddate >= min_date) & (group.enddate <= stay.startdate)
        # noinspection PyUnresolvedReferences
        num_prior_stays = sel.sum()
        sel = sel & (~group.dateoftransfer.isna())
        num_prior_rehosps = sel.sum()

        prev_stay_rehosp = 0
        prev_stay_night_rehosp = 0
        prev_stay_los = -1
        days_between_stays = -1
        # If there are prior stays within the window, then look
        # at the immediately preceding stay...
        if idx >= 1 and num_prior_stays >= 1:
            prev_stay = group.iloc[idx - 1]
            # How long was it?
            prev_stay_los = prev_stay.lengthofstay
            # Did it end in a rehosp event?
            if not pd.isna(prev_stay.dateoftransfer):
                prev_stay_rehosp = 1
                # If so, was the rehosp event at night?
                transfer_time = prev_stay.dateoftransfer.time()
                if transfer_time > night_start or transfer_time < night_end:
                    prev_stay_night_rehosp = 1
                # How long was the patient away?
                days_between_stays = (
                    pd.Timestamp(stay.startdate) - prev_stay.dateoftransfer
                ).days + 1

        num_prior_stays_l.append(num_prior_stays)
        num_prior_rehosps_l.append(num_prior_rehosps)
        prev_stay_rehosp_l.append(prev_stay_rehosp)
        prev_stay_night_rehosp_l.append(prev_stay_night_rehosp)
        prev_stay_los_l.append(prev_stay_los)
        days_between_stays_l.append(days_between_stays)

    retval = {
        "rows": np.array(rows),
        "num_prior_stays": np.array(num_prior_stays_l),
        #'num_prior_stays': np.array(num_prior_stays), # This is a cheat used to extrapolate a curve.  DO NOT USED!!!
        "num_prior_rehosps": np.array(num_prior_rehosps_l),
        "prev_stay_rehosp": np.array(prev_stay_rehosp_l),
        "prev_stay_night_rehosp": np.array(prev_stay_night_rehosp_l),
        "prev_stay_los": np.array(prev_stay_los_l),
        "days_between_stays": np.array(days_between_stays_l),
    }
    return retval


class StaysFeaturizer:
    def __init__(self):
        pass

    def featurize(self, prediction_times, stays, lookback="90 days"):
        stays = stays.copy()
        stays = stays.reset_index(drop=True)
        stays["RowNumber"] = stays.index.values

        print("Constructing jobs data...")
        jobs_data = []
        grouped_stays = stays.groupby("masterpatientid")
        td = pd.to_timedelta(lookback)
        N = len(stays)
        for pid, group in grouped_stays:
            jobs_data.append((N, td, group))

        print("Starting jobs!")
        with Pool(min(os.cpu_count() - 4, 24)) as pool:
            features_by_patient = pool.map(_stayFeaturizeWorker, jobs_data)

        # Construct final data frame
        print("Constructing data frame...")
        num_prior_stays = np.zeros(N, dtype=np.float32)
        num_prior_rehosps = np.zeros(N, dtype=np.float32)
        prev_stay_rehosp = np.zeros(N, dtype=np.float32)
        prev_stay_night_rehosp = np.zeros(N, dtype=np.float32)
        prev_stay_los = np.zeros(N, dtype=np.float32)
        days_between_stays = np.zeros(N, dtype=np.float32)

        for patient_dict in features_by_patient:
            indices = patient_dict["rows"]
            num_prior_stays[indices] = patient_dict["num_prior_stays"]
            num_prior_rehosps[indices] = patient_dict["num_prior_rehosps"]
            prev_stay_rehosp[indices] = patient_dict["prev_stay_rehosp"]
            prev_stay_night_rehosp[indices] = patient_dict["prev_stay_night_rehosp"]
            prev_stay_los[indices] = patient_dict["prev_stay_los"]
            days_between_stays[indices] = patient_dict["days_between_stays"]

        stay_indices = prediction_times.stayrowindex.values
        df = pd.DataFrame(
            {
                "stays_num_prior_stays": num_prior_stays[stay_indices],
                "stays_num_prior_rehosps": num_prior_rehosps[stay_indices],
                "stays_prev_stay_rehosp": prev_stay_rehosp[stay_indices],
                "stays_prev_stay_night_rehosp": prev_stay_night_rehosp[stay_indices],
                "stays_prev_stay_los": prev_stay_los[stay_indices],
                "stays_days_between_stays": days_between_stays[stay_indices],
            }
        )

        # Now merge into prediction times.
        print("Calculating day of stay...")
        prediction_dates = prediction_times.predictiontimestamp
        start_dates = [pd.Timestamp(d) for d in stays.startdate[stay_indices]]
        days_in_current_stay = [
            dt.days for dt in (prediction_dates - pd.Series(start_dates))
        ]
        df["stays_days_of_stay"] = np.array(days_in_current_stay, dtype=np.float32)

        return df
