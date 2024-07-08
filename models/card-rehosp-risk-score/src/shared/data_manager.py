import sys

import numpy as np

sys.path.insert(0, '/src')
from shared.admissions import AdmissionFeatures
from shared.alerts import AlertFeatures
from shared.diagnosis import DiagnosisFeatures
from shared.patient_census import PatientCensus
from shared.rehosp import RehospFeatures


class DataManager(object):
    def __init__(self, result_dict, facilityid, client, start_date, end_date, s3_bucket):
        self.facilityid = facilityid
        self.result_dict = result_dict
        self.client = client
        self.start_date = start_date
        self.end_date = end_date
        self.s3_bucket = s3_bucket

        patient_census = PatientCensus(
            census_df=result_dict.get('patient_census', None),
            start_date=self.start_date,
            end_date=self.end_date
        )
        self.census_df = patient_census.generate_features()

        self.alerts = AlertFeatures(
            census_df=self.census_df.copy(),
            alerts=self.result_dict.get('patient_alerts', None),
        )
        self.admissions = AdmissionFeatures(
            census_df=self.census_df.copy(),
            admissions=self.result_dict.get('patient_admissions', None),
        )
        self.diagnosis = DiagnosisFeatures(
            census_df=self.census_df.copy(),
            diagnosis=self.result_dict.get('patient_diagnosis', None),
            s3_bucket=self.s3_bucket,
        )
        self.rehosp = RehospFeatures(
            census_df=self.census_df.copy(),
            rehosps=result_dict.get('patient_rehosps', None)
        )

    def get_patient_census(self):
        return self.census_df

    def get_features(self):
        alerts_df = self.alerts.generate_features()
        admissions_df = self.admissions.generate_features()
        diagnosis_df = self.diagnosis.generate_features()
        rehosp_df = self.rehosp.generate_features()
        return alerts_df, admissions_df, diagnosis_df, rehosp_df

    def merge_features(self, alerts_df, admissions_df, diagnosis_df, rehosp_df):
        final_df = alerts_df.merge(
            admissions_df,
            how='left',
            left_on=['masterpatientid', 'facilityid', 'censusdate'],
            right_on=['masterpatientid', 'facilityid', 'censusdate']
        )

        final_df = final_df.merge(
            diagnosis_df,
            how='left',
            left_on=['masterpatientid', 'facilityid', 'censusdate'],
            right_on=['masterpatientid', 'facilityid', 'censusdate']
        )
        final_df = final_df.merge(
            rehosp_df,
            how='left',
            left_on=['masterpatientid', 'facilityid', 'censusdate'],
            right_on=['masterpatientid', 'facilityid', 'censusdate']
        )
        return final_df

    def generate_total_score(self, final_df):
        final_df['dx_sum'] = final_df['dx_sum'].astype(np.int16)
        final_df['readmissions_score'] = final_df['readmissions_score'].astype(np.int16)
        final_df['admissions_proximity_score'] = final_df['admissions_proximity_score'].astype(np.int16)
        final_df['alert_score'] = final_df['alert_score'].astype(np.int16)

        final_df['total_score'] = final_df['dx_sum'] + final_df['readmissions_score'] + final_df[
            'admissions_proximity_score'] + final_df['alert_score']

        final_df = final_df[
            ['masterpatientid', 'censusdate','facilityid', 'total_score', 'hosp_target_3_day_hosp', 'dx_sum', 'readmissions_score',
             'admissions_proximity_score', 'alert_score']]

        return final_df

    def generate_ranks(self, final_df):
        cutoff = 15
        final_df.sort_values(by='total_score', inplace=True, ascending=False)
        final_df['predictionrank'] = final_df.groupby('censusdate').total_score.rank(
            ascending=False,
            method='first'  # can try`dense`
        )
        final_df['group_rank'] = final_df['predictionrank']
        final_df['group_level'] = 'facility'
        
        final_df.loc[:, 'show_in_report'] = False
        final_df.loc[final_df['predictionrank'] <= cutoff, 'show_in_report'] = True
        
        return final_df
