"""
rank_level : facility/unit/floor
"""

import pandas as pd
from eliot import log_message
from shared.constants import CLIENT_NPI_CONFIG, SHOW_IN_REPORT_CUTOFF
from shared.constants import ENV


class Ranking(object):
    def __init__(self, predictions, facilityid, modelid, client, prediction_date, saiva_engine, client_engine,
                 group_level, subclients, replace_existing_predictions, test, env=ENV):
        self.env = env
        self.replace_existing_predictions = replace_existing_predictions
        self.prediction_df = predictions
        self.census_df = None
        self.prediction_date = prediction_date
        self.client = client
        self.subclients = subclients
        self.group_level = group_level
        self.modelid = modelid
        self.facilityid = facilityid
        self.saivadb_engine = saiva_engine
        self.client_sql_engine = client_engine
        self.test = test

    def execute(self):
        """
        As we may need multiple copies of dataframe for each subclient where the data
        contents change, create duplicate copy of dataframe.
        """
        # Global rank for entire facility patients
        self.prediction_df['predictionrank'] = self.prediction_df.predictionvalue.rank(ascending=False)
        # Rank the overall facility
        prediction_df = self._ranking(
            prediction_df=self.prediction_df.copy(),
            client=self.client,
            group_level=self.group_level
        )
        # Check for sub-clients and rank the same
        if self.subclients:
            self._rank_subclients()

        return prediction_df

    def _ranking(self, prediction_df, client, group_level):
        """
        - Check level at which client needs to be ranked
        - Accordingly populate group_id & group_rank
        - Save the dataframe into daily_predictions table
        """
        prediction_df['group_level'] = group_level
        prediction_df['client'] = client
        
      
        if group_level in ['unit', 'floor']:
            if not self.census_df:
                self._fetch_patient_census()
            prediction_df = self._populate_group_id(
                group_level_id=f'{group_level}_id',
                prediction_df=prediction_df
            )
        elif group_level == 'doctor':     
            prediction_df = self._filter_doctor_specific_patients(
                client=client,
                prediction_df=prediction_df
            )
            prediction_df['predictionrank'] = prediction_df.predictionvalue.rank(ascending=False)

        prediction_df = self._populate_group_rank(group_level, prediction_df)
        prediction_df = self._populate_show_in_report(prediction_df, client)
        
        # For a test run, don't write to Database
        if not self.test:
            self._save_daily_predictions(prediction_df, client)

        return prediction_df

    def _populate_show_in_report(self, prediction_df, client):
        """
        show_in_report field will be used by quicksite and reports to fetch the
        patients who are part of the report for a given day.
        Top cutoff ranks for the overall facility will be marked as show_in_report
        """
        cutoff = SHOW_IN_REPORT_CUTOFF[client][self.facilityid]
        prediction_df.loc[:, 'show_in_report'] = False
        # set show_in_report True for all patients whose rank come below cutoff value
        prediction_df.loc[prediction_df['predictionrank'] <= cutoff, 'show_in_report'] = True
        return prediction_df

    def _rank_subclients(self):
        """
        Loop through all the sub-clients and add separate rows into daily_predictions table
        for each of the sub-client
        """
        for subclient in self.subclients:
            # Rank the sub-clients according to doctor group
            prediction_df = self._ranking(
                prediction_df=self.prediction_df.copy(),
                client=subclient,
                group_level='doctor'
            )

    def _save_daily_predictions(self, prediction_df, client):
        """
        - Delete all old facility data for the given date, client & facilityid
        - Save the dataframe into daily_predictions table
        """
        if self.replace_existing_predictions:
            log_message(
                message_type='info',
                message=f'Delete all old facility data for the given date: {self.prediction_date}'
            )
            self.saivadb_engine.execute(
                f"""delete from daily_predictions where censusdate = '{self.prediction_date}' 
                and facilityid = '{self.facilityid}' and client = '{client}' and modelid = '{self.modelid}'"""
            )

        log_message(
            message_type='info',
            message=f'Save facility data for the given date: {self.prediction_date}, {client}, {self.facilityid},  {prediction_df.shape}'
        )
        prediction_df.to_sql(
            'daily_predictions',
            self.saivadb_engine,
            method='multi',
            if_exists='append',
            index=False
        )

    def _filter_doctor_specific_patients(self, client, prediction_df):
        """
        When we need to filter out patients belonging to a group of doctors
        we use this method
        """
        provider_tuple = CLIENT_NPI_CONFIG[client]
        query = f"""
                    select distinct(fp.masterpatientid)
                    from view_ods_patient_provider a 
                    left join view_ods_provider b on a.facilityid = b.facid and a.providerid = b.providerid 
                    inner join view_ods_facility_patient fp on (a.facilityid = fp.facilityid and a.patientid = fp.patientid) 
                    where b.npi in {provider_tuple}
                    and a.deleted='N'
                    and b.staffdeleted='N'
                    and b.facid = {self.facilityid}
                    """
        selected_patient_list = pd.read_sql(query, con=self.client_sql_engine)
        selected_patient_list = selected_patient_list['masterpatientid'].tolist()

        return prediction_df.query(f'masterpatientid.isin({selected_patient_list})')

    def _fetch_patient_census(self):
        """
        :return: Fetch all patient census for the given facility from saivadb.
        We need patient census to identify unit_id/floor_id for doing ranking at unit/floor level
        Latest patient_census has to get synced before this query is fired
        """
        base_path = f's3://saiva-{self.env}-data-bucket/data/{self.client}/{self.prediction_date}/{self.facilityid}/raw'
        self.census_df = pd.read_parquet(
            f'{base_path}/patient_room_details.parquet'
        )

    def _populate_group_id(self, group_level_id, prediction_df):
        """
        :param group_level_id: floor_id/unit_id
        :param prediction_df
        :return: Identify floor_id/unit_id for every patient populate `group_id` column in dataframe
        """
        census_dict = self.census_df.groupby('masterpatientid')[group_level_id].first().to_dict()
        prediction_df['group_id'] = prediction_df['masterpatientid'].apply(
            lambda x: census_dict[x]
        )
        return prediction_df

    def _populate_group_rank(self, group_level, prediction_df):
        """
        - Issue ranks based on group_level
        - If group_level==unit/floor, issue rank based on group_id
        """
        if group_level in ['facility', 'doctor']:
            prediction_df['group_rank'] = prediction_df.predictionvalue.rank(
                ascending=False
            )
        else:
            prediction_df['group_rank'] = prediction_df.groupby(
                'group_id').predictionvalue.rank(ascending=False)   

        return prediction_df
