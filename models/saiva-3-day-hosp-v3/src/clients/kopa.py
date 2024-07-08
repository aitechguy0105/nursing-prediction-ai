import sys
import pandas as pd

sys.path.insert(0, '/src')
from clients.avante import Avante


class Kopa(Avante):

    def npi(self):
        return ('1003866823', '1013522242', '1043785074', '1285978585', '1427447952', '1518122290', '1962972570')
    
    def residents_to_be_ranked(self, censusdate, facilityid,  client_sql_engine, prediction_df):
        """
        When we need to filter out patients belonging to a group of doctors
        we use this method
        """
        provider_tuple = self.npi()
        query = f"""
                    select distinct(fp.masterpatientid)
                    from view_ods_patient_provider a 
                    left join view_ods_provider b on a.facilityid = b.facid and a.providerid = b.providerid 
                    inner join view_ods_facility_patient fp on (a.facilityid = fp.facilityid and a.patientid = fp.patientid) 
                    where b.npi in {provider_tuple}
                    and a.deleted='N'
                    and b.staffdeleted='N'
                    and b.facid = {facilityid}
                    """
        include_patient_list = pd.read_sql(query, client_sql_engine)
        include_patient_list = include_patient_list['masterpatientid'].tolist()

        exclude_query = f"""
                select f.masterpatientid from dbo.view_ods_daily_census_v2 c join view_ods_payer p ON c.PayerID = p.PayerID 
                join view_ods_facility_patient f on c.ClientID = f.PatientID and c.FacilityID = f.FacilityID 
                where c.CensusDate = '{censusdate}' and p.PayerType IN ('Managed Care', 'Medicare A') and c.FacilityID={facilityid}
                """
        exclude_patient_list = pd.read_sql(exclude_query, con=client_sql_engine)
        exclude_patient_list = exclude_patient_list['masterpatientid'].tolist()

        return prediction_df[~prediction_df.masterpatientid.isin(exclude_patient_list) & prediction_df.masterpatientid.isin(include_patient_list)]