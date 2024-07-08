import pandas as pd
import numpy as np
import gc
import sys
from pathlib import Path
sys.path.insert(0, '/src')

from datetime import timedelta


class PatientCensus(object):
    def __init__(self,client_sql_engine, start_date, end_date):
        self.client_sql_engine = client_sql_engine
        self.start_date = start_date
        self.end_date = end_date
            
    
    def get_patient_transfers(self):
        query=f"""
        select distinct tl.patientid, tl.facilityid, CONVERT(varchar, tl.dateoftransfer, 23) as dateoftransfer, 
        tl.transferreason 
        from dbo.view_ods_hospital_transfers_transfer_log_v2 tl
        WHERE tl.dateoftransfer BETWEEN '{self.start_date}' AND '{self.end_date}'
        """

        transfer_df = pd.read_sql(query, con=self.client_sql_engine)

        transfer_df.drop_duplicates(subset=['patientid','facilityid','dateoftransfer','transferreason'], keep='first', inplace=True)
        transfer_df['dateoftransfer'] = pd.to_datetime(transfer_df['dateoftransfer'])

        return transfer_df
    

    def get_patient_census(self):
        query=f"""
                select fp.masterpatientid,dc.patientid, dc.censusdate, dc.facilityid, dc.bedid, dc.beddescription, 
                dc.roomratetypedescription, dc.payercode, dc.carelevelcode
                from view_ods_daily_census_v2 dc JOIN view_ods_facility_patient fp
                ON (dc.patientid = fp.patientid and dc.facilityid = fp.facilityid)
                where dc.censusdate BETWEEN '{self.start_date}' AND '{self.end_date}'
                and dc.censusactioncode not in ('DAMA', 'DD', 'DE', 'DH', 'E', 'HU', 'L', 'LV', 'MO', 'TO', 'TP', 'TU')
                and (dc.payername not like '%hospice%' or dc.payername is null)
                """

        census_df = pd.read_sql(query, con=self.client_sql_engine)

        base_df = pd.DataFrame({'censusdate': pd.date_range(
                    start=self.start_date, end=self.end_date)}
                )
        base_df = base_df.merge(census_df, how='left', on=['censusdate'])

        base_df.drop_duplicates(subset=['masterpatientid','censusdate','facilityid'], keep='first', inplace=True)
        
        transfer_df = self.get_patient_transfers()
        # Merge transfers & census
        df = base_df.merge(
                    transfer_df, 
                    how='left',
                    left_on=['patientid','facilityid', 'censusdate'],
                    right_on=['patientid','facilityid','dateoftransfer']
                )
        df['transfered'] = df['dateoftransfer'].notna()

        return df


class DataLoader(object):
    def __init__(self,client_sql_engine, facilityid=None,masterpatientid_list=[],census_date=None):
        self.facilityid = facilityid
        self.masterpatientid_list = masterpatientid_list
        self.census_date = census_date
        self.masterpatient_condition = ''
        self.client_sql_engine = client_sql_engine
        self.preload()
    
    def preload(self):
        if len(self.masterpatientid_list) > 1:
            self.masterpatient_condition += f'in {tuple(self.masterpatientid_list)}'
        elif len(self.masterpatientid_list) == 1:
            self.masterpatient_condition += f'= {self.masterpatientid_list[0]}'
        
    
    def load_demographics(self):
        query = f"""
        select fp.patientid,fp.facilityid, mp.* from view_ods_master_patient mp
        JOIN view_ods_facility_patient fp 
        ON (mp.masterpatientid = fp.masterpatientid) 
        where mp.masterpatientid {self.masterpatient_condition}
        """
        demo_df = pd.read_sql(query, con=self.client_sql_engine)
        demo_df = demo_df.sort_values(by=['masterpatientid'], ascending=False)
        print(f'Demo : {demo_df.shape}')
        return demo_df
    
    def load_vitals(self):
        query = f"""
        select vt.patientid,fp.masterpatientid, vt.facilityid, vt.date, vt.bmi, vt.vitalsdescription, vt.value, vt.diastolicvalue, vt.warnings
        from view_ods_Patient_weights_vitals vt JOIN view_ods_facility_patient fp 
        ON (fp.patientid = vt.patientid and fp.facilityid = vt.facilityid)
        where fp.masterpatientid {self.masterpatient_condition}
        """
        if self.census_date:
            query += f" and vt.date <= '{self.census_date}'"

        vital_df = pd.read_sql(query, con=self.client_sql_engine)
        vital_df = vital_df.sort_values(by=['date'], ascending=False)
        print(f'Vital : {vital_df.shape}')
        return vital_df
    
    def load_orders(self):
        query = f"""
        select distinct ord.patientid,fp.masterpatientid, ord.facilityid, ord.orderdate, ord.ordercategory, ord.ordertype,
        ord.orderdescription, ord.pharmacymedicationname, ord.diettype, ord.diettexture, ord.dietsupplement
        from view_ods_physician_order_list_v2 ord JOIN view_ods_facility_patient fp 
        ON (fp.patientid = ord.patientid and fp.facilityid = ord.facilityid)
        where fp.masterpatientid {self.masterpatient_condition}
        and ord.ordercategory in ('Diagnostic', 'Enteral - Feeding', 'Dietary - Diet', 'Dietary - Supplements','Laboratory')
        """

        if self.census_date:
            query += f" and ord.orderdate <= '{self.census_date}'"

        ord_df = pd.read_sql(query, con=self.client_sql_engine)
        ord_df = ord_df.sort_values(by=['orderdate'], ascending=False)
        print(f'Orders : {ord_df.shape}')
        return ord_df
    
    def load_meds(self):
        query = f"""
        select distinct a.patientid,fp.masterpatientid, a.facilityid, orderdate,
        gpiclass, gpisubclassdescription, orderdescription, 
        pharmacymedicationname,
        a.PhysicianOrderID, a.discontinueddate, a.MAREndDate  
        from view_ods_physician_order_list_v2 a
        inner join view_ods_physician_order_list_med b
        on a.PhysicianOrderID = b.PhysiciansOrderID 
        inner JOIN view_ods_facility_patient fp 
        ON (fp.patientid = a.patientid and fp.facilityid = a.facilityid)
        where fp.masterpatientid {self.masterpatient_condition}
        """

        if self.census_date:
            query += f" and orderdate <= '{self.census_date}'"

        med_df = pd.read_sql(query, con=self.client_sql_engine)
        med_df = med_df.sort_values(by=['orderdate'], ascending=False)
        print(f'Meds : {med_df.shape}')
        return med_df
    
    def load_diagnosis(self):
        query = f"""
        select fp.masterpatientid, dg.onsetdate, dg.facilityid, dg.diagnosiscode,
        dg.diagnosisdesc, dg.classification, dg.rank, dg.resolveddate, dg.deleted, dg.struckout
        from view_ods_patient_diagnosis dg JOIN view_ods_facility_patient fp 
        ON (fp.patientid = dg.patientid and fp.facilityid = dg.facilityid)
        where fp.masterpatientid {self.masterpatient_condition}
        """

        if self.census_date:
            query += f" and dg.onsetdate <= '{self.census_date}'"

        dg_df = pd.read_sql(query, con=self.client_sql_engine)
        dg_df = dg_df.sort_values(by=['onsetdate'], ascending=False)
        print(f'diagnosis : {dg_df.shape}')
        return dg_df
    
    def load_alerts(self):
        query = f"""
        select al.patientid,fp.masterpatientid, al.facilityid, al.createddate, al.stdalertid, 
        al.alertdescription, al.triggereditemtype
        from view_ods_cr_alert al JOIN view_ods_facility_patient fp 
        ON (fp.patientid = al.patientid and fp.facilityid = al.facilityid)
        where fp.masterpatientid {self.masterpatient_condition}
        """

        if self.census_date:
            query += f" and al.createddate <= '{self.census_date}'"

        alt_df = pd.read_sql(query, con=self.client_sql_engine)
        alt_df = alt_df.sort_values(by=['createddate'], ascending=False)
        print(f'Alerts : {alt_df.shape}')
        return alt_df
    
    def load_notes(self):
        query = f"""
        select pn.progressnotetype,fp.masterpatientid,pn.createddate, pn.sectionsequence, pn.section, pn.notetextorder, pn.notetext
        from view_ods_progress_note pn JOIN view_ods_facility_patient fp 
        ON (fp.patientid = pn.patientid and fp.facilityid = pn.facilityid)
        where fp.masterpatientid {self.masterpatient_condition}
        """

        if self.census_date:
            query += f" and pn.createddate <= '{self.census_date}'"

        notes_df = pd.read_sql(query, con=self.client_sql_engine)
        notes_df = notes_df.sort_values(by=['createddate', 'sectionsequence'], ascending=False)
        print(f'Notes : {notes_df.shape}')
        return notes_df
    
    