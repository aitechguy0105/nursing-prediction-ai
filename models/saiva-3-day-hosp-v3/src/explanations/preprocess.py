import pandas as pd


class DataProcessor(object):
    def __init__(self, raw_df_dict):
        self.raw_df_dict = raw_df_dict

    def fetch_processed_data(self):
        ret_dict = {}

        ret_dict['patient_vitals'] = self.process_vitals(self.raw_df_dict['patient_vitals'])
        ret_dict['patient_rehosps'] = self.process_rehosps(self.raw_df_dict['patient_rehosps'])
        ret_dict['patient_diagnosis'] = self.process_diagnosis(
            self.raw_df_dict['patient_diagnosis'],
            self.raw_df_dict['patient_room_details']
        )

        if len(self.raw_df_dict['patient_meds']):
            # only if there are meds result rows, do we want to process meds
            ret_dict['patient_meds'] = self.process_medications(self.raw_df_dict['patient_meds'])

        if len(self.raw_df_dict['patient_orders']):
            # only if there are order result rows, do we want to process meds
            ret_dict['patient_orders'] = self.process_diet_and_diagnostic_orders(
                self.raw_df_dict['patient_orders']
            )

        if len(self.raw_df_dict['patient_alerts']):
            ret_dict['patient_alerts'] = self.process_alerts(self.raw_df_dict['patient_alerts'])

        if 'patient_lab_results' in self.raw_df_dict.keys() and len(self.raw_df_dict['patient_lab_results']):
            ret_dict['patient_lab_results'] = self.process_labs(self.raw_df_dict['patient_lab_results'])

        if 'patient_admissions' in self.raw_df_dict.keys() and len(self.raw_df_dict['patient_admissions']):
            ret_dict['patient_admissions'] = self.process_admissions(self.raw_df_dict['patient_admissions'])

        return ret_dict

    def fetch_filter_meds(self):
        """
        Get the list of meds that are in-Significant and drop them
        """
        df = pd.read_csv(f'/src/static/medication_list.csv')
        d_list = list(df.query("Significance=='No'")['pharmacymedicationname'])
        return '|'.join(d_list)

    def process_medications(self, patient_meds):
        patient_meds['only_med_name'] = patient_meds['orderdescription'].str.replace(
            r' Tablet.*| Liquid.*| Powder.*| Packet.*| Solution.*| Suspension.*', '')
        patient_meds = patient_meds.sort_values(by='orderdate', ascending=True)
        predefined_meds = self.fetch_filter_meds()
        # medicines marked 'no' is given 'True' and vice-versa for insignificant_med columns.
        patient_meds["insignificant_med"] = patient_meds["only_med_name"].str.contains(
            predefined_meds,
            na=False,
            case=False
        )

        return patient_meds

    def process_diet_and_diagnostic_orders(self, patient_orders):
        patient_orders = patient_orders.sort_values(by='orderdate', ascending=True)
        return patient_orders

    def process_alerts(self, patient_alerts):
        patient_alerts['createddt'] = patient_alerts['createddate'].dt.date
        patient_alerts = patient_alerts.sort_values(by='createddate', ascending=True)
        return patient_alerts

    def process_diagnosis(self, patient_diagnosis, room_details_df):
        patient_diagnosis = patient_diagnosis.merge(
            room_details_df,
            how='left',
            on=['patientid', 'masterpatientid', 'facilityid']
        )
        patient_diagnosis = patient_diagnosis.sort_values(by='onsetdate', ascending=True)
        return patient_diagnosis

    def process_vitals(self, patient_vitals):
        patient_vitals['orderdt'] = patient_vitals['date'].dt.date
        patient_vitals = patient_vitals.sort_values(by='date', ascending=True)
        return patient_vitals

    def process_rehosps(self, patient_rehosps):
        patient_rehosps = patient_rehosps.sort_values(by='dateoftransfer', ascending=True)
        return patient_rehosps

    def process_labs(self, patient_labs):
        patient_labs['resultdt'] = patient_labs['resultdate'].dt.date
        patient_labs = patient_labs.sort_values(by='resultdate', ascending=True)
        return patient_labs

    def process_admissions(self, patient_admissions):
        patient_admissions['admissiondt'] = patient_admissions['dateofadmission'].dt.date
        patient_admissions = patient_admissions.sort_values(by='dateofadmission', ascending=True)
        return patient_admissions
