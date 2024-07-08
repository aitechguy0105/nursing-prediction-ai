import sys

from .admissions import AdmissionFeatures
from .demographics import DemographicFeatures
from .labs import LabFeatures
from .meds import MedFeatures
from .orders import OrderFeatures
from .vitals import VitalFeatures
from .alerts import AlertFeatures
from .rehosp import RehospFeatures
from .diagnosis import DiagnosisFeatures
from .notes import NoteFeatures
from .immunizations import ImmunizationFeatures
from .risks import RiskFeatures
from .patient_census import PatientCensus
from .assessments import AssessmentFeatures
from .adt import AdtFeatures
import pandas as pd
from eliot import start_action, log_message

class DataManager(object):
    def __init__(
            self,
            *,
            result_dict,
            facilityid,
            client,
            prediction_date,
            train_start_date,
            diagnosis_lookup_ccs_s3_file_path,
            modelid=None,
            training=False,
            save_outputs_in_s3=False,
            s3_base_path=None,
            save_outputs_in_local=False,
            local_folder=None,
            vector_model='SpacyModel',
            model_config=None
    ):
        self.facilityid = facilityid
        self.result_dict = result_dict
        self.client = client
        self.training = training
        self.train_start_date = train_start_date
        self.prediction_date = prediction_date
        self.modelid = modelid
        self.save_outputs_in_s3 = save_outputs_in_s3
        self.s3_base_path = s3_base_path
        self.save_outputs_in_local = save_outputs_in_local
        self.local_folder = local_folder
        self.diagnosis_lookup_ccs_s3_file_path = diagnosis_lookup_ccs_s3_file_path

        patient_census = PatientCensus(
            census_df=self.result_dict.get('patient_census', None),
            train_start_date=self.train_start_date,
            test_end_date=self.prediction_date
        )
        
        self.census_df = patient_census.generate_features()

        self.demo = DemographicFeatures(
            census_df=self.census_df.copy(),
            demo_df=self.result_dict.get('patient_demographics', None),
            training=self.training)
        self.vitals = VitalFeatures(
            census_df=self.census_df.copy(),
            vitals=self.result_dict.get('patient_vitals', None),
            training=self.training,
            config=model_config)
        self.orders = OrderFeatures(
            census_df=self.census_df.copy(),
            orders=self.result_dict.get('patient_orders', None),
            training=self.training,
            config=model_config)
        self.meds = MedFeatures(
            census_df=self.census_df.copy(),
            meds=self.result_dict.get('patient_meds', None),
            training=self.training,
            config=model_config)
        self.alerts = AlertFeatures(
            census_df=self.census_df.copy(),
            alerts=self.result_dict.get('patient_alerts', None),
            training=self.training,
            config=model_config)
        self.rehosp = RehospFeatures(
            census_df=self.census_df.copy(),
            rehosps=self.result_dict.get('patient_rehosps', None),
            adt_df=self.result_dict.get('patient_adt', None),
            training=self.training,
            config=model_config)
        self.admissions = AdmissionFeatures(
            census_df=self.census_df.copy(),
            admissions=self.result_dict.get('patient_admissions', None),
            training=self.training)
        self.diagnosis = DiagnosisFeatures(
            census_df=self.census_df.copy(),
            diagnosis=self.result_dict.get('patient_diagnosis', None),
            diagnosis_lookup_ccs_s3_file_path=self.diagnosis_lookup_ccs_s3_file_path,
            training=self.training,
            config=model_config)
        self.labs = LabFeatures(
            census_df=self.census_df.copy(),
            labs=self.result_dict.get('patient_lab_results', None),
            training=self.training,
            config=model_config)
        self.notes = NoteFeatures(
            census_df=self.census_df.copy(),
            notes=self.result_dict.get('patient_progress_notes', None),
            client=self.client,
            training=self.training)
        self.immuns = ImmunizationFeatures(
            census_df=self.census_df.copy(),
            immuns_df=self.result_dict.get('patient_immunizations', None),
            training=self.training,
            config=model_config)
        self.risks = RiskFeatures(
            census_df=self.census_df.copy(),
            risks_df=self.result_dict.get('patient_risks', None),
            training=self.training,
            config=model_config)
        self.assessments = AssessmentFeatures(
            census_df=self.census_df.copy(),
            assessments_df=self.result_dict.get('patient_assessments', None),
            config=model_config,
            training=self.training)
        self.adt = AdtFeatures(
            census_df=self.census_df.copy(),
            adt_df=self.result_dict.get('patient_adt', None),
            config=model_config,
            training=self.training)

    def _save_dataframes(self, feature_df_dict):
        """
        Save the input dataframes for testing
        :feature_df_dict: dictionaries of names, dataframes to save
        """
        if self.save_outputs_in_s3:
            for name, feature_df in feature_df_dict.items():
                df_s3_path = self.s3_base_path + f'/{name}_output.parquet'
                feature_df.to_parquet(df_s3_path, index=False)

        if self.save_outputs_in_local:
            for name, feature_df in feature_df_dict.items():
                local_path = self.local_folder + f'/{name}_output.parquet'
                feature_df.to_parquet(local_path, index=False)

    def get_features(self):
        feature_df_dict = {
            'census': self.census_df
        }
        
        final_df = self.census_df
        
        if not self.result_dict.get('patient_demographics', pd.DataFrame()).empty:
            demo_df = self.demo.generate_features()
            feature_df_dict['demo'] = demo_df
            final_df = demo_df

        if not self.result_dict.get('patient_vitals', pd.DataFrame()).empty:
            vitals_df = self.vitals.generate_features()
            feature_df_dict['vitals'] = vitals_df
            final_df = final_df.merge(
                vitals_df,
                how='left',
                left_on=['masterpatientid', 'facilityid', 'censusdate'],
                right_on=['masterpatientid', 'facilityid', 'censusdate']
            )
        if not self.result_dict.get('patient_rehosps', pd.DataFrame()).empty:
            rehosp_df = self.rehosp.generate_features()
            feature_df_dict['rehosp'] = rehosp_df
            final_df = final_df.merge(
                rehosp_df,
                how='left',
                left_on=['masterpatientid', 'facilityid', 'censusdate'],
                right_on=['masterpatientid', 'facilityid', 'censusdate']
            )
        
        if not self.result_dict.get('patient_admissions', pd.DataFrame()).empty:
            admissions_df = self.admissions.generate_features()
            feature_df_dict['admissions'] = admissions_df
            final_df = final_df.merge(
                admissions_df,
                how='left',
                left_on=['masterpatientid', 'facilityid', 'censusdate'],
                right_on=['masterpatientid', 'facilityid', 'censusdate']
            )
            
        if not self.result_dict.get('patient_meds', pd.DataFrame()).empty:
            meds_df, self.result_dict['patient_meds'] = self.meds.generate_features()
            feature_df_dict['meds'] = meds_df
            final_df = final_df.merge(
                meds_df,
                how='left',
                left_on=['masterpatientid', 'facilityid', 'censusdate'],
                right_on=['masterpatientid', 'facilityid', 'censusdate']
            )
            
        if not self.result_dict.get('patient_alerts', pd.DataFrame()).empty:
            alerts_df = self.alerts.generate_features()
            feature_df_dict['alerts'] = alerts_df
            final_df = final_df.merge(
                alerts_df,
                how='left',
                left_on=['masterpatientid', 'facilityid', 'censusdate'],
                right_on=['masterpatientid', 'facilityid', 'censusdate']
            )
            
        if not self.result_dict.get('patient_orders', pd.DataFrame()).empty:
            orders_df = self.orders.generate_features()
            feature_df_dict['orders'] = orders_df
            final_df = final_df.merge(
                orders_df,
                how='left',
                left_on=['masterpatientid', 'facilityid', 'censusdate'],
                right_on=['masterpatientid', 'facilityid', 'censusdate']
            )
            
        if not self.result_dict.get('patient_diagnosis', pd.DataFrame()).empty:
            diagnosis_df, self.result_dict['patient_diagnosis'] = self.diagnosis.generate_features()
            feature_df_dict['diagnosis'] = diagnosis_df
            final_df = final_df.merge(
                diagnosis_df,
                how='left',
                left_on=['masterpatientid', 'facilityid', 'censusdate'],
                right_on=['masterpatientid', 'facilityid', 'censusdate']
            )
            
        if not self.result_dict.get('patient_lab_results', pd.DataFrame()).empty:
            labs_df = self.labs.generate_features()
            feature_df_dict['labs'] = labs_df
            final_df = final_df.merge(
                labs_df,
                how='left',
                left_on=['masterpatientid', 'facilityid', 'censusdate'],
                right_on=['masterpatientid', 'facilityid', 'censusdate']
            )
            
        if not self.result_dict.get('patient_progress_notes', pd.DataFrame()).empty:
            notes_df = self.notes.generate_features()
            feature_df_dict['notes'] = notes_df
            final_df = final_df.merge(
                notes_df,
                how='left',
                left_on=['masterpatientid', 'facilityid', 'censusdate'],
                right_on=['masterpatientid', 'facilityid', 'censusdate']
            )
            
        if not self.result_dict.get('patient_immunizations', pd.DataFrame()).empty:
            immuns_df = self.immuns.generate_features()
            feature_df_dict['immuns'] = immuns_df
            final_df = final_df.merge(
                immuns_df,
                how='left',
                left_on=['masterpatientid', 'facilityid', 'censusdate'],
                right_on=['masterpatientid', 'facilityid', 'censusdate']
            )
        if not self.result_dict.get('patient_risks', pd.DataFrame()).empty:
            risks_df = self.risks.generate_features()
            feature_df_dict['risks'] = risks_df
            final_df = final_df.merge(
                risks_df,
                how='left',
                left_on=['masterpatientid', 'facilityid', 'censusdate'],
                right_on=['masterpatientid', 'facilityid', 'censusdate']
            )
        if not self.result_dict.get('patient_assessments', pd.DataFrame()).empty:
            assessments_df = self.assessments.generate_features()
            feature_df_dict['assessments'] = assessments_df
            final_df = final_df.merge(
                assessments_df,
                how='left',
                left_on=['masterpatientid', 'facilityid', 'censusdate'],
                right_on=['masterpatientid', 'facilityid', 'censusdate']
            )
        if not self.result_dict.get('patient_adt', pd.DataFrame()).empty:
            adt_df = self.adt.generate_features()
            feature_df_dict['adt'] = adt_df
            final_df = final_df.merge(
                adt_df,
                how='left',
                left_on=['masterpatientid', 'facilityid', 'censusdate'],
                right_on=['masterpatientid', 'facilityid', 'censusdate']
            )    
        cleaned_final_df = self.clean_final_dataframe(final_df)
        feature_df_dict['cleaned_final_df'] = cleaned_final_df
        self._save_dataframes(feature_df_dict)

        return cleaned_final_df

    def get_modified_result_dict(self):
        return self.result_dict

    def clean_final_dataframe(self, final_df):
        # drop unwanted columns
        cols = {'payername_x':'payername',
                'payercode_x':'payercode',
                'censusactioncode_x':'censusactioncode',
                'to_from_type_x':'to_from_type'}
        final_df.rename(columns = cols, inplace=True)
        final_df.drop(
            final_df.columns[final_df.columns.str.contains('_masterpatientid|_facilityid|hosp_censusdate|dateofadmission|_x$|_y$')].tolist()
            , axis=1, inplace=True)

        if 'patientid' in final_df.columns:
            # If patientid included in the above regex pattern it drops masterpatientid column even
            final_df.drop(
                ['patientid'],
                axis=1,
                inplace=True
            )

        return final_df
