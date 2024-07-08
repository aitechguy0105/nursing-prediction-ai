import gc
import sys

import scipy

sys.path.insert(0, '/src')
from shared import diagnosis
from shared import meds
from shared import utils
from shared import vitals
from shared import demographics
from shared import stays
from shared import notes
from eliot import log_message
import pickle as pkl


class Featurizer1(object):
    """
    Process Meds, diagnoses & vitals
    self.modelid indicates whether its a training or prediction run
    """

    def __init__(self, result_dict, prediction_times, modelid=None):
        self.diagnoses = result_dict['patient_diagnosis']
        self.meds = result_dict['patient_meds']
        self.vitals = result_dict['patient_vitals']
        self.prediction_times = prediction_times
        self.modelid = modelid  # Provided only during prediction

    def process_diagnoses(self):
        log_message(message_type='info', message='Processing Diagnoses...')
        dx_featurizer = diagnosis.BagOfAggregatedDxCodes(stop_at_dots=True)
        # get_aggregated_codes depends on get_code_map
        # During prediction use the pre-saved code_map
        dx_code_map = dx_featurizer.get_code_map(
            codes=self.diagnoses.diagnosiscode.values,
            save_file=f'/data/models/{self.modelid}/artifacts/diagnoses_codes.pkl' if self.modelid else None
        )

        if not self.modelid:
            with open('/data/processed/diagnoses_codes.pkl', 'wb') as f_out:
                pkl.dump(dx_code_map, file=f_out)

        new_dx_codes = dx_featurizer.get_aggregated_codes(self.diagnoses.diagnosiscode.values)
        self.diagnoses['AggDxCode'] = new_dx_codes
        dx_features = dx_featurizer.featurize(self.prediction_times, self.diagnoses, 'AggDxCode')

        stay_ids = self.prediction_times.stayrowindex.values
        dx_features = utils.forward_fill_by_stay(dx_features, stay_ids)
        dx_features_csr = utils.convert_df_to_csr_matrix(dx_features)
        dx_colnames = dx_features.columns.values
        return dx_features_csr, dx_colnames

    def process_meds(self):
        log_message(message_type='info', message='Processing Meds...')
        med_featurizer = meds.BagOfAggregatedMedNames()

        # During prediction use the pre-saved code_map
        med_code_map = med_featurizer.aggregate_codes(
            meds=self.meds,
            save_file=f'/data/models/{self.modelid}/artifacts/med_codes.pkl' if self.modelid else None
        )

        if not self.modelid:
            with open('/data/processed/med_codes.pkl', 'wb') as f_out:
                pkl.dump(med_code_map, file=f_out)

        new_med_names = med_featurizer.get_aggregated_codes(self.meds.pharmacymedicationname.values)
        self.meds['AggMedName'] = new_med_names
        rx_features = med_featurizer.featurize(self.prediction_times, self.meds, 'AggMedName')
        stay_ids = self.prediction_times.stayrowindex.values
        rx_features = utils.forward_fill_by_stay(rx_features, stay_ids)
        rx_features_csr = utils.convert_df_to_csr_matrix(rx_features)
        rx_colnames = rx_features.columns.values
        return rx_features_csr, rx_colnames

    def process_vitals(self):
        log_message(message_type='info', message='Processing Vitals...')
        time_bins = [('-5 hours', '7 hours'),  # Last night
                     ('-17 hours', '-5 hours'),  # Yesterday day shift
                     ('-41 hours', '-17 hours')]

        # If modelid exists load vital_bins from model Artifacts or else during
        # training dump the vital_bins into the Artifacts
        if not self.modelid:
            with open('/data/processed/vital_bins.pkl', 'wb') as f_out:
                pkl.dump(time_bins, file=f_out)
        else:
            with open(f'/data/models/{self.modelid}/artifacts/vital_bins.pkl', 'rb') as f_in:
                time_bins = pkl.load(f_in)

        vitals_featurizer = vitals.BasicTimeBinnedVitals()
        vitals_features = vitals_featurizer.featurize(
            self.prediction_times,
            self.vitals,
            time_bins
        )
        vitals_colnames = vitals_features.columns.values
        vitals_features_csr = utils.convert_df_to_csr_matrix(vitals_features)
        return vitals_features_csr, vitals_colnames

    def process(self):
        self.diagnoses = self.diagnoses.sort_values(by=['masterpatientid', 'onsetdate'])
        self.meds = self.meds.sort_values(by=['masterpatientid', 'orderdate'])
        self.vitals = self.vitals.sort_values(by=['masterpatientid', 'date'])

        dx_features_csr, dx_colnames = self.process_diagnoses()
        rx_features_csr, rx_colnames = self.process_meds()
        vitals_features_csr, vitals_colnames = self.process_vitals()

        features = scipy.sparse.hstack([
            dx_features_csr,
            rx_features_csr,
            vitals_features_csr
        ])
        log_message(message_type='info', message='Create CSR Matrix...')
        features = scipy.sparse.csr_matrix(features)
        feature_names = list(dx_colnames) \
                        + list(rx_colnames) \
                        + list(vitals_colnames)

        #         with open('/data/test/p_dx_col.pkl', 'wb') as f_out:
        #             pkl.dump(dx_colnames, file=f_out)
        #         with open('/data/test/p_rx_col.pkl', 'wb') as f_out:
        #             pkl.dump(rx_colnames, file=f_out)
        #         with open('/data/test/p_vitals_col.pkl', 'wb') as f_out:
        #             pkl.dump(vitals_colnames, file=f_out)

        return features, feature_names


class Featurizer2(object):
    """
    Process Demographics, stays & progress notes
    """

    def __init__(self, result_dict, prediction_times):
        self.demographics = result_dict['patient_demographics']
        self.progress_notes = result_dict['patient_progress_notes']
        self.stays = result_dict['stays']
        self.prediction_times = prediction_times

    def process_demographics(self):
        """
        * DOB => Age (but note that some missing DOB so include an NA indicator).
        * Gender - easy.
        * Education - somewhat messy; ignore for now.
        * Citizenship - almost all US so ignore.
        * Race - messy; ignore for now.
        * Religion - very messy; ignore.
        * State - long tail; plenty of blanks and tons of people from IL, IN, AR, KY, TX for some reason. Ignore for now.
        * Primary Language - Mostly English, about 900 Spanish, then a long tail. Ignore for now.
        """
        log_message(message_type='info', message='Processing Demographics...')
        demo_featurizer = demographics.BasicDemographicsFeatures()
        demo_features = demo_featurizer.featurize(self.prediction_times, self.demographics)
        demo_features_csr = utils.convert_df_to_csr_matrix(demo_features)
        demo_features_colnames = demo_features.columns.values
        return demo_features_csr, demo_features_colnames

    def process_stays(self):
        """
        Encode information from prior stays...  Try to capture idea that hospitalizations seem to occur in clusters...
        * Define observation period of M months
        * Number of stays in last M months (defined as number of transfer out events)
        * Number of rehosps in last M months (defined as number of transfer out events ending in a rehosp)
        * Prior stay ended in rehosp (0 if no prior stay or prior stay didn't end in rehosp; 1 otherwise).
        * Prior rehosp was a night (between say 9 and 7).
        * Days into current admission.
        * Days between last admission and prior transfer out (or 0 if no prior transfer out).
        """
        log_message(message_type='info', message='Processing Stays...')
        stays_featurizer = stays.StaysFeaturizer()
        stay_features = stays_featurizer.featurize(self.prediction_times, self.stays)
        stay_features_csr = utils.convert_df_to_csr_matrix(stay_features)
        stay_features_colnames = stay_features.columns.values
        return stay_features_csr, stay_features_colnames

    def process_progress_notes(self):
        log_message(message_type='info', message='Processing Notes...')
        notes_featurizer = notes.SWEMNotesFeaturizer()
        mean_note_embeddings = notes_featurizer.featurize(
            self.prediction_times,
            self.progress_notes,
            agg_mode='mean'
        )
        mean_note_embedding_csr = utils.convert_df_to_csr_matrix(mean_note_embeddings)
        mean_note_embedding_colnames = mean_note_embeddings.columns.values

        del mean_note_embeddings
        gc.collect()

        max_note_embeddings = notes_featurizer.featurize(
            self.prediction_times,
            self.progress_notes,
            agg_mode='max'
        )
        max_note_embeddings_csr = utils.convert_df_to_csr_matrix(max_note_embeddings)
        max_note_embedding_colnames = max_note_embeddings.columns.values

        del max_note_embeddings
        gc.collect()

        return mean_note_embedding_csr, mean_note_embedding_colnames, \
               max_note_embeddings_csr, max_note_embedding_colnames

    def process(self, features_csr, features_colnames):
        demo_features_csr, demo_colnames = self.process_demographics()
        stay_features_csr, stay_colnames = self.process_stays()
        mean_note_embedding_csr, mean_note_embedding_colnames, \
        max_note_embeddings_csr, max_note_embedding_colnames = self.process_progress_notes()

        features = scipy.sparse.hstack([
            features_csr,
            demo_features_csr,
            stay_features_csr,
            mean_note_embedding_csr,
            max_note_embeddings_csr
        ])
        log_message(message_type='info', message='Creating CSR Matrix...')
        features = scipy.sparse.csr_matrix(features)

        feature_names = list(features_colnames) \
                        + list(demo_colnames) \
                        + list(stay_colnames) \
                        + list(mean_note_embedding_colnames) \
                        + list(max_note_embedding_colnames)

        #         with open('/data/test/p_demo_col.pkl', 'wb') as f_out:
        #             pkl.dump(demo_colnames, file=f_out)
        #         with open('/data/test/p_stay_col.pkl', 'wb') as f_out:
        #             pkl.dump(stay_colnames, file=f_out)
        #         with open('/data/test/p_mean_note_col.pkl', 'wb') as f_out:
        #             pkl.dump(mean_note_embedding_colnames, file=f_out)
        #         with open('/data/test/p_max_note_col.pkl', 'wb') as f_out:
        #             pkl.dump(max_note_embedding_colnames, file=f_out)

        return features, feature_names
