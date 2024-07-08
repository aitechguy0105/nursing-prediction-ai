import abc

import six


@six.add_metaclass(abc.ABCMeta)
class BaseClient(object):

    @abc.abstractmethod
    def get_prediction_queries(self, prediction_date, facilityid, train_start_date):
        raise NotImplementedError(
            self.__class__.__name__ + ' is an abstract class!!! Use proper implementation!!!')

    @abc.abstractmethod
    def get_training_queries(self, test_end_date, train_start_date):
        raise NotImplementedError(
            self.__class__.__name__ + ' is an abstract class!!! Use proper implementation!!!')

    def client_specific_transformations(self, result_dict):
        return result_dict

    def get_note_embeddings_valid_section(self):
        return []

    def get_note_embeddings_emar_types(self):
        return []

    def get_note_embeddings_nan_threshold(self):
        return 0.25

    def get_hospice_payercodes(self):
        return []

    def get_training_dates(self):
        return None, None

    def validate_dataset(self, facilityid, dataset_name, dataset_len):
        # The base implementation makes sure the dataset always has data in it
        assert dataset_len != 0, f'''{dataset_name} , Empty Dataset!'''

    def get_excluded_censusactioncodes(self):
        return []
