import abc

import six


@six.add_metaclass(abc.ABCMeta)
class BaseClient(object):

    @abc.abstractmethod
    def get_prediction_queries(self, prediction_date, facilityid):
        raise NotImplementedError(
            self.__class__.__name__ + ' is an abstract class!!! Use proper implementation!!!')
    
    def client_specific_transformations(self, result_dict):
        return result_dict

    def get_training_dates(self):
        return None, None

    def validate_dataset(self, facilityid, dataset_name, dataset_len):
        # The base implementation makes sure the dataset always has data in it
        assert dataset_len != 0, f'''{dataset_name} , Empty Dataset!'''
