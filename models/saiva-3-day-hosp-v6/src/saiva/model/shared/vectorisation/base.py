import abc

import six

from saiva.model.shared.utils import get_client_class


@six.add_metaclass(abc.ABCMeta)
class VectorisationModel(object):
    def __init__(self, notes, client):
        self.notes_df = notes
        self.client = client
        self.clientClass = get_client_class(self.client)

    @abc.abstractmethod
    def get_embedding_models(self):
        """
        Calls get_grouped_notes, get_config and returns the dict config
        """
        raise NotImplementedError(
            self.__class__.__name__ + ' is an abstract class!!! Use proper implementation!!!')

    @abc.abstractmethod
    def get_grouped_notes(self):
        """
        If we have to group notes based on notetypes and do any processing
        we do it in this method.
        Eg: In Fasttext models we seperate emar_notes, progress_notes
        """
        raise NotImplementedError(
            self.__class__.__name__ + ' is an abstract class!!! Use proper implementation!!!')

    @abc.abstractmethod
    def get_config(self, *args):
        """
        Create a dictionary where we can add multiple models
        and segregate notes.
        embedding_models = [
            {
                'name': 'eMar',
                'model_path': f'/data/models/ft_emar.model',
                'notes': emar_notes,
            },
            {
                'name': 'pn',
                'model_path': f'/data/models/ft_non_emar.model',
                'notes': progress_notes,
            },
        ]
        """
        raise NotImplementedError(
            self.__class__.__name__ + ' is an abstract class!!! Use proper implementation!!!')

    @abc.abstractmethod
    def load(self, name):
        """
        Load model into memory
        """
        raise NotImplementedError(
            self.__class__.__name__ + ' is an abstract class!!! Use proper implementation!!!')

    @abc.abstractmethod
    def vectorise(self, notes_df, model):
        """
        Given model and notes as input parameters, vectorise the notes
        """
        raise NotImplementedError(
            self.__class__.__name__ + ' is an abstract class!!! Use proper implementation!!!')