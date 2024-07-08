import sys

from .base_matrixcare import BaseClientMatrixcare

class Woodruff(BaseClientMatrixcare):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_experiment_dates(self):
        """
        Configure train_start_date and test_end_date before training a new client
        """
        return {
            'train_start_date': '2023-03-12',
            'train_end_date': '2023-05-18',
            'validation_start_date': '2023-05-19',
            'validation_end_date': '2023-06-24',
            'test_start_date': '2023-06-25',
            'test_end_date': '2023-07-31',
        }
