import sys

from .base_sigmacare import BaseClientSigmacare

class EpicHcm(BaseClientSigmacare):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_experiment_dates(self):
        return {
            'train_start_date': '2021-01-01',
            'train_end_date': '2022-08-26',
            'validation_start_date': '2022-08-27',
            'validation_end_date': '2022-12-22',
            'test_start_date': '2022-12-23',
            'test_end_date': '2023-04-16',
        }
