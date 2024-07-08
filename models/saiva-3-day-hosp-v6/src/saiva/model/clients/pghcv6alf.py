import sys

from .base import Base

class Pghcv6Alf(Base):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.facilities = "SELECT FacilityID FROM view_ods_facility WHERE LineOfBusiness = 'ALF' AND Deleted='N'"
        
    def get_experiment_dates(self):
        return {
            'train_start_date': '2018-08-01',
            'train_end_date': '2022-02-10',
            'validation_start_date': '2022-02-11',
            'validation_end_date': '2023-04-03',
            'test_start_date': '2023-04-04',
            'test_end_date': '2023-07-31',
        }
