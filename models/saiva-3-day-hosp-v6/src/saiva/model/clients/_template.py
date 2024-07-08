import sys

from .base import Base

# --1-- Change the class name below
class Template(Base):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ============== --2-- self.client_specific_*_queries ================
        # ==========================   (optional)   ========================== 
        # Use these attributes to set specific queries for the given client
        # Use {key: None} if you want to exclude query
        #
        # Examples:
        # self.client_specific_prediction_queries = dict()
        # self.client_specific_training_queries = dict()
        
        # ======================= --3-- self.faciliies =======================
        # ==========================   (optional)   ========================== 
        # This is an SQL-compliant string used to fetch the list of facilities
        # whose data will be used for training and predictions. It should not
        # be confused with the list of facilities for which predictions will
        # be made, which is stored in the database. We can make predictions
        # for 5 facilities while using data from 10, as patients may move
        # between facilities.
        #
        # You do not need to set this attribute unless your client requires
        # customization of any kind. By default, all sniffs will be used.
        #
        # Examples:
        # self.facilities = '1,2,3,4,5,6,7,8,9,10,11,12,13'
        # self.facilities = "SELECT FacilityID FROM view_ods_facility WHERE LineOfBusiness = 'SNF' AND FacilityState = 'CT'"

    def get_experiment_dates(self):
        """
        --3-- Before training a new client, configure the 'train_start_date' and 'test_end_date'.
        
        Here are some recommendations for selecting these dates:
        - It is recommended to use a dataset of 30 months in length.
        - Try to use the most recent data available, but avoid using the last week of the dataset,
          as it may be corrupted by back corrections.
        
        You will need to adjust four other dates later.
        """
        return {
            'train_start_date': '2020-07-01',
            'train_end_date': '2022-06-04',
            'validation_start_date': '2022-06-05',
            'validation_end_date': '2022-09-18',
            'test_start_date': '2022-09-19',
            'test_end_date': '2022-12-31'
        }