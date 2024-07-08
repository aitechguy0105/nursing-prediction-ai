import datetime

import typing
from pydantic.dataclasses import dataclass


@dataclass
class JoinMasterPatientLookupConfig:
    merge_on: typing.List[str]
    column_subset: typing.Optional[typing.List[str]] = None
    required: typing.Optional[bool] = False


@dataclass
class ExperimentDates:
    train_start_date: typing.Optional[datetime.date] = None
    train_end_date: typing.Optional[datetime.date] = None
    validation_start_date: typing.Optional[datetime.date] = None
    validation_end_date: typing.Optional[datetime.date] = None
    test_start_date: typing.Optional[datetime.date] = None
    test_end_date: typing.Optional[datetime.date] = None

    def __post_init__(self):
        # if any of train_end_date, validation_start_date, validation_end_date, test_start_date is provided, all dates must be provided
        if self.train_end_date or self.validation_start_date or self.validation_end_date or self.test_start_date:
            assert self.train_start_date and self.train_end_date and self.validation_start_date and self.validation_end_date and self.test_start_date and self.test_end_date, \
            "If any of train_end_date, validation_start_date, validation_end_date or test_start_date is provided, all dates must be provided"


@dataclass
class JoinFeaturesConfig:
    merge_on: typing.List[str]
    feature_group: str
    required: typing.Optional[bool] = False


@dataclass
class ClientConfiguration:
    client: str
    datasource_id: typing.Optional[str] = None
    experiment_dates: typing.Optional[ExperimentDates] = ExperimentDates()
    experiment_dates_facility_wise_overrides: typing.Optional[typing.Dict[int, ExperimentDates]] = None
    """
    facility_ids:
        - if None, all SNF facilities are used, 
        - if str, custom query is used, 
        - if list, list of facility ids is used
    """
    facility_ids: typing.Optional[typing.Union[typing.List[int], str]] = None

    def __post_init__(self):
        if self.datasource_id is None:
            self.datasource_id = self.client
