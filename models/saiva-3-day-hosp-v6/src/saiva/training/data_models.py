import numpy as np
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BaseModel:
    """Class for keeping track of base models"""

    model_name: str
    model_type: str
    model: Any
    config: Any = field(default_factory=dict)
    recalibration: Optional[dict] = None

    def predict(self, x) -> float:
        if self.model_type == "rf":
            return self.model.predict_proba(x)[:, 1]
        elif self.model_type == "lgb":
            preds = self.model.predict(x)
            if self.model.params.get('objective') == 'binary':
                preds = np.log(preds/(1-preds))

            if hasattr(self, 'recalibration') and self.recalibration is not None:
                if self.recalibration['type'].lower() == 'platt':
                    preds = preds * self.recalibration['coef'] + self.recalibration['intercept']
                else:
                    raise NotImplementedError('Only Platt recalibration is supported')
            else:
                warnings.warn("FutureWarning: running inference for the models without recalibration parameter will be deprecated")

            preds = 1 / (1 + np.exp(-preds))
            return preds
        else:
            raise NotImplementedError
            
    def feature_name(self):
        if self.model_type == "lgb":
            return self.model.feature_name()
        elif 'feature_name' in self.config:
            return self.config['feature_name']
        else:
            raise ValueError("For the models other than LightGBM, `feature_name` must be specified in the model.config")

    def truncate_v6_suffix(self):
        """ This is workaround to handle difference in the client names caused by suffices like "v6" or "v6falls".
            These suffices may lead to the different values of the `facility` feature which depends on the client name.
            For example, the problem can happen when the facility is named as "avantev6_1" during the training and
            "avante_1" during the prediction.
            To eliminate the effect during the prediction we do a surgery in the categorical columns values inside the model,
            as well as we remove any suffices started with "v6" while creating the feature during the prediction.
        """
        if self.model_type == "lgb":
            if self.model.pandas_categorical:
                # removing everything starting from v6 or _v6 and till the last underscore
                trunc = lambda x: re.sub('\_*v6.*(_\d+)$', '\g<1>', x)
                self.model.pandas_categorical = [[trunc(value) for value in feat_values] for feat_values in self.model.pandas_categorical]