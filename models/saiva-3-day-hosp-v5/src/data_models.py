from dataclasses import dataclass
from typing import Any


@dataclass
class BaseModel:
    """Class for keeping track of base models"""

    model_name: str
    model_type: str
    model: Any

    def predict(self, x) -> float:
        if self.model_type == "rf":
            return self.model.predict_proba(x)[:, 1]
        elif self.model_type == "lgb":
            return self.model.predict(x)
        else:
            raise NotImplementedError
