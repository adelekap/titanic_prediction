from abc import ABC

from sklearn.linear_model import LogisticRegression

from classes.model_data import ModelData
from models.ml_model import MLModel


class LogRegression(MLModel, ABC):
    def __init__(self, data: ModelData, model_settings: dict = None):
        super().__init__(data, model_settings)

        self._model = LogisticRegression(**self.model_settings)
