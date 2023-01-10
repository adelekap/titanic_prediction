from abc import ABC
from typing import List

import pandas as pd
from sklearn.svm import SVC

from classes.feature import Feature

from models.ml_model import MLModel


class SVM(MLModel, ABC):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, features: List[Feature], model_settings: dict):
        super().__init__(X, y, features, model_settings)

        self._model = SVC(**self.model_settings)

    @property
    def model(self):
        return self._model
