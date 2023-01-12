from abc import ABC

from sklearn.svm import SVC

from classes.model_data import ModelData
from models.ml_model import MLModel


class SVM(MLModel, ABC):
    def __init__(self, data: ModelData, model_settings: dict = None):
        super().__init__(data, model_settings)

        self._model = SVC(**self.model_settings)
