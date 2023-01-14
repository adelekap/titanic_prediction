from sklearn.ensemble import RandomForestClassifier

from classes.model_data import ModelData
from models.ml_model import MLModel


class RandomForest(MLModel):
    def __init__(self, data: ModelData,  model_settings: dict = None):
        super().__init__(data, model_settings)
        self._model = RandomForestClassifier()
