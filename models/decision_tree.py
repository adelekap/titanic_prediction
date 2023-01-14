from sklearn.tree import DecisionTreeClassifier

from classes.model_data import ModelData
from models.ml_model import MLModel


class DecisionTree(MLModel):
    def __init__(self, data: ModelData):
        super().__init__(data)
        self._model = DecisionTreeClassifier()
