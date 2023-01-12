from abc import ABC

from sklearn import metrics

from classes.model_data import ModelData


class MLModel(ABC):
    def __init__(self, data: ModelData, model_settings: dict = None):
        self.data: ModelData = data
        self.model_settings: dict = model_settings or {}

        self._model = None

        self._accuracy = None

    @property
    def model(self):
        return self._model

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @accuracy.setter
    def accuracy(self, accuracy):
        self._accuracy = accuracy

    def _train(self):
        self.model.fit(self.data.X_train, self.data.y_train)

    def _test(self):
        self._y_pred = self.model.predict(self.data.X_test)

    def train_and_test(self):
        self._train()
        self._test()
        self.accuracy = metrics.accuracy_score(self.data.y_test, self._y_pred)
