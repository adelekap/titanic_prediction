from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

from classes.feature import Feature


class MLModel(ABC):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, features: List[Feature], model_settings: dict = None):
        self.X: pd.DataFrame = X[[f.name for f in features]]
        self.y: pd.DataFrame = y
        self.model_settings: dict = model_settings or {}

        self._model = None
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self._y_pred = None
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
        self.model.fit(self._X_train, self._y_train)

    def _test(self):
        self._y_pred = self.model.predict(self._X_test)

    def _split_datasets(self, test_size: float):
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self.X, self.y,
                                                                                    test_size=test_size, random_state=1)

    def train_and_test(self, test_size: float = 0.33):
        self._split_datasets(test_size)
        self._train()
        self._test()
        self.accuracy = metrics.accuracy_score(self._y_test, self._y_pred)
