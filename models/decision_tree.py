import pandas as pd
from typing import List

from classes.feature import Feature
from models.ml_model import MLModel


class DecisionTree(MLModel):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, features: List[Feature]):
        super().__init__(X, y, features)
