from typing import Tuple

import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.DataFrame,
                              X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[SVC, float]:
    return
