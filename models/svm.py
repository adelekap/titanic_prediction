from typing import Tuple

import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics


def train_svm(X_train: pd.DataFrame, y_train: pd.DataFrame,
              X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[SVC, float]:
    svm = SVC(kernel='rbf', gamma=0.6)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    svm_accuracy = metrics.accuracy_score(y_test, y_pred)

    return svm, svm_accuracy
