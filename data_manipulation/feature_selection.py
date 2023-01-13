from typing import List

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

from classes.feature import Feature


def score_feature_correlation(all_features: List[Feature], X: pd.DataFrame, y: pd.DataFrame) -> List[Feature]:
    features = SelectKBest(score_func=f_regression, k='all')
    features.fit(X, y)

    for feature, score in zip(all_features, features.scores_):
        feature.set_score(score)

    return all_features
