from typing import List

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression

from models.feature import Feature


def score_feature_correlation(X: pd.DataFrame, y: pd.DataFrame) -> List[Feature]:
    feature_names = X.columns
    features = SelectKBest(score_func=f_regression, k='all')
    features.fit(X, y)

    scored_features = [Feature(feature_names[i], score) for i, score in enumerate(features.scores_)]

    return scored_features


def suggested_features(X: pd.DataFrame, y: pd.DataFrame, quartile_filter: int = 3) -> List[Feature]:
    scored_features = score_feature_correlation(X, y)
    cutoff = np.quantile([f.score for f in scored_features], quartile_filter/4)

    strongest_features = [f for f in scored_features if f.score >= cutoff]

    return strongest_features
