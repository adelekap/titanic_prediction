import pandas as pd
import numpy as np
import statistics as stats


def median_imputation(feature_data: pd.Series) -> pd.Series:
    median = np.median(feature_data)

    return feature_data.fillna(median)


def mean_imputation(feature_data: pd.Series) -> pd.Series:
    mean = np.mean(feature_data)

    return feature_data.fillna(mean)


def mode_imputation(feature_data:pd.Series) -> pd.Series:
    mode = stats.mode(feature_data)

    return feature_data.fillna(mode)
