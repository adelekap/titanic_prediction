import pandas as pd
import logging

import data_manipulation.engineered_features as eng_features
from data_manipulation.settings import DATA_IMPUTATIONS

logger = logging.getLogger(__name__)


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    data['HasCabin'] = eng_features.has_cabin_feature(data)
    data['FamilySize'] = eng_features.family_size_feature(data)
    data['Title'] = eng_features.title_feature(data)

    return data


def impute_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    for field, imputation in DATA_IMPUTATIONS.items():
        if field in data:
            data[field] = imputation(data[field])

    print(f'Records before data imputation: {len(data)}')
    # Todo: also need to do more intelligent imputation for Age
    data.dropna(inplace=True)
    print(f'Records after data imputation: {len(data)}')

    return data
