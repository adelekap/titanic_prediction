import pandas as pd

from data_manipulation.data_imputation import mode_imputation, median_imputation
import data_manipulation.engineered_features as eng_features


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    data['HasCabin'] = eng_features.has_cabin_feature(data)
    data['FamilySize'] = eng_features.family_size_feature(data)
    data['Title'] = eng_features.title_feature(data)

    return data


def impute_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    data['Embarked'] = mode_imputation(data['Embarked'])
    data['Fare'] = median_imputation(data['Fare'])
    # Todo: also need to do more intelligent imputation for Age
    data.dropna(inplace=True) # eventually remove

    return data