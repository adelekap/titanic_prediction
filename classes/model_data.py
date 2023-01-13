import numpy as np
from typing import List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from classes.categorical_mapper import CategoricalMapper
from classes.feature import Feature
from classes.feature_type import FeatureType
from data_manipulation.feature_selection import score_feature_correlation


class ModelData:
    def __init__(self, data: pd.DataFrame,
                 features: List[Feature],
                 response_variable: Feature,
                 feature_correlation_quartile: int = 3,
                 test_size: float = 0.33):
        self.feature_names = [f.name for f in features]
        self.data: pd.DataFrame = data[self.feature_names + [response_variable.name]]
        self.all_features: List[Feature] = features
        self.response_variable: Feature = response_variable

        self._categorical_mappings = None
        self._encoded_data = None

        self.X: pd.DataFrame = self.encoded_data
        self.y: pd.DataFrame = self.data[self.response_variable.name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=test_size, random_state=1)

    @property
    def categorical_features(self) -> List[Feature]:
        return [f for f in self.all_features + [self.response_variable] if f.type == FeatureType.Categorical]

    @property
    def categorical_mappings(self) -> Dict[Feature, CategoricalMapper]:
        if self._categorical_mappings is None:
            self._categorical_mappings = {}

            for feature in self.categorical_features:
                self._categorical_mappings[feature] = CategoricalMapper(feature, self.data[feature.name])

        return self._categorical_mappings

    @property
    def encoded_data(self) -> pd.DataFrame:
        # if self._encoded_data is None:
        #     self._encoded_data = self.data.copy()[self.feature_names]
        #
        #     for feature in self.categorical_features:
        #         mapper = self.categorical_mappings[feature]
        #
        #         self._encoded_data[feature.name] = [mapper.get_one_hot_from_value(v) for v in
        #                                             self.encoded_data[feature.name]]
        #
        # return self._encoded_data
        return pd.get_dummies(self.data[self.feature_names],
                              columns=[f.name for f in self.all_features if f.type == FeatureType.Categorical])

    def suggested_features(self, quartile_filter: int = 3) -> List[Feature]:
        scored_features = score_feature_correlation(self.all_features,
                                                    self.encoded_data,
                                                    self.data[self.response_variable.name])
        cutoff = np.quantile([f.score for f in scored_features], quartile_filter / 4)

        strongest_features = [f for f in scored_features if f.score >= cutoff]

        return strongest_features
