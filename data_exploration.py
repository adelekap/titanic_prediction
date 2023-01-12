import os

import pandas as pd

from classes.model_data import ModelData
from data_manipulation.data_cleaning import add_engineered_features, impute_missing_data
from data_manipulation.titanic_features import TITANIC_FEATURES, RESPONSE_VARIABLE
from models.log_regression import LogRegression
from models.svm import SVM

dir_path = os.path.dirname(os.path.realpath(__file__))

raw_data = pd.read_csv(f'{dir_path}/data/train.csv')
enriched_data = impute_missing_data(add_engineered_features(raw_data))


data = ModelData(enriched_data, TITANIC_FEATURES, RESPONSE_VARIABLE)
print(data.suggested_features())

svm = SVM(data, {'kernel': 'rbf', 'gamma': 0.6})
svm.train_and_test()
print(svm.accuracy)

log_reg = LogRegression(data)
log_reg.train_and_test()
print(log_reg.accuracy)
