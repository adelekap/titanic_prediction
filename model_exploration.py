import os

import pandas as pd

from classes.model_data import ModelData
from data_manipulation.data_cleaning import add_engineered_features, impute_missing_data
from data_manipulation.settings import TITANIC_FEATURES, RESPONSE_VARIABLE
from models.decision_tree import DecisionTree
from models.log_regression import LogRegression
from models.random_forest import RandomForest
from models.svm import SVM

dir_path = os.path.dirname(os.path.realpath(__file__))

raw_data = pd.read_csv(f'{dir_path}/data/train.csv')
data_with_engineered_features = add_engineered_features(raw_data)
enriched_data = impute_missing_data(
    data_with_engineered_features[[f.name for f in TITANIC_FEATURES] + [RESPONSE_VARIABLE.name]])

data = ModelData(enriched_data, TITANIC_FEATURES, RESPONSE_VARIABLE)

svm = SVM(data, {'kernel': 'rbf', 'gamma': 0.6, 'max_iter': 800})
svm.train_and_test()
print(svm.accuracy)

log_reg = LogRegression(data, {'C': 0.7, 'max_iter': 500})
log_reg.train_and_test()
print(f'Logistic Regression Accuracy: {log_reg.accuracy}')

decision_tree = DecisionTree(data)
decision_tree.train_and_test()
print(f'Decision Tree Accuracy: {decision_tree.accuracy}')

random_forest = RandomForest(data)
random_forest.train_and_test()
print(f'Random Forest Accuracy: {random_forest.accuracy}')
