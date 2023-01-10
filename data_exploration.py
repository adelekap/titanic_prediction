import os

import pandas as pd

from sklearn.model_selection import train_test_split

from feature_selection import suggested_features
from models.svm import train_svm

dir_path = os.path.dirname(os.path.realpath(__file__))

titanic_train_data = pd.read_csv(f'{dir_path}/data/train.csv')
titanic_train_data['Sex'] = [0 if s == 'female' else 1 for s in titanic_train_data['Sex']]

# 20% of Age data is null
titanic_train_data.dropna(inplace=True)  # Todo: remove after get imputation working

dependent_var = 'Survived'
independent_vars = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = titanic_train_data[independent_vars]
y = titanic_train_data[dependent_var]

suggested_features = suggested_features(X, y, 3)
X_train, X_test, y_train, y_test = train_test_split(X[[f.name for f in suggested_features]], y,
                                                    test_size=0.33, random_state=1)

svm, svm_accuracy = train_svm(X_train, y_train, X_test, y_test)
print(svm_accuracy)
