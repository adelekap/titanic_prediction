from classes.feature import Feature
from classes.feature_type import FeatureType

TITANIC_FEATURES = [
                    # Feature('Pclass', FeatureType.Numerical),
                    Feature('Sex', FeatureType.Categorical),
                    Feature('Age', FeatureType.Numerical),
                    # Feature('Fare', FeatureType.Numerical),
                    # Feature('Embarked', FeatureType.Categorical),
                    # Feature('HasCabin', FeatureType.Numerical),
                    # Feature('FamilySize', FeatureType.Numerical),
                    # Feature('Title', FeatureType.Categorical)
                    ]

RESPONSE_VARIABLE = Feature('Survived', FeatureType.Numerical)
