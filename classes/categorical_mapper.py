from typing import List, Any, Set, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from classes.feature import Feature


class CategoricalMapper:
    def __init__(self, feature: Feature, values: List[Any]):
        self.feature: Feature = feature
        self.values: List[Any] = values

        self.distinct_values: Set[Any] = set(self.values)
        self.value_to_category: Dict[Any, int] = {v: i for i, v in enumerate(self.distinct_values)}
        self.category_to_value: Dict[int, Any] = {v: k for k, v in self.value_to_category.items()}

    def _encode(self, category: int) -> NDArray:
        encoding = [0] * len(self.distinct_values)
        encoding[category] = 1

        # return np.array(encoding)
        return category  # Todo: need to fix data shape

    def get_value_from_category(self, category: int) -> Optional[Any]:
        return self.category_to_value.get(category)

    def get_category_from_value(self, value: Any) -> Optional[int]:
        return self.value_to_category.get(value)

    def get_one_hot_from_value(self, value: Any) -> Optional[NDArray]:
        category = self.get_category_from_value(value)

        if category is not None:
            return self._encode(category)

        raise Warning(f"Categorical feature value not found: {value}")

    def get_value_from_one_hot(self, one_hot_encoding: List[int]) -> Optional[Any]:
        return self.get_value_from_category(one_hot_encoding.index(1))
