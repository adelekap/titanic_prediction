from dataclasses import dataclass

from classes.feature_type import FeatureType


@dataclass
class Feature:
    name: str
    type: FeatureType
    _score: float = None

    @property
    def score(self) -> float:
        return self._score

    def set_score(self, score: float):
        self._score = score

    def __hash__(self):
        return hash(self.name)