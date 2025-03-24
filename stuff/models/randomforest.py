import typing as t

from sklearn.ensemble import RandomForestClassifier

from ..classifier import BaseClassifier
from ..data import Dataset


class RandomF(BaseClassifier):

    def __init__(self, dataset: Dataset, n_estimators: int = 100):
        super().__init__(RandomForestClassifier(n_estimators), dataset, 'random forest')
