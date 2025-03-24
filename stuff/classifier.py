import typing as t
import abc
from sklearn.base import ClassifierMixin
from sklearn.metrics import classification_report
from .data import Dataset


class BaseClassifier(abc.ABC):

    def __init__(self, classifier: ClassifierMixin, dataset: Dataset, name: t.Optional[str] = None):
        self.classifier = classifier
        self.dataset = dataset
        self.name = self.__class__.__name__ if name is None else name

    def train(self):
        self.classifier.fit(self.dataset.X_train, self.dataset.y_train)
        return self

    def predict(self, X: t.Iterable):
        return self.classifier.predict(X)

    def evaluate(self):
        y_pred = self.classifier.predict(self.dataset.X_test)
        return classification_report(self.dataset.y_test, y_pred)
