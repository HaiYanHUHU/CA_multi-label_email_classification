from .classifier import BaseClassifier
from ..data import Dataset

from sklearn.linear_model import LogisticRegression


class Logistic(BaseClassifier):

    def __init__(self, dataset: Dataset, max_iter=1000, solver='liblinear', C=1.0, penalty='l2',
                 class_weight='balanced'):
        super().__init__(
            LogisticRegression(max_iter=max_iter, solver=solver, C=C, penalty=penalty, class_weight=class_weight),
            dataset, 'logistic regression')
