from sklearn.neural_network import MLPClassifier
from ..classifier import BaseClassifier
from ..data import Dataset

class DNN(BaseClassifier):
    def __init__(self, dataset: Dataset, hidden_layer_sizes=(100,), max_iter=500):
        super().__init__(
            MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter),
            dataset,
            'dnn'
        )
