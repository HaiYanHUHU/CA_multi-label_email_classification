from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


class BaseModel(ABC):
    """
    The third and final feature that architecture should allow is to implement multiple ML models in such a way that the model-level
    behavioral differences (e.g., differences in training and testing codes related to each model) should be hidden from the controller. In
    other words, all the modeling-related functionalities should be able to access using a consistent interface (e.g., a set of methods) no
    matter how different each model is in terms of their coding for those functionalities.
    """

    def __init__(self) -> None:
        ...


    @abstractmethod
    def train(self) -> None:
        """
        Train the model using ML Models for Multi-class and mult-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """
        ...

    @abstractmethod
    def predict(self) -> int:
        """

        """
        ...

    @abstractmethod
    def data_transform(self) -> None:
        return

    # def build(self, values) -> BaseModel:
    def build(self, values={}):
        values = values if isinstance(values, dict) else utils.string2any(values)
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self

    
    def print_results(self, data):
        """
        Print classification report comparing true vs predicted labels.
        """
        print(f"\nResults for model: {self.__class__.__name__}")
        print(classification_report(data.y_test, self.y_pred))