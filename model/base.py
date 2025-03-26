from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


class BaseModel(ABC):
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