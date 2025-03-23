from model.base import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class Logistic(BaseModel):
    def __init__(self, model_name, embeddings, y):
        super().__init__() 
        
        # 
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y

        # 
        self.model = LogisticRegression(
            max_iter=1000,
            penalty='l2',
            solver='liblinear',
            C=1.0,
            class_weight='balanced'
        )

    def train(self, data):
        self.model.fit(data.X_train, data.y_train)

    def predict(self, X_test):
        self.y_pred = self.model.predict(X_test)

    def print_results(self, data):
        # 
        super().print_results(data)

    def data_transform(self, X):
        # Convert sparse to dense
        return X.toarray()
