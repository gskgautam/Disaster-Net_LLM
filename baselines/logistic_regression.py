from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LogisticRegressionBaseline:
    def __init__(self, C=1.0, max_iter=1000):
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs', multi_class='auto')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y, y_pred, average='macro', zero_division=0)
        } 