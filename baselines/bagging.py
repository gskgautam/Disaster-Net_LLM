from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BaggingBaseline:
    def __init__(self, base_estimator=None, n_estimators=10):
        self.model = BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators)

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