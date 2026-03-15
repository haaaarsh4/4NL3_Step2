class Model:
    def __init__(self):
        self.majority_label = None

    def fit(self, X, y):
        self.majority_label = max(set(y), key=list(y).count)

    def predict(self, X):
        return [self.majority_label] * len(X)