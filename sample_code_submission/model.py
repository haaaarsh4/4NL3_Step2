from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

class Model:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.clf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', multi_class='auto')

    def fit(self, X, y):
        X_transformed = self.vectorizer.fit_transform(X)
        self.clf.fit(X_transformed, y)

    def predict(self, X):
        X_transformed = self.vectorizer.transform(X)
        return list(self.clf.predict(X_transformed))
