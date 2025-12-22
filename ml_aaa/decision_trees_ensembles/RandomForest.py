import numpy as np
from MyDecisionTreeClassifier import DecisionTreeClassifier


class RandomForest:

    def __init__(self, n_estimators=3, max_depth=3, max_features=3, min_leaf_samples=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_leaf_samples = min_leaf_samples
        self.trees = []
        self.classes = None


    def predict(self, X):
        proba_mean = self.predict_proba(X)
        return proba_mean.argmax(axis=1)

    def predict_proba(self, X):
        k, c = X.shape[0], len(self.classes)
        proba_sum = np.zeros(shape=(k, c))
        for clf in self.trees:
            proba_sum += clf.predict_proba(X)

        return proba_sum / self.n_estimators

    def fit(self, X, y):
        self.trees = []
        n = X.shape[0]
        self.classes = np.unique(y)
        for _ in range(self.n_estimators):
            ids = np.random.choice(n, size=n)
            X_boot = X[ids]
            y_boot = y[ids]
            clf = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_leaf_samples=self.min_leaf_samples
            )
            clf.fit(X_boot, y_boot)
            self.trees.append(clf)

