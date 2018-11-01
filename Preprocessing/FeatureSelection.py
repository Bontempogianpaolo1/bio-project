
from sklearn.feature_selection import SelectKBest,f_classif


class FeatureSelection:
    def __init__(self):
        self.start = 0
        self.stop = 0
        self.interval = 0
        self.lca = None
        self.name="feature selection"

    def clear(self, data):
        return

    def setcomponents(self, i):
        self.lca = SelectKBest(f_classif, k=i)
        self.lca.n_components = i

    def fit(self, X, y):
        self.lca.fit(X, y)

    def transform(self, x_tr):
        return self.lca.transform(x_tr)

    def get(self):
        return self.lca
