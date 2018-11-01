from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class FeatureLca:
    def __init__(self):
        self.lca = None
        self.name = "lca"

    def clear(self, data):
        return

    def setcomponents(self, i):
        self.lca = LinearDiscriminantAnalysis(n_components=i, store_covariance=False, tol=0.0001)

    def fit(self, X, y):
        self.lca.fit(X, y)

    def transform(self, x_tr):
        return self.lca.transform(x_tr)

    def get(self):
        return self.lca