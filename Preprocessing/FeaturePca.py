from sklearn.decomposition import PCA


class FeaturePca:
    def __init__(self):
        self.pca = None
        self.name = "pca"

    def clear(self, data):
        return

    def setcomponents(self, i):
        del self.pca
        self.pca = PCA(n_components=i,random_state=241819)

    def fit(self, x_train_scaled,y):
        self.pca.fit(x_train_scaled)


    def transform(self, x_tr):
        return self.pca.transform(x_tr)

    def get(self):
        return self.pca
