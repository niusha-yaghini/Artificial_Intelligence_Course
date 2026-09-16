class StandardScalerScratch:

    def fit(self, X):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)