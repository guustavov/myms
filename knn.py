from sklearn.neighbors import KNeighborsClassifier

class OriginalKNN:
    def __init__(self):
        self.k_neighbors = 1

    def fit(self, x, Y):
        self.model = KNeighborsClassifier(self.k_neighbors, weights='uniform', algorithm='brute')
        self.model.fit(x, Y)

    def predict(self, x):
        return self.model.predict(x)
