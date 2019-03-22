from sklearn.neighbors import KNeighborsClassifier

class OriginalKNN(object):
    def __init__(self):
        self.k_neighbors = 1

    def fit(self, x, Y):
        self.model = KNeighborsClassifier(self.k_neighbors, weights='uniform', algorithm='brute')
        self.model.fit(x, Y)

    def predict(self, test_x):
        return self.model.predict(test_x)

    def getName(self):
        return self.__class__.__name__