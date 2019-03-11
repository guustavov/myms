import numpy as np
import pandas as pd
from ann import OriginalANN
from knn import OriginalKNN

class OriginalHybrid:
    def __init__(self, train_x, train_Y, superiorLimit, inferiorLimit):
        self.superiorLimit = superiorLimit
        self.inferiorLimit = inferiorLimit

        self.ann = OriginalANN()
        self.knn = OriginalKNN()

        self.ann.fit(train_x, train_Y)
        self.knn.fit(train_x, train_Y)

    def predict(self, test_x, test_Y):
        annOutput = self.ann.predict(test_x)
        annPredictions = self.ann.predict_classes(test_x)

        hybridPredictions = annPredictions

        # get indexes of the elements to reclassify with knn
        # ravel used to transform Nd to 1d
        mask = self.getIntermediateRangeMask(annOutput).ravel()
        indexesOfElementsToReclassify = np.where(mask)[0]

        knn_test_x = test_x[indexesOfElementsToReclassify]

        self.numberOfReclassified = len(indexesOfElementsToReclassify)

        knnPredictions = self.knn.predict(knn_test_x)
        
        self.replaceNewPredictions(hybridPredictions, knnPredictions, indexesOfElementsToReclassify)

        return hybridPredictions

    def getIntermediateRangeMask(self, predictions):
        # ann output segmented into positive and negative values
        positiveElements = predictions[predictions > 0]
        negativeElements = predictions[predictions < 0]

        # set upper and lower thresholds based on ann output
        self.upperThreshold = np.percentile(positiveElements, self.superiorLimit)
        self.lowerThreshold = np.percentile(negativeElements, (100 - self.inferiorLimit))

        # mask to get elements contained in the intermediate range
        return (predictions <= self.upperThreshold) & (predictions >= self.lowerThreshold)

    def replaceNewPredictions(self, oldPredictions, newPredictions, newPredictionsIndexes):
        for (index, newPrediction) in zip(newPredictionsIndexes, newPredictions):
            oldPredictions[index] = newPrediction

    def getNumberOfReclassified(self):
        return self.numberOfReclassified