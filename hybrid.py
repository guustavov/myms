# to ensure true division in python 2
from __future__ import division

import numpy as np
import pandas as pd
from ann import OriginalANN
from knn import OriginalKNN

class OriginalHybrid(object):
    def __init__(self, superiorLimit, inferiorLimit):
        self.superiorLimit = superiorLimit
        self.inferiorLimit = inferiorLimit

    def fit(self, train_x, train_Y):
        self.ann = OriginalANN()
        self.knn = OriginalKNN()

        self.ann.fit(train_x, train_Y)
        self.knn.fit(train_x, train_Y)

    def predict(self, test_x):
        annOutput = self.ann.predict(test_x)
        np.set_printoptions(threshold=np.inf)
        annPredictions = self.ann.predict_classes(test_x)

        hybridPredictions = annPredictions

        # get indexes of the elements to reclassify with knn
        # ravel used to transform Nd to 1d
        mask = self.getIntermediateRangeMask(annOutput).ravel()
        indexesOfElementsToReclassify = np.where(mask)[0]

        knn_test_x = test_x[indexesOfElementsToReclassify]  

        self.percentageOfReclassified = (len(indexesOfElementsToReclassify) / len(test_x)) * 100
        
        if(self.percentageOfReclassified > 0):
            knnPredictions = self.knn.predict(knn_test_x)
            
            self.replaceNewPredictions(hybridPredictions, knnPredictions, indexesOfElementsToReclassify)

        return hybridPredictions

    def getIntermediateRangeMask(self, predictions):
        # ann output segmented into positive and negative values
        positiveElements = predictions[predictions > 0]
        negativeElements = predictions[predictions < 0]

        # set upper and lower thresholds based on ann output
        if(len(positiveElements)):
            self.upperThreshold = np.percentile(positiveElements, self.superiorLimit)
        else:
            self.upperThreshold = 1
        if(len(negativeElements)):
            self.lowerThreshold = np.percentile(negativeElements, (100 - self.inferiorLimit))
        else:
            self.lowerThreshold = -1

        # mask to get elements contained in the intermediate range
        return (predictions <= self.upperThreshold) & (predictions >= self.lowerThreshold)

    def replaceNewPredictions(self, oldPredictions, newPredictions, newPredictionsIndexes):
        for (index, newPrediction) in zip(newPredictionsIndexes, newPredictions):
            oldPredictions[index] = newPrediction

    def getName(self):
        return self.__class__.__name__