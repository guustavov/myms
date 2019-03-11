from auxiliaryLog import auxlog
from multiprocessing import Process, Pool, TimeoutError
import sys, time, os
import pandas as pd

from hybrid import OriginalHybrid

from sklearn.datasets.samples_generator import make_classification
from sklearn import metrics

def multiRunHybrid(args):
    runHybrid(*args)

def runHybrid(x, Y, superiorLimit, inferiorLimit):
    hybrid = OriginalHybrid(x, Y, superiorLimit, inferiorLimit)
    hybridPredictions = hybrid.predict(x, Y)
    accuracy_score = metrics.accuracy_score(Y, hybridPredictions)
    precision_score = metrics.precision_score(Y, hybridPredictions)
    recall_score = metrics.recall_score(Y, hybridPredictions)
    f1_score = metrics.f1_score(Y, hybridPredictions)
    confusion_matrix = metrics.confusion_matrix(Y, hybridPredictions)
    print(accuracy_score)
    print(precision_score)
    print(recall_score)
    print(f1_score)
    print(confusion_matrix)

if __name__ == '__main__':
    pool = Pool(processes=8)
    
    x, Y = make_classification(n_samples=100, n_features=23, n_classes=2)

    allHybridConfigs = [
        (x, Y, 0, 25),
        (x, Y, 0, 50),
        (x, Y, 0, 75),
        (x, Y, 0, 100),
        (x, Y, 25, 25),
        (x, Y, 25, 50),
        (x, Y, 25, 75),
        (x, Y, 25, 100)
    ]

    pool.map(multiRunHybrid, allHybridConfigs)


    