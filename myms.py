import auxiliaryLog
from multiprocessing import Process, Pool, TimeoutError
import sys, time, os
import pandas as pd
import fileUtils as f
import crossValidation as cv

from hybrid import OriginalHybrid

from sklearn.datasets.samples_generator import make_classification
from sklearn import metrics

def multiRunHybrid(args):
    runHybrid(*args)

def runHybrid(superiorLimit, inferiorLimit):
    originalHybrid = OriginalHybrid(superiorLimit, inferiorLimit)
    cv.run(originalHybrid, '/media/gstav/Data/github/cicids2017/1-percent/')

if __name__ == '__main__':
    pool = Pool(processes=8)
    
    # x, Y = make_classification(n_samples=100, n_features=23, n_classes=2)
    # allHybridConfigs = [
    #     (x, Y, 0, 25),
    #     (x, Y, 0, 50),
    #     (x, Y, 0, 75),
    #     (x, Y, 0, 100),
    #     (x, Y, 25, 25),
    #     (x, Y, 25, 50),
    #     (x, Y, 25, 75),
    #     (x, Y, 25, 100)
    # ]


    allHybridConfigs = [
        (0, 25),
        (0, 50),
        (0, 75),
        (0, 100),
        (25, 25),
        (25, 50),
        (25, 75),
        (25, 100)
    ]

    pool.map(multiRunHybrid, allHybridConfigs)