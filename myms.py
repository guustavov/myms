import auxiliaryLog
from multiprocessing import Process, Pool, TimeoutError
import sys
import time
import os
import pandas as pd
import fileUtils as f
import crossValidation as cv

from hybrid import OriginalHybrid
from ann import SoftmaxANN

from sklearn.datasets.samples_generator import make_classification
from sklearn import metrics

from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping

def runSoftmax(hiddenLayers):
    softmax = SoftmaxANN()

    modelSuffix = '_{}'.format(str(hiddenLayers + 1))

    cv.run(softmax, '/home/gstav/projects/cicids2017/binary_23f_undersampled/test_folds/', hiddenLayers, modelSuffix)

# def multiRunHybrid(args):
#     runHybrid(*args)

# def runHybrid(superiorLimit, inferiorLimit):
#     originalHybrid = OriginalHybrid(superiorLimit, inferiorLimit)
#     cv.run(originalHybrid, '/home/gstav/projects/cicids2017/folds_10_percent/binary/')


if __name__ == '__main__':
    pool = Pool(5)

    out = pool.map(runSoftmax, range(0, 5))

    print(out)
