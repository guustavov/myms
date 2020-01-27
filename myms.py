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

def runSoftmax1(hiddenLayers):
    modelSuffix = '_{}'.format(str(hiddenLayers + 1))

    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/binary_total_undersampled/folds-1-percent/', hiddenLayers, modelSuffix)
    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/binary_total_undersampled/folds-10-percent/', hiddenLayers, modelSuffix)
    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/binary_total_undersampled/folds/', hiddenLayers, modelSuffix)

def runSoftmax2(hiddenLayers):
    modelSuffix = '_{}'.format(str(hiddenLayers + 1))

    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/binary_total_oversampled/folds-1-percent/', hiddenLayers, modelSuffix)
    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/binary_total_oversampled/folds-10-percent/', hiddenLayers, modelSuffix)
    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/binary_total_oversampled/folds/', hiddenLayers, modelSuffix)

def runSoftmax3(hiddenLayers):
    modelSuffix = '_{}'.format(str(hiddenLayers + 1))

    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/binary_23f_undersampled/folds-1-percent/', hiddenLayers, modelSuffix)
    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/binary_23f_undersampled/folds-10-percent/', hiddenLayers, modelSuffix)
    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/binary_23f_undersampled/folds/', hiddenLayers, modelSuffix)

def runSoftmax4(hiddenLayers):
    modelSuffix = '_{}'.format(str(hiddenLayers + 1))

    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/binary_23f_oversampled/folds-1-percent/', hiddenLayers, modelSuffix)
    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/binary_23f_oversampled/folds-10-percent/', hiddenLayers, modelSuffix)
    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/binary_23f_oversampled/folds/', hiddenLayers, modelSuffix)

def runSoftmax5(hiddenLayers):
    modelSuffix = '_{}'.format(str(hiddenLayers + 1))

    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/multiclass_total_imbalanced/folds-1-percent/', hiddenLayers, modelSuffix)
    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/multiclass_total_imbalanced/folds-10-percent/', hiddenLayers, modelSuffix)
    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/multiclass_total_imbalanced/folds/', hiddenLayers, modelSuffix)

def runSoftmax6(hiddenLayers):
    modelSuffix = '_{}'.format(str(hiddenLayers + 1))

    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/multiclass_23f_imbalanced/folds-1-percent/', hiddenLayers, modelSuffix)
    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/multiclass_23f_imbalanced/folds-10-percent/', hiddenLayers, modelSuffix)
    cv.run(SoftmaxANN(), '/home/wzalewski/gustavo/cicids2017/multiclass_23f_imbalanced/folds/', hiddenLayers, modelSuffix)

if __name__ == '__main__':
    pool_binary_total_under = Pool(5)
    pool_binary_total_over = Pool(5)
    pool_binary_23_under = Pool(5)
    pool_binary_23_over = Pool(5)
    pool_multi_total = Pool(5)
    pool_multi_23 = Pool(5)

    out1 = pool_binary_total_under.map(runSoftmax1, range(0, 5))
    out2 = pool_binary_total_over.map(runSoftmax2, range(0, 5))
    out3 = pool_binary_23_under.map(runSoftmax3, range(0, 5))
    out4 = pool_binary_23_over.map(runSoftmax4, range(0, 5))
    out5 = pool_multi_total.map(runSoftmax5, range(0, 5))
    out6 = pool_multi_23.map(runSoftmax6, range(0, 5))

    print(out1)
    print(out2)
    print(out3)
    print(out4)
    print(out5)
    print(out6)
