import numpy as np
import pandas as pd
import fileUtils as f
import glob
import auxiliaryLog
from sklearn.model_selection import StratifiedKFold
from hybrid import OriginalHybrid

import os, os.path

def createFolds(datasetPath, foldsPath):
    dataset = pd.read_csv(datasetPath)

    f.createDirectory(foldsPath)

    classFeatureName = dataset.columns[len(dataset.columns) - 1]

    # removing all instances that have no class value
    dataset = dataset.dropna(subset=[classFeatureName])

    dataset = binarizeDataset(dataset, classFeatureName)

    folds = splitDataset(dataset, classFeatureName, 10)

    # using only 10% of the original dataset
    folds = splitDataset(folds[0], classFeatureName, 10)

    for index, fold in enumerate(folds):
        f.saveFoldToCsv(fold, index, foldsPath)

def binarizeDataset(dataset, classFeatureName):
    benignFilter = dataset[classFeatureName] == "BENIGN"
    notBenignFilter = dataset[classFeatureName] != "BENIGN"

    dataset.loc[benignFilter, classFeatureName] = 0
    dataset.loc[notBenignFilter, classFeatureName] = 1
    
    return dataset

def splitDataset(dataset, classFeatureName, numberOfSplits = 10):
    names = dataset.columns

    x = dataset.drop([classFeatureName], axis = 1) # all instances with no class feature
    y = getattr(dataset, classFeatureName).values # class feature of all instances

    splitter = StratifiedKFold(numberOfSplits)

    folds = []
    for indexes in splitter.split(x, y):
        folds.append(pd.DataFrame(dataset.values[indexes[1],], columns = names))

    return folds

def run(model, foldsPath):
	numberOfFolds = len([name for name in os.listdir(foldsPath) if os.path.isfile(os.path.join(foldsPath, name))])
	if (isinstance(model, OriginalHybrid)):
		for iteration in range(0,numberOfFolds):
			trainFolds = glob.glob(foldsPath + 'fold_[!' + str(iteration) + ']*.csv')
			trainData = pd.concat((pd.read_csv(fold) for fold in trainFolds))
			testData = pd.read_csv(foldsPath + "fold_" + str(iteration) + ".csv")
			print(str(trainData.shape) + ' ' + str(testData.shape))