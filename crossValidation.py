import numpy as np
import pandas as pd
import fileUtils as f
import glob
import auxiliaryLog
from sklearn.model_selection import StratifiedKFold
from hybrid import OriginalHybrid
from ann import SoftmaxANN
from sklearn import metrics

import os, os.path

def createFolds(datasetPath, foldsPath):
    dataset = pd.read_csv(datasetPath)

    f.createDirectory(foldsPath)

    classFeatureName = dataset.columns[len(dataset.columns) - 1]

    # removing all instances that have no class value
    dataset = dataset.dropna(subset=[classFeatureName])

    dataset = binarizeDataset(dataset, classFeatureName)

    folds = splitDataset(dataset, classFeatureName, 10)

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

	modelClassName = (model.__class__.__name__) 
					#+ str(model.superiorLimit) + "_" 
					#+ str(model.inferiorLimit))
	pathToPersistModels = foldsPath + modelClassName + '/'

	if (isinstance(model, SoftmaxANN)):
		for iteration in range(0,numberOfFolds):
			if (os.path.isfile(pathToPersistModels + 'results/result_' + str(iteration))):
				auxiliaryLog.log('skipped ' + modelClassName + ' [iteration ' + str(iteration) + ']')
				continue

			trainFolds = glob.glob(foldsPath + 'fold_[!' + str(iteration) + ']*.csv')
			trainData = pd.concat((pd.read_csv(fold) for fold in trainFolds))
			testData = pd.read_csv(foldsPath + "fold_" + str(iteration) + ".csv")

			train_x, train_Y = splitXY(trainData)
			test_x, test_Y = splitXY(testData)

			history = model.fit(train_x, train_Y)
			
			print(history.history['acc'])
			print(history.history['loss'])

			f.createDirectory(pathToPersistModels)
			f.saveModelToFile(model, pathToPersistModels, iteration)
			# f.saveModelToFile(model.knn.model, pathToPersistModels, iteration)

			predictions = model.predict(test_x)

			print(predictions)

			accuracy_score = metrics.accuracy_score(test_Y, predictions)
			precision_score = metrics.precision_score(test_Y, predictions)
			recall_score = metrics.recall_score(test_Y, predictions)
			f1_score = metrics.f1_score(test_Y, predictions)
			confusion_matrix = metrics.confusion_matrix(test_Y, predictions)
                        
			auxiliaryLog.log("performed prediction of " + modelClassName + "[iteration" + str(iteration) + "]")
			result = ('acc: ' + str(accuracy_score)
				+ '\npre: ' + str(precision_score)
				+ '\nrec: ' + str(recall_score)
				+ '\nf1: ' + str(f1_score)
				+ '\nreclassified: ' + str(model.percentageOfReclassified) + '%'
				+ '\n' + 'matrix: ' + str(confusion_matrix))

			f.saveResultToFile(result, pathToPersistModels + "results_softmax/", iteration)
	
def splitXY(data):
	if(isinstance(data, pd.DataFrame)):
		data = data.values
	return data[:, :-1], data[:, -1]
