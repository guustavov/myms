import numpy as np
import pandas as pd
import fileUtils as f
import glob
import auxiliaryLog
from sklearn.model_selection import StratifiedKFold
from hybrid import OriginalHybrid
from ann import SoftmaxANN
from sklearn import metrics, preprocessing

import os, os.path
import pickle

def createFolds(datasetPath, foldsPath):
	dataset = pd.read_csv(datasetPath)
	
	multiclassPath = '{}multiclass/'.format(foldsPath)
	binaryPath = '{}binary/'.format(foldsPath)

	f.createDirectory(multiclassPath)
	f.createDirectory(binaryPath)

	classFeatureName = dataset.columns[len(dataset.columns) - 1]

	# removing all instances that have no class value
	dataset = dataset.dropna(subset=[classFeatureName])

	folds = splitDataset(dataset, classFeatureName, 10)

	for index, fold in enumerate(folds):
		f.saveFoldToCsv(fold, index, multiclassPath)

		binarizedFold = binarizeDataset(fold, classFeatureName)
		f.saveFoldToCsv(binarizedFold, index, binaryPath)

def binarizeDataset(dataset, classFeatureName):
	notBenignFilter = dataset[classFeatureName] != 'BENIGN'

	dataset.loc[notBenignFilter, classFeatureName] = 'ATTACK'
	
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

	modelClassName = model.getName()

	pathToPersistModels = foldsPath + modelClassName + '/'

	columns = ['acc', 'pre', 'rec', 'f1']
	if isinstance(model, OriginalHybrid):
		columns.append('reclassified')

	resultMetricsDataFrame = pd.DataFrame(columns=columns)

	for iteration in range(0,numberOfFolds):
		modelFileName = pathToPersistModels + str(iteration) + '/pickled' + modelClassName
		if (os.path.isfile(modelFileName)):
			with open(modelFileName) as pickledAnnModel:
				model.setAnnModel(pickle.load(pickledAnnModel))
			auxiliaryLog.log('skipped ' + modelClassName + ' [iteration ' + str(iteration) + ']')
		else:
			model.clearModel()

		trainFolds = glob.glob(foldsPath + 'fold_[!' + str(iteration) + ']*.csv')
		trainData = pd.concat((pd.read_csv(fold) for fold in trainFolds))
		testData = pd.read_csv(foldsPath + "fold_" + str(iteration) + ".csv")

		train_x, train_Y = splitXY(trainData)
		test_x, test_Y = splitXY(testData)

		labels = np.unique(train_Y)

		train_Y = transformLabelToOrdinal(train_Y)
		test_Y = transformLabelToOrdinal(test_Y)

		history = model.fit(train_x, train_Y)

		predictions = model.predict(test_x)
		auxiliaryLog.log('performed prediction of {} [iteration {}]'.format(modelClassName, str(iteration)))

		resultMetrics = {
			'acc': metrics.accuracy_score(test_Y, predictions),
			'pre': metrics.precision_score(test_Y, predictions),
			'rec': metrics.recall_score(test_Y, predictions),
			'f1': metrics.f1_score(test_Y, predictions)
		}

		if isinstance(model, OriginalHybrid):
			resultMetrics['reclassified'] = '{}%'.format(model.percentageOfReclassified)

		resultMetricsDataFrame = resultMetricsDataFrame.append(resultMetrics, ignore_index=True)
		confusion_matrix = pd.DataFrame(metrics.confusion_matrix(test_Y, predictions), columns=labels, index=labels)

		iterationArtifacts = {
			'model': model,
			'history': history,
			'confusion_matrix': confusion_matrix
		}

		f.saveIterationArtifactsToFile(iterationArtifacts, pathToPersistModels, iteration)

	f.saveResultToFile(resultMetricsDataFrame, pathToPersistModels)
	
def splitXY(data):
	if(isinstance(data, pd.DataFrame)):
		data = data.values
	return data[:, :-1], data[:, -1]

def transformLabelToOrdinal(Y):
	le = preprocessing.LabelEncoder()
	le.fit(np.unique(Y))
	return le.transform(Y)
