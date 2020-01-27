import numpy as np
import pandas as pd
import fileUtils as f
import glob
import auxiliaryLog
from sklearn.model_selection import StratifiedKFold
from hybrid import OriginalHybrid
from ann import SoftmaxANN
from sklearn import metrics, preprocessing

from sklearn.preprocessing import Normalizer


import os, os.path
import pickle

def undersample(dataset, classFeatureName):
	# shuffle the dataset
	shuffled_df = dataset.sample(frac=1, random_state=4)
	
	# put all the attack class in a separate dataset
	notBenignFilter = dataset[classFeatureName] != 'BENIGN'
	attack_df = shuffled_df.loc[notBenignFilter]

	# randomly select same number of attack observations from the benign (majority class)
	benignFilter = dataset[classFeatureName] == 'BENIGN'
	benign_df = shuffled_df.loc[benignFilter].sample(n=len(attack_df.index), random_state=42)

	# concatenate both dataframes again
	normalized_df = pd.concat([attack_df, benign_df])

	return normalized_df

def createFolds(datasetPath, foldsPath):
	folds1Path = '{}folds-1-percent/'.format(foldsPath)
	folds10Path = '{}folds-10-percent/'.format(foldsPath)
	foldsPath = '{}folds/'.format(foldsPath)

	f.createDirectory(foldsPath)
	f.createDirectory(folds10Path)
	f.createDirectory(folds1Path)

	dataset = pd.read_csv(datasetPath)
	classFeatureName = dataset.columns[len(dataset.columns) - 1]

	dataset.drop(dataset.columns[[0]], axis=1, inplace=True)

	folds = splitDataset(dataset, classFeatureName, 10)
	folds10 = splitDataset(folds[0], classFeatureName, 10)
	folds1 = splitDataset(folds10[0], classFeatureName, 10)

	for index, fold in enumerate(folds):
		f.saveFoldToCsv(fold, index, foldsPath)

	for index, fold in enumerate(folds10):
		f.saveFoldToCsv(fold, index, folds10Path)

	for index, fold in enumerate(folds1):
		f.saveFoldToCsv(fold, index, folds1Path)

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

def run(model, foldsPath, hiddenLayers, appendToModelName=''):
	numberOfFolds = len([name for name in os.listdir(foldsPath) if os.path.isfile(os.path.join(foldsPath, name))])

	modelClassName = model.getName()

	pathToPersistModels = foldsPath + modelClassName + appendToModelName + '/'

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

		# Shuffle all data before training/test
		trainData = trainData.sample(frac=1).reset_index(drop=True)
		testData = testData.sample(frac=1).reset_index(drop=True)

		train_x, train_Y = splitXY(trainData)
		test_x, test_Y = splitXY(testData)

		labels = np.unique(train_Y)

		train_Y = transformLabelToOrdinal(train_Y)
		test_Y = transformLabelToOrdinal(test_Y)

		# Normalization of both train and test data
		scaler = Normalizer().fit(train_x)
		train_x = np.array(scaler.transform(train_x))
		scaler = Normalizer().fit(test_x)
		test_x = np.array(scaler.transform(test_x))

		history = model.fit(train_x, train_Y, hiddenLayers)

		predictions = model.predict(test_x)
		auxiliaryLog.log('performed prediction of {} [iteration {}]'.format(modelClassName, str(iteration)))

		report = pd.DataFrame(metrics.classification_report(test_Y, predictions, output_dict=True)).transpose()
		
		resultMetrics = {
			'acc': metrics.accuracy_score(test_Y, predictions),
			'pre': metrics.precision_score(test_Y, predictions, average=None),
			'rec': metrics.recall_score(test_Y, predictions, average=None),
			'f1': metrics.f1_score(test_Y, predictions, average=None)
		}

		if isinstance(model, OriginalHybrid):
			resultMetrics['reclassified'] = '{}%'.format(model.percentageOfReclassified)

		resultMetricsDataFrame = resultMetricsDataFrame.append(resultMetrics, ignore_index=True)
		confusion_matrix = pd.DataFrame(metrics.confusion_matrix(test_Y, predictions), columns=labels, index=labels)

		iterationArtifacts = {
			'model': model,
			'history': history,
			'confusion_matrix': confusion_matrix,
			'report': report
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
