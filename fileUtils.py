import pickle, datetime, os, auxiliaryLog
import numpy as np
from hybrid import OriginalHybrid

def saveIterationArtifactsToFile(artifacts, path, cvIteration):
	model = artifacts['model']
	history = artifacts['history']
	confusion_matrix = artifacts['confusion_matrix']

	modelName = model.getName()

	if(isinstance(model, OriginalHybrid)):
		model = model.ann.model
	else:
		model = model.model

	path = '{}{}/'.format(path, str(cvIteration))
	createDirectory(path)

	# persist model
	modelFullPath = '{}{}'.format(path, 'pickled{}'.format(modelName))
	pickle.dump(model, open(modelFullPath, 'wb'))

	# persist training history
	if not history == '':
		historyFullPath = '{}{}'.format(path, 'pickledTrainingHistory')
		pickle.dump(history.history, open(historyFullPath, 'wb'))

	# persist confusion matrix
	confusionMatrixFullPath = '{}{}'.format(path, 'confusion_matrix.csv')
	confusion_matrix.to_csv(confusionMatrixFullPath)

	auxiliaryLog.log('Artifacts for {} [iteration {}] saved'.format(modelName, cvIteration))

def saveFoldToCsv(fold, foldIndex, destinationPath):
	fileName = "fold_" + str(foldIndex) + ".csv"
	fold.to_csv(destinationPath + fileName, index = False)
	auxiliaryLog.log(fileName + ' saved')

def saveResultToFile(result, path):
	createDirectory(path)
	
	fileName = '{}{}'.format(path, 'result.csv')
	result.to_csv(fileName)

	auxiliaryLog.log(fileName + ' saved')

def createDirectory(directoryPath):
	if not os.path.exists(directoryPath):
		os.makedirs(directoryPath)