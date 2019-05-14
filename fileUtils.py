import pickle, datetime, os, auxiliaryLog

def saveModelToFile(model, path, cvIteration):
    createDirectory(path)

    fileName = path + model.__class__.__name__ + "_" + str(cvIteration)
    pickle.dump(model, open(fileName, 'wb'))
    auxiliaryLog.log(fileName + ' saved')

def saveFoldToCsv(fold, foldIndex, destinationPath):
    fileName = "fold_" + str(foldIndex) + ".csv"
    fold.to_csv(destinationPath + fileName, index = False)
    auxiliaryLog.log(fileName + ' saved')

def saveResultToFile(result, path, cvIteration):
	createDirectory(path)

	fileName = path + "result_" + str(cvIteration)
	pickle.dump(result, open(fileName, 'wb'))
	auxiliaryLog.log(fileName + ' saved')

def createDirectory(directoryPath):
    if not os.path.exists(directoryPath):
        os.makedirs(directoryPath)