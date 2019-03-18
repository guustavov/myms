import pickle, datetime, os, auxiliaryLog

def saveModelToFile(model, path, cvIteration, prefix = ''):
    createDirectory(path)

    fileName = path + prefix + str(cvIteration)
    pickle.dump(model, open(fileName, 'wb'))
    auxiliaryLog.log(fileName + ' saved')

def createDirectory(directoryPath):
    if not os.path.exists(directoryPath):
        os.makedirs(directoryPath)