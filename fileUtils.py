import pickle, datetime, os

def saveModelToFile(model, path, cvIteration):
    createDirectory(path)

    fileName = path + str(cvIteration)
    pickle.dump(model, open(fileName, 'wb'))
    print('[' + str(datetime.datetime.now()).split('.')[0] + '] ' + fileName + ' saved [')

def createDirectory(directoryPath):
    if not os.path.exists(directoryPath):
        os.makedirs(directoryPath)