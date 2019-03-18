import datetime
import inspect

def log(message):
    print('[' + str(datetime.datetime.now()).split('.')[0] + '] ' + message)