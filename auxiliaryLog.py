import datetime
import inspect

class auxlog:
    @staticmethod
    def log(message):
        print('[' + str(datetime.datetime.now()).split('.')[0] + '] ' + message)