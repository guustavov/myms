from auxiliaryLog import auxlog
from multiprocessing import Process, Pool, TimeoutError
import sys, time, os

import pandas as pd

def multiRunHybrid(args):
    runHybrid(*args)

def runHybrid(superiorLimit, inferiorLimit):
    print(superiorLimit)
    print(inferiorLimit)
    
if __name__ == '__main__':
    pool = Pool(processes=8)

    intermediateRanges = [(0,25),(0,50),(0,75),(0,100),(25,25),(25,50),(25,75),(25,100)]

    pool.map(multiRunHybrid, intermediateRanges)