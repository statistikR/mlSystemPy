'''
Created on Dec 11, 2013

Acknowledgement: Some of this code is adapted from code found in repository: 
https://github.com/luispedro/BuildingMachineLearningSystemsWithPython

@author: micha
'''

import math
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np


## zstandardize
def zStandardize(X):
    zX = X.copy()
    zX -= zX.mean(axis=0)
    zX /= zX.std(axis=0)
    return zX
    

## euclidian distance calculator

def distance(p0,p1):
    'compute square euclidian'
    return np.sum((p1-p0)**2)
    
# nn classifier that returns % correct classifications based on cross validation

def nn_classify_crossVal(features, labels, folds):
    ''' this is a nearest neighbor classifier that takes X and y 
        and the number of folds for the cross-validation and it
        returns the percent of correct classification based on the
        cross validation'''

    # creates bin for each iteration, observations are randomly assigned to bins
    n = int(math.ceil(features.shape[0]*1.0/folds))
    ccVector = np.array(((range(folds))*n)[:len(features)])
    ccVector = np.random.permutation(ccVector)
    
    # list that stores boolean for each prediction
    predCorr=[]

    # loop through the different fold bins
    # use all the observations not in bin as training dataset and 
    # observations in the bin for cross validation
    for it in range(0,folds):   
    
        # create trainings and validation datasets based on bin
        trainFeat = features[ccVector!=it]
        trainLab = labels[ccVector!=it]
        valFeat = features[ccVector==it]
        valLab = labels[ccVector==it]
        
        # run nearest neighbor classification for each observation in the val dataset
        
        for obs in range(0,len(valFeat)):
            
            dists = np.array([distance(t,valFeat[obs]) for t in trainFeat])
            nearest = dists.argmin()
            
            # store boolean indicating whether classification is correct in predCorr list
            predCorr.append(valLab[obs]== trainLab[nearest])
            
    return sum(predCorr)*1.0/len(predCorr)
            
