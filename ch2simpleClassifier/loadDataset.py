'''
Created on Dec 9, 2013

Acknowledgement: Some of this code is adapted from code found in repository: 
https://github.com/luispedro/BuildingMachineLearningSystemsWithPython

@author: micha
'''
import numpy as np

def loadTabDataset(path,dataset_name):
    '''
    X,y = TabDataset(dataset_name,path)

    Load a tab delimited file and return
    X : numpy ndarray
    y : list of str
    '''
    data = []
    labels = []
    
    if(path[-1])!="/":
        path += "/"

    with open(path + dataset_name.format(dataset_name)) as ifile:
        for line in ifile:
            tokens = line.strip().split('\t')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels
