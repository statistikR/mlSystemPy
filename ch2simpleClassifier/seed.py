'''
Created on Dec 9, 2013

Acknowledgement: Some of this code is adapted from code found in repository: 
https://github.com/luispedro/BuildingMachineLearningSystemsWithPython

@author: micha
'''
import loadDataset as load
import os
import classify


print(os.getcwd())

datadir = "/Users/micha/GoogleDrive/WORKSPACE/travaille/dataScience/mlSystemPy/gitRepo/BuildingMachineLearningSystemsWithPython/ch02/data"

features, labels = load.loadTabDataset(datadir,"seeds.tsv")


# classify seeds dataset and print % correct classifications 
    
print(classify.nn_classify_crossVal(features, labels, 10))

# zstandardize first and then classify

zFeatures = classify.zStandardize(features)
print(classify.nn_classify_crossVal(zFeatures, labels, 10))




