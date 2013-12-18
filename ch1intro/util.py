'''
Created on Dec 4, 2013

@author: micha
'''

import os

path = "/Users/micha/GoogleDrive/WORKSPACE/travaille/dataScience/mlSystemPy/gitRepo/BuildingMachineLearningSystemsWithPython/ch01"

DATA_DIR = os.path.join(path, "data")

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "charts")

for d in [DATA_DIR, CHART_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)

