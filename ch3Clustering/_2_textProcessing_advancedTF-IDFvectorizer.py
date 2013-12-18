
"""
author:         Micha Segeritz (the.statistikR@gmail.com)
first created: 2013-12-13
program name:  _2_advancedTF-IDFvectorizer.py
original path: /Users/micha/GoogleDrive/WORKSPACE/travaille/dataScience/mlSystemPy/ch3Custering/_2_advancedTF-IDFvectorizer.py

program function:
-----------------
testing the advanced vectorizer from _util_txtProcessing

edits:
------

date:        comment:
-----        --------


"""

#############################################
# Import Python Libraries:

import _util_txtProcessing as tp
import os
import numpy as np

#############################################
# Set User Parameters:

DATADIR = "/Users/micha/GoogleDrive/WORKSPACE/travaille/dataScience/mlSystemPy/gitRepo/BuildingMachineLearningSystemsWithPython/ch03/data/toy"


#############################################

# read in posts from disk
posts = [open(os.path.join(DATADIR,f)).read() for f in os.listdir(DATADIR)]
print posts

X = tp.vectorizer.fit_transform(posts) # analyze posts
print tp.vectorizer.get_feature_names() # look at terms found after deleting stop word and stemming 
print(X) # sparse matrix
Xfull = X.toarray().transpose() # full matrix
print(Xfull)

# combining TF-IDF matrix with terms

y = np.array(tp.vectorizer.get_feature_names()).reshape(17,1)
Xy = np.hstack((Xfull,y))


print Xy


