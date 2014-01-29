
"""
author:         Micha Segeritz (the.statistikR@gmail.com)
first created: 2014-01-27
program name:  _1_userRecommendations.py
original path: /Users/micha/GoogleDrive/WORKSPACE/travaille/dataScience/mlSystemPy/ch7RegRecomm/_1_userRecommendations.py

Acknowledgement: Some of this code is adapted from code found in repository: 
https://github.com/luispedro/BuildingMachineLearningSystemsWithPython

program function:
-----------------


input data downloaded with bash script downloadML100data.sh


edits:
------

date:        comment:
-----        --------


"""

#############################################
# Import Python Libraries:

import numpy as np
from scipy import sparse
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.cross_validation import KFold
import os

#############################################
# Set User Parameters:

DATA_DIR ="/Users/micha/GoogleDrive/WORKSPACE/data/production/privateProj/pyMLSy/movieRatingDB/ml-100k"

#############################################

# data structure: user id | item id | rating | timestamp (data in random order) --> see README
# 943 users on 1682 items (each user has rated at least 20 movies)
filename = os.path.join(DATA_DIR, "u.data")
data = np.array([[int(tok) for tok in line.split('\t')[:3]]
                for line in open(filename)])


# subset dataset to user - movies
ij = data[:, :2]
ij -= 1  # original data is in 1-based system

#ratings
values = data[:, 2]


# convert to a large sparse matrix (user are rows, movies are columns)
reviews = sparse.csc_matrix((values, ij.T)).astype(float)


''' normalize standardize a matrix 
        - set missings to 0
        - subtract row mean from all non missing row values
        - return xc: matrix showing deviation from row mean
                 x1: row means
'''
def movie_norm(xc):
    xc = xc.copy().toarray()
    x1 = np.array([xi[xi > 0].mean() for xi in xc]) #loop through rows and calculate row means
    x1 = np.nan_to_num(x1)   # replace missing ratings with 0

    for i in range(xc.shape[0]):          # loop through ratings/rows
        xc[i] -= (xc[i] > 0) * x1[i]      # subtract mean if rating not missing/0
    return xc, x1

u = reviews[700]  # shows ratings for a user of all movies s/he has seen
us = np.delete(np.arange(reviews.shape[0]), 700)  # array with all user ids but user i
ps, = np.where(u.toarray().ravel() > 0)



reg = ElasticNetCV(fit_intercept=True, alphas=[
                   0.0125, 0.025, 0.05, .125, .25, .5, 1., 2., 4.])

'''
    develop a prediction model: predicting ratings for movies a user has seen 
    based on ratings of other users who have seen the same movie
    
    returns: SE for prediction and SE if we had used a users mean to predict each movie (null model)
    
    additional features: 
        - apply this model to a set of movies of the user not used for training (cv
            how the model would perform for new movies)
        - using CV class for Elastic Nets; it performs CV across several sample
            parameters for alpha --> inner fold CV
'''

def learn_for(i):
    u = reviews[i]  # shows ratings for a user of all movies s/he has seen
    us = np.delete(np.arange(reviews.shape[0]), i)  # array with all user ids but user i
    ps, = np.where(u.toarray().ravel() > 0)         # flattened array with all the movies user i has seen
    x = reviews[us][:, ps].T           # create X with all the users but user i and all their movie rating that user has rated
    y = u.data                         # all ratings of user i
    err = 0
    eb = 0
    kf = KFold(len(y), n_folds=4)
    for train, test in kf:
        xc, x1 = movie_norm(x[train]) # return matrix showing deviations for each rater from average rating for movie     
        reg.fit(xc, y[train] - x1)    # predict rating of user i of all the movies s/he has seen based on ratings of all ohter people

        xc, x1 = movie_norm(x[test])  # normalize dataset that was held back
        p = np.array([reg.predict(xi) for xi in xc]).ravel()
        e = (p + x1) - y[test]
        err += np.sum(e * e)       # sum of squared errors
        eb += np.sum((y[train].mean() - y[test]) ** 2)  # sum of square errors if we had used 
    return np.sqrt(err / float(len(y))), np.sqrt(eb / float(len(y))) 

'''
    loop through all users and estimate how well we can predict their movie taste
    base on a cv test dataset of movie ratings.
    Calculate for each user if we are better in predicting their taste compared to
    Null Model (using user mean to predict their taste)
'''

whole_data = []
for i in range(reviews.shape[0]):
    s = learn_for(i)
    print(s[0] < s[1])
    print s
    whole_data.append(s)
    
print whole_data
print whole_data[0]
print len(whole_data)


