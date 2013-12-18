'''
Created on Dec 12, 2013

@author: micha
'''

import scipy as sp
import os
import sys
import nltk

from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(min_df=1)

DATADIR = "/Users/micha/GoogleDrive/WORKSPACE/travaille/dataScience/mlSystemPy/gitRepo/BuildingMachineLearningSystemsWithPython/ch03/data/toy"

# simple sample comment to experiment
content = ["How to format my hard disk","Hard disk format problems"]

# analyze content and store in sparse matrix
X = vectorizer.fit_transform(content)
print(vectorizer.get_feature_names())

# convert sparse matrix to normal matrix
X = X.toarray().transpose()


## p53

# read in posts from disk
posts = [open(os.path.join(DATADIR,f)).read() for f in os.listdir(DATADIR)]
print posts

X_train = vectorizer.fit_transform(posts)
# look at samples and features and word contents
print "number of sample and features: " + str(X_train.shape)
print vectorizer.get_feature_names()

# read in new post, and apply "sparse matrix" vectorization to it, based on model "fitted above"
new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])


# calculate Euclidian Distance for similarity measure of this post with all other posts
def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

def dist_norm(v1, v2):
    v1 = v1 / sp.linalg.norm(v1.toarray())
    v2 = v2 / sp.linalg.norm(v2.toarray())
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

best_doc = None
best_dist = sys.maxint
best_i = None

for i in range(0,len(posts)):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_raw(post_vec,new_post_vec)
    print " Post %i with dist=%.2f: %s" %(i, d, post)
    
    if d<best_dist:
        best_dist = d
        best_i = i
        best_doc = post
        
print "Best Post is post %i with dist=%.2f" %(best_i, best_dist)
        


    



