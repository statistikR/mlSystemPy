
"""
autor:         Micha Segeritz (the.statistikR@gmail.com)
first created: 2013-12-13
program name:  _3_clustering.py
original path: /Users/micha/GoogleDrive/WORKSPACE/travaille/dataScience/mlSystemPy/ch3Custering/_3_clustering.py

program function:
-----------------
This program applies clustering algorithms to newsgroup posts
(see Richerts & Coelho, 2013, p66f)

edits:
------

date:        comment:
-----        --------


"""

#############################################
# Import Python Libraries:

import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import _util_txtProcessing
from sklearn.cluster import KMeans
from sklearn import metrics
import scipy as sp

#############################################
# Set User Parameters:

MLCOMP_DIR = "/Users/micha/GoogleDrive/WORKSPACE/data/production/privateProj/mlcomp"

#############################################
# read all data
#data = sklearn.datasets.load_mlcomp("20news-18828", mlcomp_root = MLCOMP_DIR)

# read test and train dataset for a subset of the newsgroups
groups = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.ma c.hardware', 'comp.windows.x', 'sci.space']

train_data = sklearn.datasets.load_mlcomp("20news-18828", "train", 
                                          mlcomp_root = MLCOMP_DIR,categories = groups)
train_labels = train_data.target

test_data = sklearn.datasets.load_mlcomp("20news-18828", "test", 
                                          mlcomp_root = MLCOMP_DIR,categories = groups)

test_labels = test_data.target

#############################################
# create vectorizer
vectorizer = _util_txtProcessing.StemmedTfidfVectorizer(min_df = 10, max_df = 0.5,
                                    stop_words='english', decode_error='ignore')

## vectorize train_data.data 
vectorized = vectorizer.fit_transform(train_data.data)
print "dimensions of vectorized data:" + str(vectorized.shape)

## setup Kmeans to extract 50 clusters
km = KMeans(n_clusters=50, init='k-means++', n_init=1,
            verbose=1)

## create clustering of trainings dataset
clustered = km.fit(vectorized)

#############################################
## feed new post to clustering algorithm and find cluster

# extract one post
newPost = test_data.data[10]
print newPost
# vectorize post
vectorizedNewPost = vectorizer.transform([newPost])

# apply km to new post
kmPredNewPost = km.predict(vectorizedNewPost)
print type(kmPredNewPost)
# extract label
new_post_label = kmPredNewPost[0]

##############################################################
# find post within the same cluster that is the most similar

# extract indices of post of original clustering dataset with same cluster label
indices = (km.labels_==new_post_label).nonzero()[0] # creates array of indices
print indices

# calculate distance of post at hand to every other post in same cluster

similar = []
for i in indices:
    dist = sp.linalg.norm((vectorizedNewPost - vectorized[i]).toarray())
    similar.append((dist, train_data.data[i]))

# find out how many similar posts are there
print len(similar)

similar = sorted(similar)

show_at_1 = similar[0]
show_at_2 = similar[len(similar) / 2]
show_at_3 = similar[-1]

print("=== #1 ===")
print(show_at_1)
print()

print("=== #2 ===")
print(show_at_2)
print()

print("=== #3 ===")
print(show_at_3)


############################
print("look at some metrics of classification via clustering")
print("(most of these methods look at pre-defined labels vs. cluster labels found during clustering)")

print("Homogeneity: %0.3f" % metrics.homogeneity_score(train_labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(train_labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(train_labels, km.labels_))
print("Adjusted Rand Index: %0.3f" %
      metrics.adjusted_rand_score(train_labels, km.labels_))
print("Adjusted Mutual Information: %0.3f" %
      metrics.adjusted_mutual_info_score(train_labels, km.labels_))
print(("Silhouette Coefficient: %0.3f" %
       metrics.silhouette_score(vectorized, train_labels, sample_size=1000)))




