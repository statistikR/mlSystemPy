
"""
autor:         Micha Segeritz (the.statistikR@gmail.com)
first created: 2013-12-20
program name:  _1_topicModeling.py
original path: /Users/micha/GoogleDrive/WORKSPACE/travaille/dataScience/mlSystemPy/ch4topicModeling/_1_topicModeling.py

program function:
-----------------

data is downloaded from here:  http://www.cs.princeton.edu/~blei/lda-c/ap.tgz


edits:
------

date:        comment:
-----        --------


"""

#############################################
# Import Python Libraries:

from gensim import corpora, models, similarities
import os
import numpy as np
from scipy.spatial import distance

#############################################
# Set User Parameters:

DATADIR = "/Users/micha/GoogleDrive/WORKSPACE/data/production/privateProj/pyMLSy/ap"
numTopics = 100
#############################################

# read 2246 documents, word list
corpus = corpora.BleiCorpus(os.path.join(DATADIR,"ap.dat"),os.path.join(DATADIR,"vocab.txt"))
docs =  [open(os.path.join(DATADIR,"ap.txt")).read()]

#print docs.readlines()
         
# creates a topic model with 100 topics based on the corpus and the documents                           
model = models.ldamodel.LdaModel(corpus,num_topics= numTopics,id2word=corpus.id2word)

# create a list element for each document. check to which topic a document belongs to (topics may belong to several topics with different intensity)
documents = [model[c] for c in corpus]

# create a distance matrix

dense = np.zeros((len(documents), numTopics), float)

# loop through documents (ti) and extract list of tuples with topic and weight (t)
# then loop through topics for each document
# then add weight (v) in document, topic matrix (dense)
for ti, t in enumerate(documents):
    # print "ti and t: " + str(ti)+ " -" + str(t)
    for tj, v in t:
        # print "tj and v: " + str(tj)+ " -" + str(v)
        dense[ti,tj] = v


# convert document*topic matrix in a document * document matrix with pairwise distances

pairwise = distance.squareform(distance.pdist(dense))


# set diagonal to max + 1

largest = pairwise.max()
for ti in range(len(documents)):
    pairwise[ti,ti]=largest + 1
    


# write function that returns the closest document for each document

def closest_to(doc_id):
    return "id: " + str(pairwise[doc_id].argmin()) + ", distance: " + str(pairwise[doc_id].min())
 
# return which document is closest to a specific document in terms of topics        
print(closest_to(2))
print(pairwise[2].tolist())
print(pairwise[pairwise[2].argmin()].tolist())


