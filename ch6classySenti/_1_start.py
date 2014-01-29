
"""
author/adaptor:         Micha Segeritz (the.statistikR@gmail.com)
first created: 2014-01-04
program name:  ch6classySenti._1_start.py
original path: /Users/micha/GoogleDrive/WORKSPACE/travaille/dataScience/mlSystemPy/ch6classySenti/_1_start.py

Acknowledgement: Some of this code is adapted from code found in repository: 
https://github.com/luispedro/BuildingMachineLearningSystemsWithPython

program function:
-----------------

This is a good start for using a Naive Bayesian Classifier, luispedro's repository
provides some additional additional scripts ( 02_tuning.py, 03_clean.py and 04_sent.py)
with tweaks that make this base model quite a bit more sophisticated
02_tuning: - changing Ngram settings between unigrams(1,1), bigrams(1,2) and trigrams(1,3)
           - feature variations (exclude stop words, logarithmic word counts, etc.)
           - setting Laplace smoothing
           - applying grid search to loop through all the permutations
03_clean:  - cleaning tweets / improve feature extracts
04_sent:   - taking word type into account
           - using SentiWordNet (corpus?) to assign positive AND negative values to each word

edits:
------

date:        comment:
-----        --------


"""

#############################################
# Import Python Libraries:

import time

import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.cross_validation import ShuffleSplit

from utils import plot_pr
from utils import load_sanders_data
from utils import tweak_labels

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

#############################################
# Set User Parameters:

#############################################



start_time = time.time()

''' invoke a Tfidf (term frequency - inverse document frequency) vectorizer 
    and a Multinomial Naive Bayes Classifier
    join them and return a pipeline instance (that can use fit() and predict() like any classifier)
'''

def create_ngram_model():
    tfidf_ngrams = TfidfVectorizer(ngram_range=(1, 3),
                                   analyzer="word", binary=False)
    clf = MultinomialNB()
    pipeline = Pipeline([('vect', tfidf_ngrams), ('clf', clf)])
    return pipeline


def train_model(clf_factory, X, Y, name="NB ngram", plot=False):
    # setup cross-validation
    cv = ShuffleSplit(
        n=len(X), n_iter=10, test_size=0.3, indices=True, random_state=0)

    train_errors = []
    test_errors = []

    scores = []
    pr_scores = []
    precisions, recalls, thresholds = [], [], []

    # loop through cross-validation/test datasets and train/test the classifier
    for train, test in cv:
        # setup datasets
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]

        # invoke classifier (e.g., ngram_model) and fit data
        clf = clf_factory()
        clf.fit(X_train, y_train)

        # return correct prediction rate for cv datasets
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)


        # store errors
        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)

        scores.append(test_score)
        # returns pairs of probability for each observation of being 0 / 1
        proba = clf.predict_proba(X_test)

        # extract quality of model indicators
        fpr, tpr, roc_thresholds = roc_curve(y_test, proba[:, 1])
        precision, recall, pr_thresholds = precision_recall_curve(
            y_test, proba[:, 1])

        pr_scores.append(auc(recall, precision))
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(pr_thresholds)

    scores_to_sort = pr_scores
    median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]

    if plot:
        plot_pr(pr_scores[median], name, "01", precisions[median],
                recalls[median], label=name)

        summary = (np.mean(scores), np.std(scores),
                   np.mean(pr_scores), np.std(pr_scores))
        print "%.3f\t%.3f\t%.3f\t%.3f\t" % summary

    return np.mean(train_errors), np.mean(test_errors)


def print_incorrect(clf, X, Y):
    Y_hat = clf.predict(X)
    wrong_idx = Y_hat != Y
    X_wrong = X[wrong_idx]
    Y_wrong = Y[wrong_idx]
    Y_hat_wrong = Y_hat[wrong_idx]
    for idx in xrange(len(X_wrong)):
        print "clf.predict('%s')=%i instead of %i" %\
            (X_wrong[idx], Y_hat_wrong[idx], Y_wrong[idx])


if __name__ == "__main__":
    # load twitter posts and labels (pos, neg, neutral, irrelevant) in numpy arrays
    X_orig, Y_orig = load_sanders_data()
    
    # summarize the number of posts available in each category
    classes = np.unique(Y_orig)
    for c in classes:
        print "#%s: %i" % (c, sum(Y_orig == c))

    
    print "== Pos vs. neg =="
    # create boolean variable indicating if a post is positive or negative
    pos_neg = np.logical_or(Y_orig == "positive", Y_orig == "negative")

    # subset X, Y to only positive and negative posts
    X = X_orig[pos_neg]
    Y = Y_orig[pos_neg]

    # convert Y to a 0/1 nparray
    Y = tweak_labels(Y, ["positive"])


    train_model(create_ngram_model, X, Y, name="pos vs neg", plot=True)

    print "== Pos/neg vs. irrelevant/neutral =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive", "negative"])
    train_model(create_ngram_model, X, Y, name="sent vs rest", plot=True)

    print "== Pos vs. rest =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive"])
    train_model(create_ngram_model, X, Y, name="pos vs rest", plot=True)

    print "== Neg vs. rest =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["negative"])
    train_model(create_ngram_model, X, Y, name="neg vs rest", plot=True)

    print "time spent:", time.time() - start_time
