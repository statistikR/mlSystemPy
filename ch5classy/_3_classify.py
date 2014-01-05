
"""
author:         Micha Segeritz (the.statistikR@gmail.com)
first created: 2014-01-02
program name:  ch5classy._3_classify.py
original path: /Users/micha/GoogleDrive/WORKSPACE/travaille/dataScience/mlSystemPy/ch5classy/_3_classify.py

program function:
-----------------
- this program aims to develop a classifier for stackoverflow answer posts into 
  good and bad answers
- we use supervised learning alg. (NN & logReg) to develop a classifier procedure
  that uses post features and preassigned "good" and "bad" labels to learn how
  to classify posts

1) setup stackoverflow answer datasets
2) create some additional features based on stackoverflow post characteristics
3) plot feature histograms
4) k-nearest neighbor classifier (+ bias_variance_analysis, complexity analysis)
5) logistic regression classifier (+ bias_variance_analysis)


edits:
------

date:        comment:
-----        --------


"""

#############################################
# Import Python Libraries:

import time
import os
start_time = time.time()

import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.cross_validation import KFold
from sklearn import neighbors

# the util module is written by Richerts und Coelho and can be accessed through github
from utils import plot_roc, plot_pr
from utils import plot_feat_importance
from utils import load_meta
from utils import fetch_posts
from utils import plot_feat_hist
from utils import plot_bias_variance
from utils import plot_k_complexity

import nltk

#############################################
# Set User Parameters:

DATA_DIR ="/Users/micha/GoogleDrive/WORKSPACE/data/production/privateProj/pyMLSy/stackOverflow"
CHART_DIR = "./charts"
#############################################


# create paths for data to write out
chosen = os.path.join(DATA_DIR, "chosen.tsv")
chosen_meta = os.path.join(DATA_DIR, "chosen-meta.json")



# question Id -> {'features'->feature vector, 'answers'->[answer Ids]}, 'scores'->[scores]}
# scores will be added on-the-fly as the are not in meta

# returns 3 dictionaries; dictionary 2 and 3 are "translation dicts"
meta, id_to_idx, idx_to_id = load_meta(chosen_meta)


# splitting questions into train (70%) and test(30%) and then take their
# answers
all_posts = list(meta.keys())

# creates lists with all questions and all answer ids
all_questions = [q for q, v in meta.items() if v['ParentId'] == -1]
all_answers = [q for q, v in meta.items() if v['ParentId'] != -1]  # [:500]

print len(all_questions)
print len(all_answers)

feature_names = np.array((
    'NumTextTokens',
    'NumCodeLines',
    'LinkCount',
    'AvgSentLen',
    'AvgWordLen',
    'NumAllCaps',
    'NumExclams',
    'NumImages'
))

# activate the following for reduced feature space
"""
feature_names = np.array((
    'NumTextTokens',
    'LinkCount',
))
"""

# function that creates additional features and adds them to meta

def prepare_sent_features():
    for pid, text in fetch_posts(chosen, with_index=True):
        if not text:
            meta[pid]['AvgSentLen'] = meta[pid]['AvgWordLen'] = 0
        else:
            sent_lens = [len(nltk.word_tokenize(
                sent)) for sent in nltk.sent_tokenize(text)]
            meta[pid]['AvgSentLen'] = np.mean(sent_lens)
            meta[pid]['AvgWordLen'] = np.mean(
                [len(w) for w in nltk.word_tokenize(text)])

        meta[pid]['NumAllCaps'] = np.sum(
            [word.isupper() for word in nltk.word_tokenize(text)])

        meta[pid]['NumExclams'] = text.count('!')


prepare_sent_features()

# returns feature values for specified id
def get_features(aid):
    return tuple(meta[aid][fn] for fn in feature_names)

## create data arrays for X and y

qa_X = np.asarray([get_features(aid) for aid in all_answers])
# Score > 0 tests => positive class is good answer
# Score <= 0 tests => positive class is poor answer
# results in True/False
qa_Y = np.asarray([meta[aid]['Score'] > 0 for aid in all_answers])
classifying_answer = "good"


# print features to histogram
for idx, feat in enumerate(feature_names):
    plot_feat_hist([(qa_X[:, idx], feat)])
"""
plot_feat_hist([(qa_X[:, idx], feature_names[idx]) for idx in [1,0]], 'feat_hist_two.png')
plot_feat_hist([(qa_X[:, idx], feature_names[idx]) for idx in [3,4,5,6]], 'feat_hist_four.png')
"""
avg_scores_summary = []


def measure(clf_class, parameters, name, data_size=None, plot=False):
    start_time_clf = time.time()
    if data_size is None:
        X = qa_X
        Y = qa_Y
    else:
        X = qa_X[:data_size]
        Y = qa_Y[:data_size]

    cv = KFold(n=len(X), n_folds=10, indices=True)

    train_errors = []
    test_errors = []

    scores = []
    roc_scores = []
    fprs, tprs = [], []

    pr_scores = []
    precisions, recalls, thresholds = [], [], []
    
    # loop through n cross validation folds
    for train, test in cv:
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]

        clf = clf_class(**parameters)

        # fit model
        clf.fit(X_train, y_train)
        
        # predict training set, predict test set
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        # save training error
        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)

        scores.append(test_score)
        proba = clf.predict_proba(X_test)   # return probability test dataset
                                            # [0.8  0.2], --> prob for 0 and 1 (average of n closest neighbors) 

        # calculate false prediction rate, true prediction rate
        # for different threshold values (threshold values are
        # calculated as proportion m/n nearest neighbors
        label_idx = 1
        fpr, tpr, roc_thresholds = roc_curve(y_test, proba[:, label_idx])
        
        # calculate precision, recall, and precision thresholds
        precision, recall, pr_thresholds = precision_recall_curve(y_test, proba[:, label_idx])

        # store results in container variables
        roc_scores.append(auc(fpr, tpr))
        fprs.append(fpr)
        tprs.append(tpr)

        pr_scores.append(auc(recall, precision))
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(pr_thresholds)
        
        print "classification report (accepted -> p > 0.63)"
        print(classification_report(y_test, proba[:, label_idx] >
              0.63, target_names=['not accepted', 'accepted']))

    # get medium clone
    scores_to_sort = pr_scores  # roc_scores
    medium = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]

    if plot:
        #plot_roc(roc_scores[medium], name, fprs[medium], tprs[medium])
        plot_pr(pr_scores[medium], name, precisions[medium],
                recalls[medium], classifying_answer + " answers")

        if hasattr(clf, 'coef_'):
            plot_feat_importance(feature_names, clf, name)

    summary = (name,
               np.mean(scores), np.std(scores),
               np.mean(roc_scores), np.std(roc_scores),
               np.mean(pr_scores), np.std(pr_scores),
               time.time() - start_time_clf)
    
    print "summary: name, validation_dataset_correct_prediction_mean, score_std, rocscore_mean, rocscore_std, prscore_mean, prscore_std, time"
    print(summary)
    avg_scores_summary.append(summary)
    precisions = precisions[medium]
    recalls = recalls[medium]
    thresholds = np.hstack(([0], thresholds[medium]))
    idx80 = precisions >= 0.8
    print("P=%.2f R=%.2f thresh=%.2f" % (precisions[idx80][0], recalls[
          idx80][0], thresholds[idx80][0]))

    return np.mean(train_errors), np.mean(test_errors)


def bias_variance_analysis(clf_class, parameters, name):
    data_sizes = np.arange(60, 2000, 4)

    train_errors = []
    test_errors = []

    for data_size in data_sizes:
        train_error, test_error = measure(
            clf_class, parameters, name, data_size=data_size)
        train_errors.append(train_error)
        test_errors.append(test_error)

    plot_bias_variance(data_sizes, train_errors,
                       test_errors, name, "Bias-Variance for '%s'" % name)


def k_complexity_analysis(clf_class, parameters):
    ks = np.hstack((np.arange(1, 20), np.arange(21, 100, 5)))

    train_errors = []
    test_errors = []

    for k in ks:
        parameters['n_neighbors'] = k
        train_error, test_error = measure(
            clf_class, parameters, "%dNN" % k, data_size=2000)
        train_errors.append(train_error)
        test_errors.append(test_error)

    plot_k_complexity(ks, train_errors, test_errors)


'''
Following code run a nearest neighbor classifying algorithm
'''

for k in [5]:  #  [5, 10, 40, 90]:  
    
    bias_variance_analysis(neighbors.KNeighborsClassifier, {
                           'n_neighbors': k, 'warn_on_equidistant': False}, "%iNN" % k)
    k_complexity_analysis(neighbors.KNeighborsClassifier, {'n_neighbors': k,
                                                           'warn_on_equidistant': False})
    '''
    measure(neighbors.KNeighborsClassifier, {'n_neighbors': k, 'p': 2}, "%iNN" % k)
    '''

from sklearn.linear_model import LogisticRegression
for C in [0.01]:   # [0.1, 0.1, 1.0, 10.0]:
    name = "LogReg C=%.2f" % C
    bias_variance_analysis(LogisticRegression, {'penalty': 'l2', 'C': C}, name)
    measure(LogisticRegression, {'penalty': 'l2', 'C': C}, name, plot=True)

print("=" * 50)
from operator import itemgetter
for s in reversed(sorted(avg_scores_summary, key=itemgetter(1))):
    print("%-20s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f" % s)

print("time spent:", time.time() - start_time)

    

