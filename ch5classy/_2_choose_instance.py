
"""
author:         Micha Segeritz (the.statistikR@gmail.com)
first created: 2014-01-02
program name:  ch5classy._2_choose_instance.py
original path: /Users/micha/GoogleDrive/WORKSPACE/travaille/dataScience/mlSystemPy/ch5classy/_2_choose_instance.py

Acknowledgement: Some of this code is adapted from code found in repository: 
https://github.com/luispedro/BuildingMachineLearningSystemsWithPython

program function:
-----------------

processes filtered_meta and filtered datasets and creates datasets
chosen and chosen_meta that should be used for classification analysis

This program provides several conditional rules how the question
answer pairs get sampled

The dataset chosen contains all chosen posts for analysis as well as
the id of the post

The dataset chosen_meta contains all chosen posts as well:

for each post it contains important features of the posts such as
answer ids for questions, and Parent ID for answers. Additionally, it also contains scores etc.

Question 73697
"73697": {"idx": 3, "MisSpelledFraction": 0.0, "NumImages": 0, "Answers": [73728, 73730], "IsAccepted": 0, "NumTextTokens": 97, "Score": 2, "ParentId": -1, "NumCodeLines": 0, "TimeToAnswer": 0, "LinkCount": 1},
Answer 73728 to question 73697
"73728": {"idx": 4, "MisSpelledFraction": 0.0, "NumImages": 0, "IsAccepted": 0, "NumTextTokens": 30, "Score": -6, "ParentId": 73697, "NumCodeLines": 0, "TimeToAnswer": 50545, "LinkCount": 0},


edits:
------

date:        comment:
-----        --------


"""

#############################################
# Import Python Libraries:

import os
try:
    import ujson as json  # UltraJSON if available
except:
    import json
import sys
from collections import defaultdict

try:
    import enchant
    speller = enchant.Dict("en_US")

except:
    print("""\
Enchant is not installed, which is not a problem since spell correction features
will not be used in the chapter. If, however, you want to experiment with them
(highly encouraged!), you can get the library from http://packages.python.org/pyenchant/.
""")
    class EnchantMock:
        def __init__(self):
            pass
        def check(self, word):
            return True
    speller = EnchantMock()


#############################################
# Set User Parameters:

DATA_DIR ="/Users/micha/GoogleDrive/WORKSPACE/data/production/privateProj/pyMLSy/stackOverflow"


#############################################

# create paths for pre-processed data
filtered = os.path.join(DATA_DIR, "filtered.tsv")
filtered_meta = os.path.join(DATA_DIR, "filtered-meta.json")

# create paths for data to write out
chosen = os.path.join(DATA_DIR, "chosen.tsv")
chosen_meta = os.path.join(DATA_DIR, "chosen-meta.json")

# load data
filtered_meta = json.load(open(filtered_meta, "r"))

def misspelled_fraction(p):
    tokens = p.split()
    if not tokens:
        return 0.0
    return 1 - float(sum(speller.check(t) for t in tokens)) / len(tokens)


def data(filename, col=None):
    for line in open(filename, "r"):
        data = line.strip().split("\t")

        # check format
        Id, ParentId, IsAccepted, TimeToAnswer, Score, Text, NumTextTokens, NumCodeLines, LinkCount, NumImages = data

        if col:
            yield data[col]
        else:
            yield data

posts_to_keep = set()

found_questions = 0

num_qestion_sample = 1000

# keep the best and worst, but only if we have one with positive and one with negative score
# filter_method = "negative_positive"

# if true, only keep the lowest scoring answer per class in addition to the accepted one
# filter_method = "only_one_per_class "

# if not None, specifies the number of unaccepted per question
# filter_method = "sample_per_question"
filter_method = "negative_positive"  # warning: this does not retrieve many!
# filter_method = "only_one_per_class"
MaxAnswersPerQuestions = 10  # filter_method == "sample_per_question"

# filter_method = "all"

# equal share of questions that are unanswered and those that are answered
# filter_method = "half-half"

unaccepted_scores = {}

has_q_accepted_a = {}
num_q_with_accepted_a = 0
num_q_without_accepted_a = 0

#loop through questions answer dictionary
for ParentId, posts in filtered_meta.items(): # convert dictionary filtered_meta into a list of tuples

    assert ParentId != -1  # ?what does assert mean?
    
    # only keep questions that have at least 2 answers
    if len(posts) < 2:
        continue

    ParentId = int(ParentId)
    AllIds = set([ParentId])
    AcceptedId = None
    UnacceptedId = None
    UnacceptedIds = []
    UnacceptedScore = sys.maxsize

    NegativeScoreIds = []
    PositiveScoreIds = []

    if filter_method == "half-half":

        has_accepted_a = False
        for post in posts:
            Id, IsAccepted, TimeToAnswer, Score = post

            if IsAccepted:
                has_accepted_a = True
                break

        has_q_accepted_a[ParentId] = has_accepted_a

        if has_accepted_a:
            if num_q_with_accepted_a < num_qestion_sample / 2:
                num_q_with_accepted_a += 1
                posts_to_keep.add(ParentId)
        else:
            if num_q_without_accepted_a < num_qestion_sample / 2:
                num_q_without_accepted_a += 1
                posts_to_keep.add(ParentId)

        if num_q_without_accepted_a + num_q_with_accepted_a > num_qestion_sample:
            assert -1 not in posts_to_keep
            break

    else:
        
        # loop through answers of a question
        for post in posts:
            
            # extract key variables of answer
            Id, IsAccepted, TimeToAnswer, Score = post

            if filter_method == "all":
                AllIds.add(int(Id))

            elif filter_method == "only_one_per_class":
                if IsAccepted:
                    AcceptedId = Id
                elif Score < UnacceptedScore:
                    UnacceptedScore = Score
                    UnacceptedId = Id

            elif filter_method == "sample_per_question":
                if IsAccepted:
                    AcceptedId = Id
                else:
                    UnacceptedIds.append(Id)

            elif filter_method == "negative_positive":
                if Score < 0:
                    NegativeScoreIds.append((Score, Id))
                elif Score > 0:
                    PositiveScoreIds.append((Score, Id))

            else:
                raise ValueError(filter_method)

        added = False
        if filter_method == "all":
            posts_to_keep.update(AllIds)
            added = True
        elif filter_method == "only_one_per_class":
            if AcceptedId is not None and UnacceptedId is not None:
                posts_to_keep.add(ParentId)
                posts_to_keep.add(AcceptedId)
                posts_to_keep.add(UnacceptedId)
                added = True

        elif filter_method == "sample_per_question":
            if AcceptedId is not None and UnacceptedIds is not None:
                posts_to_keep.add(ParentId)
                posts_to_keep.add(AcceptedId)
                posts_to_keep.update(UnacceptedIds[:MaxAnswersPerQuestions])
                added = True

        elif filter_method == "negative_positive":
            if PositiveScoreIds and NegativeScoreIds:
                posts_to_keep.add(ParentId)

                posScore, posId = sorted(PositiveScoreIds)[-1]
                posts_to_keep.add(posId)

                negScore, negId = sorted(NegativeScoreIds)[0]
                posts_to_keep.add(negId)
                print("%i: %i/%i %i/%i" % (ParentId, posId,
                      posScore, negId, negScore))
                added = True

        if added:
            found_questions += 1

    if num_qestion_sample and found_questions >= num_qestion_sample:
        break

total = 0
kept = 0

already_written = set()
chosen_meta_dict = defaultdict(dict)

# write all posts that should be kept as id - post text pair
with open(chosen, "w") as f:
    for line in data(filtered):
        strId, ParentId, IsAccepted, TimeToAnswer, Score, Text, NumTextTokens, NumCodeLines, LinkCount, NumImages = line
        Text = Text.strip()

        total += 1

        Id = int(strId)
        if Id in posts_to_keep:
            if Id in already_written:
                print(Id, "is already written")
                continue

            if kept % 100 == 0:
                print(kept)

            # setting meta info
            post = chosen_meta_dict[Id]
            post['ParentId'] = int(ParentId)
            post['IsAccepted'] = int(IsAccepted)
            post['TimeToAnswer'] = int(TimeToAnswer)
            post['Score'] = int(Score)
            post['NumTextTokens'] = int(NumTextTokens)
            post['NumCodeLines'] = int(NumCodeLines)
            post['LinkCount'] = int(LinkCount)
            post['MisSpelledFraction'] = misspelled_fraction(Text)
            post['NumImages'] = int(NumImages)
            post['idx'] = kept  # index into the file

            if int(ParentId) == -1:
                q = chosen_meta_dict[Id]

                if not 'Answers' in q:
                    q['Answers'] = []

                if filter_method == "half-half":
                    q['HasAcceptedAnswer'] = has_q_accepted_a[Id]

            else:
                q = chosen_meta_dict[int(ParentId)]

                if int(IsAccepted) == 1:
                    assert 'HasAcceptedAnswer' not in q
                    q['HasAcceptedAnswer'] = True

                if 'Answers' not in q:
                    q['Answers'] = [Id]
                else:
                    q['Answers'].append(Id)

            f.writelines("%s\t%s\n" % (Id, Text))
            kept += 1

with open(chosen_meta, "w") as fm:
    json.dump(chosen_meta_dict, fm)

print("total posts (questions and answers)=", total)
print("kept posts=", kept)

