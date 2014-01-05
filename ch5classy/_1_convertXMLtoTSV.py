
"""
autor:         Micha Segeritz (the.statistikR@gmail.com)
first created: 2013-12-27
program name:  ch5classy._1_convertXMLtoTSV.py
original path: /Users/micha/GoogleDrive/WORKSPACE/travaille/dataScience/mlSystemPy/ch5classy/_1_convertXMLtoTSV.py

Acknowledgement: Some of this code is adapted from code found in repository: 
https://github.com/luispedro/BuildingMachineLearningSystemsWithPython

program function:
-----------------

This code is based on code from Willi Richert and Luis Pedro Coelho for their book Building Machine Learning Systems with Python

This code selects stackoverflow posts from the years 2011 and 2012 
and creates a tsv file for the posts and a json file for some meta
information about the posts. The json file and the tsv file are linked
via a shared id

data structure meta dictionary:
key: question id/parent id
values: list of lists for each answer to question 


features:
parse xml file and create tab delimited file and json file



edits:
------

date:        comment:
-----        --------


"""

#############################################
# Import Python Libraries:

import os,re

try:
    import ujson as json  # UltraJSON if available
except:
    import json
from dateutil import parser as dateparser

from operator import itemgetter
from xml.etree import cElementTree as etree
from collections import defaultdict

from itertools import imap

#############################################
# Set User Parameters:

DATA_DIR ="/Users/micha/GoogleDrive/WORKSPACE/data/production/privateProj/pyMLSy/stackOverflow"

#############################################



# metaPosts is a posting file from the metaPost page, it's a lot smaller and so I use is for testing, later I will use Posts.xml
filename = os.path.join(DATA_DIR, "metaPosts.xml")
filename_filtered = os.path.join(DATA_DIR, "filtered.tsv")


# setup containers to store key information
q_creation = {}  # creation datetimes of questions
q_accepted = {}  # id of accepted answer


years = defaultdict(int) # counts number of post from each year
num_questions = 0        # counts number of questions among all posts
num_answers = 0


# question -> [(answer Id, IsAccepted, TimeToAnswer, Score), ...]
meta = defaultdict(list)

# regegx to find code snippets
code_match = re.compile('<pre>(.*?)</pre>', re.MULTILINE | re.DOTALL)
link_match = re.compile(
    '<a href="http://.*?".*?>(.*?)</a>', re.MULTILINE | re.DOTALL)
img_match = re.compile('<img(.*?)/>', re.MULTILINE | re.DOTALL)
tag_match = re.compile('<[^>]*>', re.MULTILINE | re.DOTALL)



# function to process body of post; it would be better to use beautiful soup 
def filter_html(s):
    num_code_lines = 0
    link_count_in_code = 0
    code_free_s = s

    num_images = len(img_match.findall(s))

    # remove source code and count how many lines
    for match_str in code_match.findall(s):
        num_code_lines += match_str.count('\n')
        code_free_s = code_match.sub("", code_free_s)

        # sometimes source code contain links, which we don't want to count
        link_count_in_code += len(link_match.findall(match_str))

    anchors = link_match.findall(s)
    link_count = len(anchors)

    link_count -= link_count_in_code

    html_free_s = re.sub(
        " +", " ", tag_match.sub('', code_free_s)).replace("\n", "")

    link_free_s = html_free_s
    for anchor in anchors:
        if anchor.lower().startswith("http://"):
            link_free_s = link_free_s.replace(anchor, '')

    num_text_tokens = html_free_s.count(" ")

    # extract key features from body of post
    return link_free_s, num_text_tokens, num_code_lines, link_count, num_images



## parse xml file; 

def parsexml(filename):
    global num_questions, num_answers

    counter = 0

    # create imap object / iterator
    it = imap(itemgetter(1),
             iter(etree.iterparse(filename, events=('start',))))
   
    # not sure what this is for, somewhat related to memory issues
    root = next(it)  # get posts element
    

    # loop through elements of iterator
    for elem in it:

        # provide status update about data processing
        if counter % 100000 == 0:
            print(counter)
        counter += 1
        
        # only process rows with element tag "row"
        if elem.tag == 'row':
            ## extract some key variables that apply to both questions and answers
            # extract date string and convert to datetime object
            creation_date = dateparser.parse(elem.get('CreationDate'))
            Id = int(elem.get('Id'))
            PostTypeId = int(elem.get('PostTypeId'))
            Score = int(elem.get('Score'))
            
            
            # only process posts from year 2011 and 2012
            if ((creation_date.year < 2011) | (creation_date.year > 2012)):
                continue
            

            # process questions (PostTypeId == 1)
            if PostTypeId == 1:
                num_questions += 1
                years[creation_date.year] += 1
                ParentId = -1                  # questions don't have parentId --> set to -1
                TimeToAnswer = 0
                q_creation[Id] = creation_date
                accepted = elem.get('AcceptedAnswerId')

                if accepted:
                    q_accepted[Id] = int(accepted)
                IsAccepted = 0

            # process answers (PostTypeId ==2)
            elif PostTypeId == 2:
                num_answers += 1

                ParentId = int(elem.get('ParentId'))
                if not ParentId in q_creation:
                    # question was too far in the past
                    continue

                TimeToAnswer = (creation_date - q_creation[ParentId]).seconds

                if ParentId in q_accepted:
                    IsAccepted = int(q_accepted[ParentId] == Id)
                else:
                    IsAccepted = 0
                    

                # add important variables as list to defaultdict meta
                meta[ParentId].append((Id, IsAccepted, TimeToAnswer, Score))

            # if row is neither question nor answer then stop and process next post
            else:
                continue

            # extract key features from posting text
            Text, NumTextTokens, NumCodeLines, LinkCount, NumImages = filter_html(
                elem.get('Body'))

            values = (Id, ParentId,
                      IsAccepted,
                      TimeToAnswer, Score,
                      Text.encode("utf-8"),
                      NumTextTokens, NumCodeLines, LinkCount, NumImages)

            yield values

            root.clear()  # preserve memory
            
            
# save posts in tab delimited document
with open(os.path.join(DATA_DIR, filename_filtered), "w") as f:
    for values in parsexml(filename):
        line = "\t".join(map(str, values))
        f.write(line + "\n")

# save meta information in json file
with open(os.path.join(DATA_DIR, "filtered-meta.json"), "w") as f:
    json.dump(meta, f)

# provide some key information ab out data processing
print("years:", years)
print("#qestions: %i" % num_questions)
print("#answers: %i" % num_answers)
