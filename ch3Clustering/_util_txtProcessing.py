
"""
autor:         Micha Segeritz (the.statistikR@gmail.com)
first created: 2013-12-13
program name:  _util_txtProcessing.py
original path: /Users/micha/GoogleDrive/WORKSPACE/travaille/dataScience/mlSystemPy/ch3Custering/_util_txtProcessing.py

Acknowledgement: Some of this code is adapted from code found in repository: 
https://github.com/luispedro/BuildingMachineLearningSystemsWithPython

program function:
-----------------

This program creates a vectorizer that takes in a [list of texts] and processes them:

X = vectorizer.fit_transform([list of texts])

This command

1) tokenizes text in each list element
2) removes stop words
3) stem tokenized words
4) counting remaining words and calculate TF-IDF (Term Frequency - Inverse Document Frequency)

# looks at the terms
> print(vectorizer.get_feature_names())


# convert sparse matrix to normal matrix
> X = X.toarray().transpose()


edits:
------

date:        comment:
-----        --------


"""

#############################################
# Import Python Libraries:

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem


#############################################
# Set User Parameters:

#############################################


english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
    
#vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english', decode_error='ignore')
print("""
initialize vectorizer
popular methods are: 
X = vectorizer.fit_transform([listOfTextDocuments]) # analyze new data
newX = vectorizer.transform([newListOfDocs]         # apply methods to new text
""") 