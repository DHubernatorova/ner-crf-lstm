from itertools import chain
import pickle
import scipy.stats
import sklearn
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import train_test_split

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

with open("word_data_file.obj", "rb") as infile:
    data = pickle.load(infile)


def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],        
    }
    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
        
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
                
    return features

# A function for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, postag, label) in doc]

X = [extract_features(doc) for doc in data]
y = [get_labels(doc) for doc in data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("Train data: " + str(len(y_train)))
print("Test data: " + str(len(y_test)))

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs', 
    c1=0.1, 
    c2=0.01, 
    max_iterations=200, 
    all_possible_transitions=True,
    verbose = True
)
crf.fit(X_train, y_train)
with open('crf.model', 'wb') as model:
        pickle.dump(crf, model)
labels = list(crf.classes_)
print(labels)

y_pred = crf.predict(X_test)
f1_flat = metrics.flat_f1_score(y_test, y_pred, 
                      average='weighted', labels=labels)
print(f1_flat)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))



