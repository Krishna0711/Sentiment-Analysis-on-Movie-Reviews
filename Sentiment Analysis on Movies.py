# -*- coding: utf-8 -*-
"""
Created on Fri May  5 19:54:56 2023

@author: Krishna Vamsi Kolli
"""

## Import necessary libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
import nltk.data
import re
import string
import random
import itertools
import matplotlib.pyplot as plt
#matplotlib inline
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

imdb_df = pd.read_csv("./Movie_reviews.csv")
imdb_df.shape # In order to know structure of the input file

imdb_df.columns # for to column names 
imdb_df['label'].dtype
imdb_df.head(5) # To get the first 5 rows
imdb_df.rename(columns = {'text':'movie_review','label':'opinion'}, inplace = True) # Rename the column names
imdb_df.opinion.value_counts()

stop_words = stopwords.words('english') #Loading the english stopwords and storing in a variable
stop_words.remove('not')
#print(stop_words)

contraction_mapping = {"isn't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have", 'yticket' : 'ticket', "yreferred" : "referred", "activestatus" : "actives tatus",
                           
                           "activesegment" : "active segment", "individualfinancial" : "individual financial", "ysupport" : "support", "reviewnext" : "review next",
                           
                           "ction": "action", "tconsultation": "consultation", "activeaccount": "active account", "nsupport" : "support", "selfservetype" : "self serve type",
                      
                          "individualfinancial" : "individual financial", "i'am" : "i am"}

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower() # Converting the sentence into lower case
    sentence=sentence.replace('{html}',"") #Removing html tags
    cleanr = re.compile('<.*?>')        #For removing the special characters
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext) #Removing html links
    rem_num = re.sub(r'\b[0-9]+\b\s*', '', rem_url)# Removing numbers in a sentence
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in rem_num.split(" ")])
    text = newString.strip(' ')  #Removing the leading spaces 
    text = re.sub(r'[^a-zA-Z]',' ', text) #Keeping only the alphabets related words
    tokenizer = RegexpTokenizer(r'\w+') 
    tokens = tokenizer.tokenize(text)  #tokenizing the sentence into words
    filtered_words = [w for w in tokens if not w in stop_words] #Looping the sentces words and checking stopwords contians or not
    #word for word in words if word not in stop
    #stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words] #lemmatizing each word
    return " ".join(lemma_words)

#Function by which we only take the important words only i.e Noun, adjective, Adverb , Verb
def noun_words(text, verb = True, adverb = True, adjective = True):
    sen = sp(text)
    if verb == True:
        verb = 'VERB'
    else:
        verb == False
    if adverb == True:
        adverb = 'ADVERB'
    else:
        adverb == False
    if adjective == True:
        adjective = 'ADJ'
    else:
        adjective == False
    pun_list = []
    for i in sen:
        #print(i.pos_, i)
        if i.pos_ == "NOUN" or i.pos_ == verb or i.pos_ == adverb or i.pos_ == adjective:
            pun_list.append(i.text)
    text = " ".join(pun_list)
    return text

imdb_df['movie_review_cleanText'] =  imdb_df['movie_review'].astype(str)
imdb_df['movie_review_cleanText'] =  imdb_df['movie_review_cleanText'].map(preprocess)
# imdb_df['important_words'] =  imdb_df['important_words'].astype(str)

imdb_df['movie_review_cleanText'] =  imdb_df['movie_review'].astype(str)
imdb_df['movie_review_cleanText'] =  imdb_df['movie_review_cleanText'].map(preprocess)
# imdb_df['important_words'] =  imdb_df['important_words'].astype(str)

imdb_df.info()
imdb_df.opinion.value_counts(normalize=True)

imdb_df.groupby("opinion").size().plot(kind = 'bar')

imdb_df.head(5)
imdb_df.to_csv("./output.csv")

imdb_df['word_count_before'] = imdb_df['movie_review'].apply(lambda x: len(str(x).split(" ")))
imdb_df.head(10)

imdb_df['word_count_after'] = imdb_df['movie_review_cleanText'].apply(lambda x: len(str(x).split(" ")))
imdb_df.head(10)

imdb_df.columns
imdb_df.word_count_before.describe()

imdb_df.word_count_after.describe()

#After prepocessing we can see that avg of each sentence around 120 words but before preprocessing it is around 230
imdb_df.describe()
imdb_df[imdb_df.word_count_before < 5] # trying to remove the sentences that contain less than 5 words

training, testing = train_test_split(imdb_df, test_size=0.33)
imdb_df.columns
X_train=training["movie_review_cleanText"]
Y_train=training["opinion"]
training_dataset_df = pd.DataFrame({'movie_review_cleanText':X_train,'opinion': Y_train})
training_dataset_df.head(2)

training_case_list=[]
for row in training_dataset_df["movie_review_cleanText"]:
    training_case_list.append(row)
    
## Convert the text form data into vector form using bag of words and tfidf
def vectorization_bow_tfidf(training_case_list, bow = None, tfidf = None):
    if tfidf == True:        
        vectorizer_i1 = TfidfVectorizer()
        tfidf_matrix_iteartion1 = vectorizer_i1.fit_transform(training_case_list)
    if bow == True:
        vectorizer_i1 = CountVectorizer(analyzer='word',ngram_range=(1,3), encoding="ISO-8859-1")
        tfidf_matrix_iteartion1 = vectorizer_i1.fit_transform(training_case_list) ## Transform strings to tf-idf matrix

    X_train=tfidf_matrix_iteartion1
    y_train=training_dataset_df["opinion"].values
    
    return X_train, y_train, vectorizer_i1

# For the clean text got the tfidf vector form 
X_train, y_train, vectorizer_i2 = vectorization_bow_tfidf(training_dataset_df["movie_review_cleanText"].tolist(),\
                                                          False, True)
    
#####Training the model with the manually tagged notes

######### Linear SVC ############
clf_svc=LinearSVC()
clf_fit_svc=clf_svc.fit(X_train,y_train)

########## Naive Bayes #############
clf_nb=(MultinomialNB(fit_prior=True, class_prior=None))
clf_fit_nb=clf_nb.fit(X_train,y_train)

############  Logistic Regression  ################
clf_lr=OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)
clf_fit_lr=clf_lr.fit(X_train,y_train)

#############  Random Forest ############
from sklearn.model_selection import GridSearchCV
params = {'n_estimators':list(range(25,50))}
clf_rf = RandomForestClassifier()

gridcv = GridSearchCV(clf_rf, params,n_jobs=-1)
# gridcv.fit(X_train, y_train)x
# gridcv.best_params_

from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=100) 
clf_fit_rf = clf_rf.fit(X_train,y_train)

############# Test the models ###################
testing_data_list = []
for i in testing["movie_review_cleanText"]:
    testing_data_list.append(i)
    
X_test_overall = vectorizer_i2.transform(testing_data_list)
y_test_overall=testing['opinion'].values

def classify_cases(X_test,model_fit):
    clf_predict=model_fit.predict(X_test)
    return clf_predict

## Function to predict case reason for test data of SVC
clf_predict_svc=classify_cases(X_test_overall,clf_fit_svc)

## Function to predict case reason for test data of logistic regression
clf_predict_lr=classify_cases(X_test_overall,clf_fit_lr)

## Function to predict case reason for test data of naive bayes
clf_predict_nb=classify_cases(X_test_overall,clf_fit_nb)

## Function to predict case reason for test data of Random Forest
clf_predict_rf=classify_cases(X_test_overall,clf_fit_rf)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def generate_accuracy_report(y_test,clf_predict,dataset):
    print ("Overall Classifier's Accuracy: ", accuracy_score(y_test,clf_predict))
    
    print ("############### Confusion Matrix ###############")
    x=confusion_matrix(y_test, clf_predict)
    confusion_mat=pd.DataFrame(x)
    print (confusion_mat)

    print ("############### Classification Report ###############")
    print(classification_report(y_test, clf_predict))
    
    dataset["actual"]=y_test
    dataset["predicted"]=clf_predict
    
generate_accuracy_report(y_test_overall,clf_predict_svc,testing)

print('Test accuracy is {}'.format(accuracy_score(y_test_overall,clf_predict_svc)))

testing['opinion_svc'] = clf_predict_svc
testing.head(1)

generate_accuracy_report(y_test_overall,clf_predict_lr,testing)
print('Test accuracy is {}'.format(accuracy_score(y_test_overall,clf_predict_lr)))

testing['opinion_LR'] = clf_predict_lr
testing.head(1)

generate_accuracy_report(y_test_overall,clf_predict_nb,testing)

testing['opinion_NB'] = clf_predict_nb
testing.head(1)

testing.tail(1)

########### Test on Random forest model ####################
# performing predictions on the test dataset
clf_predict_rf = clf_rf.predict(X_test_overall)

from sklearn import metrics 
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test_overall, clf_predict_rf))

generate_accuracy_report(y_test_overall,clf_predict_rf,testing)

testing['opinion_RF'] = clf_predict_rf
testing.head(1)

testing.sample(5)
imdb_df.shape

################ Result Summary ################
# I had used different classification models in order to predict whether given sentence positive or negative
# i.e Naive bayes, Random forest, Logistic Regression, SVC

################ Finding ###############
# After cleaning the reviews of movies the words contains in a sentence almost decrese by half

################# From the models logistic regression and SVC model are performed very well with a accuracy of almost 90 percentage ##############################
'''Intial expectaion I though of trying to do this project using the bag of words(BOW) but after comparing the result with
tfidf the accurary seems to be prety good with tfidf. So, I continued with tdidf'''

'''If there any time i thought of implementing for neural sentences also'''
