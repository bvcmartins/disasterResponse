#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE


def load_data(database_filepath):
    '''
        Loads database to dataframe and perform preliminary cleaning
        input: database filepath
        output: dataframe
    '''
    engine = create_engine('sqlite:///disasterResponse_test.db')
    
    #engine = create_engine(database_filepath)
    df = pd.read_sql('SELECT * FROM messages_categories', engine)
    df.iloc[:,4:] = df.iloc[:,4:].astype('int64')
    df.drop(['related','child_alone'], axis=1, inplace=True)
    return df.message.copy(), df.iloc[:,4:].values, df.iloc[:,4:].columns.values


def tokenize(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    text = re.sub("[^a-zA-Z]", " ", text)
    return [lemmatizer.lemmatize(word).lower().strip() for word in nltk.word_tokenize(text)\
            if not word in stop_words] 

class Clean_text(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
        
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X, y=None):
        
        return pd.Series(X).apply(lambda x: " ".join(self.tokenize(x))).values
    
    def tokenize(self, sentence):
        #stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
        return [lemmatizer.lemmatize(word).lower().strip() for word in nltk.word_tokenize(sentence)\
            if not word in stop_words] 

def build_model():
    return Pipeline([
    ('clean_text', Clean_text()),
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('abclf', MultiOutputClassifier(AdaBoostClassifier()))])

def train_model(X, y, pipeline):
    #X = df.message.copy()
    #y = df.iloc[:,4:].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32) 
    return pipeline.fit(X_train, y_train)

def cross_val(X_train, y_train, pipeline):
    labels = []
    preds = []
    kf = KFold(n_splits=5, shuffle=True)
    for train_indices, test_indices in kf.split(X_train, y_train):
   
        X_train_kf = X_train.iloc[train_indices]
        y_train_kf = y_train[train_indices]
    
        X_test_kf = X_train.iloc[test_indices]
        y_test_kf = y_train[test_indices]
  
        pipeline.fit(X_train_kf, y_train_kf)
    
                
        labels.extend(y_test_kf)
        preds.extend(pipeline.predict(X_test_kf))

    return pd.DataFrame(labels), pd.DataFrame(preds)

def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        #print(database_filepath)
        #engine = create_engine(database_filepath)
        #df = pd.read_sql('SELECT * FROM messages_categories', engine)
        
        #engine = create_engine('sqlite:///data/disasterResponse.db')
        #print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        #load_data(database_filepath)
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        #print('Building model...')
        #model = build_model()
        
        #print('Training model...')
        #model.fit(X_train, Y_train)
        
        #print('Evaluating model...')
        #evaluate_model(model, X_test, Y_test, category_names)

        #print('Saving model...\n    MODEL: {}'.format(model_filepath))
        #save_model(model, model_filepath)

        #print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
