# import packages
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import defaultdict
from sqlalchemy import create_engine
import numpy as np
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
from sklearn.metrics import recall_score, make_scorer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

def load_data(data_file):
    # read in file
    try:
        file = pd.read_csv(data_file)
    except:
        file = None
    return file

def str_to_dict(x):
    list_ = re.split('\W+',x)
    return {key:value for key, value 
    in zip(list_[0:][::2], list_[1:][::2])}

def clean_categories(categories):
    df_ = pd.DataFrame(list(categories['categories'].map(lambda x : str_to_dict(x))))
    return pd.concat([categories['id'], df_], axis=1)

def merge_datasets(messages, categories):
    return messages.merge(categories, 
    on=('id')).drop_duplicates()

def data_preprocessing(messages, categories):
    clean_df = clean_categories(categories)
    return merge_datasets(messages, clean_df)
    
def load_to_db(df, db_name):
    # load to database
    engine = create_engine('sqlite:///{}'.format(db_name), echo=False)
    df.to_sql('messages_categories',engine, index=False)

def generate_X(df):
    return df.message.copy()

def generate_y(df):
    return df.iloc[:,4:].values 

class Clean_text(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
                        
    def fit(self, X=None, y=None):
        return self
                                        
    def transform(self, X, y=None):
        return pd.Series(X).apply(lambda x: 
        " ".join(self.tokenize(x))).values
                                                                
    def tokenize(self, sentence):
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
        return [stemmer.stem(lemmatizer.
        lemmatize(word).lower().strip()) 
        for word in nltk.word_tokenize(sentence) \
        if not word in stop_words] 
    
def build_model():
    # text processing and model pipeline
    f1_scorer = make_scorer(f1_score, average='macro')

    ml_pipeline = Pipeline([
    ('clean_text', Clean_text()),
    ('vectorizer', CountVectorizer(min_df= 5,stop_words="english", ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(sublinear_tf=True, norm='l2')),
    ('abclf', MultiOutputClassifier(AdaBoostClassifier(random_state=34, n_estimators=100, learning_rate=1.0)))])

    # define parameters for GridSearchCV
    #parameters = {
    #'vectorizer__min_df': [1,2],
    #'vectorizer__ngram_range': [(1,1),(1,2)],
    #'abclf__estimator__learning_rate': [0.1,0.2,1.0],
    #'abclf__estimator__n_estimators': [50, 100]
    #}
    # limited set for tests
    parameters = {
    'vectorizer__ngram_range': [(1,1),(1,2)],
    'abclf__estimator__learning_rate': [0.1,0.2,1.0],
    'abclf__estimator__n_estimators': [50, 100]
    }


    # create gridsearch object and return as final model pipeline
    grid_model = GridSearchCV(ml_pipeline, param_grid=parameters, verbose=10)
    return grid_model

def train(X, y, model):
    # train test split
    X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.2, random_state=32) 

    # fit model
    model.fit(X_train, y_train)

    # output model test results
    model.predict(X_test)

    dict_predict = {column: f1_score(y_test[:,n], y_pred[:,n]) for n, column in enumerate(df.iloc[:,4:].columns.values)}

    return model, dict_predict


def export_model(model):
    # Export model as a pickle file
    filename='data/model.pkl'
    pickle.dump(model, open(filename,'wb'))

def run_pipeline(data_file):
    messages_df = load_data(data_file[0])  # run ETL pipeline
    categories_df = load_data(data_file[1])  # run ETL pipeline
    df = data_preprocessing(messages_df, categories_df)
    X = generate_X(df)
    y = generate_y(df)
    model = build_model()  # build model pipeline
    # train model pipeline
    model, dict_predict = train(X, y, model)  
    export_model(model)  # save model

if __name__ == '__main__':
    data_file = []
    data_file.append(sys.argv[1])  # get filename of dataset
    data_file.append(sys.argv[2])
    run_pipeline(data_file)  # run data pipeline
