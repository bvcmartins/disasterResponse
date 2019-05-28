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
    """
        Loads data from csv file
        input: path to csv file
        output: pandas data frame
    """
    # read in file
    try:
        file = pd.read_csv(data_file)
    except:
        file = None
    return file

def str_to_dict(x):
    """
        Converts string to dictionary  
        input: string
        output: dictionary
    """
    list_ = re.split('\W+',x)
    return {key:value for key, value 
    in zip(list_[0:][::2], list_[1:][::2])}

def clean_categories(categories):
    """
        Creates dataframe from dictionary of categories
        input: dataframe
        output: dataframe with categories expanded
    """
    df_ = pd.DataFrame(list(categories['categories'].map(
        lambda x : str_to_dict(x))))
    return pd.concat([categories['id'], df_], axis=1)

def merge_datasets(messages, categories):
    """
        Merges messages and categories datasets
        input: messages and categories dataframes
        output: merged dataset 
    """
    return messages.merge(categories, 
            on=('id')).drop_duplicates()

def data_preprocessing(messages, categories):
    """
        Cleans and merges messages and categories datasets
        input: messages and categories datasets
        output: merged and cleaned dataset
    """
    clean_df = clean_categories(categories)
    df = merge_datasets(messages, clean_df)
    df.iloc[:,4:] = df.iloc[:,4:].astype('int64')
    df.drop(['related','child_alone'], axis=1, inplace=True)
    return df
    
def load_to_db(df, db_name):
    """
        Loads dataframe to database
        input: dataframe, database path
        output: none 
    """
    # load to database
    engine = create_engine(
            'sqlite:///{}'.format(db_name), echo=False)
    df.to_sql('messages_categories',engine, index=False)

def generate_X(df):
    """
        Generates features series 
        input: messages-categories dataframe
        output: message series 
    """
    return df.message.copy()

def generate_y(df):
    """
        Generates categories dataframe
        input: messages-categories dataframe
        output: dataframe with categories 
    """
    return df.iloc[:,4:].values 

class Clean_text(BaseEstimator, TransformerMixin):
    """
        Class for text cleaning and nlp procssing  
        Operations applied:
        - stemmer
        - lemmatizer
        - punctuation removal
        - lower case
        - strip white spaces
    """
    def __init__(self):
        return None
                        
    def fit(self, X=None, y=None):
        return self
                                        
    def transform(self, X, y=None):
        """
            Transforms dataset
            input: dataframe with messages
            output: tokenized series
        """
        return pd.Series(X).apply(lambda x: 
        " ".join(self.tokenize(x))).values
                                                                
    def tokenize(self, sentence):
        """
            Cleans and tokenizes sentence
            input: string
            output: cleaned and tokenized list 
        """
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
        return [stemmer.stem(lemmatizer.
        lemmatize(word).lower().strip()) 
        for word in nltk.word_tokenize(sentence) \
        if not word in stop_words] 
    
def build_model():
    """
        Builds nlp and classifier pipeline
        input: none
        output: model
    """
    # text processing and model pipeline
    ml_pipeline = Pipeline([
    ('clean_text', Clean_text()),
    ('vectorizer', CountVectorizer(min_df= 5,
        stop_words="english", ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(sublinear_tf=True, 
        norm='l2')),
    ('abclf', MultiOutputClassifier(
        AdaBoostClassifier(random_state=34, 
            n_estimators=100, learning_rate=1.0)))])

    # define parameters for GridSearchCV
    #parameters = {
    #'vectorizer__min_df': [1,2],
    #'vectorizer__ngram_range': [(1,1),(1,2)],
    #'abclf__estimator__learning_rate': [0.1,0.2,1.0],
    #'abclf__estimator__n_estimators': [50, 100]
    #}
    # limited set for tests
    parameters = {
    #'vectorizer__ngram_range': [(1,1),(1,2)],
    #'abclf__estimator__learning_rate': [0.1,1.0],
    #'abclf__estimator__n_estimators': [50, 100]
    'abclf__estimator__n_estimators': [50, 100]
    }


    # create gridsearch object 
    return GridSearchCV(ml_pipeline, 
            param_grid=parameters, verbose=10)

def train(X, y, model, columns):
    """
       Trains model and assess scores  
       input: X, y, model and list of categories
       output: 
    """
    # train test split
    X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.2, random_state=32) 

    # fit model
    model.fit(X_train, y_train)

    # output model test results
    list_scorings = [f1_score, 
            precision_score, 
            recall_score, 
            accuracy_score]
    labels_df = pd.DataFrame(y_test)
    preds_df = pd.DataFrame(model.predict(X_test))
    return pd.DataFrame(
            {scoring.__name__: 
                {column: scoring(
            labels_df.iloc[:,n], 
            preds_df.iloc[:,n])
           # pos_label=1) 
            for n, column in enumerate(columns)} 
        for scoring in list_scorings})

def export_model(model):
    """
        Export model as a pickle file
        input: model 
        output: none
    """
    # Export model as a pickle file
    filename='model.pkl'
    pickle.dump(model, open(filename,'wb'))

def run_pipeline(data_file):
    """
        Runs data processing pipeline
        input: list with paths for messages and categories
        output: none
    """
    messages_df = load_data(data_file[0])
    categories_df = load_data(data_file[1])
    df = data_preprocessing(messages_df, categories_df)
    X = generate_X(df)
    y = generate_y(df)
    # build model pipeline
    model = build_model() 
    # train model pipeline
    columns = df.iloc[:,4:].columns.values
    score_df = train(X, y, model, columns)  
    # sace model
    export_model(model) 
    score_df.to_csv('scoring.csv', index=False)

if __name__ == '__main__':
    data_file = []
    data_file.append(sys.argv[1])  # get filename of dataset
    data_file.append(sys.argv[2])
    run_pipeline(data_file)  # run data pipeline
