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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
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
        Loads database to dataframe
        and performs preliminary cleaning
        input: database filepath
        output: dataframes X, y, columns 
    '''
    #engine = create_engine('sqlite:///../data/disasterResponse_test.db')
    
    engine = create_engine(database_filepath)
    query = 'SELECT * FROM messages_categories'
    df = pd.read_sql(query, engine)
    df.iloc[:,4:] = df.iloc[:,4:].astype('int64')
    df.drop(['related','child_alone'], axis=1, inplace=True)
    return df.message.copy(), df.iloc[:,4:].values, 
df.iloc[:,4:].columns.values

class Clean_text(BaseEstimator, TransformerMixin):
    """
        This class applies the following
        operations to test:
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
            Transforms text to vectorized form
            input: X
            output: modified pandas series
        """
        return pd.Series(X).apply\
                (lambda x: " ".join(self.tokenize(x))).values
                                                    
    def tokenize(self, sentence):
        """
            Tokenize text
            input: string
            output list of tokens
        """
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
        return [stemmer.stem(lemmatizer.\
                lemmatize(word).\
                lower().\
                strip()) for word in \
                nltk.word_tokenize(sentence) if not \
                word in stop_words] 

def build_model():
    """
        Builds text classification model 
        input: none
        output: pipeline object
    """
    return Pipeline([
        ('clean_text', Clean_text()),
        ('vectorizer', CountVectorizer(min_df = 10,\
                stop_words="english", ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(sublinear_tf=True, \
                norm='l1')),
        ('mbclf', MultiOutputClassifier(AdaBoostClassifier\
                (random_state=34, \
                n_estimators=100, \
                learning_rate=1.0)))])
    
def evaluate_model(model, X_test, y_test, columns):
    """
        Calculates prediction scores
        input: dataframes - labes and predictions
        output: dataframe with scores
    """
    preds = model.predict(X_test) 
    list_scorings = [f1_score, accuracy_score,\
                precision_score, recall_score]
    return pd.DataFrame({scoring.__name__:\
                {column: scoring(y_test[:,n],\
                preds[:,n]) for n, column in \
                enumerate(columns)}for scoring in list_scorings})

def save_model(model, model_filepath):
    """
        Saves model as a pickle file
        input: model, model filepath
        output: none
    """
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'\
        .format(database_filepath))
        df = load_data(database_filepath)
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        df_evaluate = evaluate_model(model, X_test, \
                y_test, category_names)
        print(df_evaluate)

        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
