# import libraries
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
from sklearn.externals import joblib
import plotly.graph_objs as go
from sklearn.externals import joblib

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
from sklearn.model_selection import KFold


# load data
engine = create_engine('sqlite:///../data/disasterResponse.db')
df = pd.read_sql_table('messages_categories', engine)
df_scores = pd.read_csv('scoring.csv')

# load model
#model = joblib.load("../models/model.pkl")

# extract data needed for visuals
# TODO: Below is an example - modify to extract data for your own visuals


def return_names():
    return df.iloc[:,4:].columns.values.tolist()

# create visuals
# TODO: Below is an example - modify to create your own visuals

def return_graphs():
    df_melt = df.iloc[:,4:].melt()
    df_melt['value'].astype('int64')
    df_mod = df_melt.groupby('variable').agg({'value':'sum'}).reset_index()

    print(int(df_mod['value'][0]))

    category_names = df_mod.values[:,0].tolist()
    cat_df = df.iloc[:,4:].apply(lambda x: x.value_counts())
    print(cat_df.columns.values)

    graph_one=[]
    graph_one.append(
      go.Bar(
        x = df_scores['Unnamed: 0'].values.tolist(),
        y = df_scores['f1_score'].values.tolist(),
      )
    )


    layout_one = dict(title = 'F1 Score per category',
                xaxis = dict(title = 'Category',),
                yaxis = dict(title = 'F1 score'),
                )


    graph_two=[]
    graph_two.append(
          go.Bar(
            x = df_scores['Unnamed: 0'].values.tolist(),
            y = df_scores['precision_score'].values.tolist(),
          )
        )


    layout_two = dict(title = 'Precision Score per category',
                    xaxis = dict(title = 'Category',),
                    yaxis = dict(title = 'Precision score'),
                    )

    graph_three=[]
    graph_three.append(
          go.Bar(
            x = df_scores['Unnamed: 0'].values.tolist(),
            y = df_scores['recall_score'].values.tolist(),
          )
        )


    layout_three = dict(title = 'Recall Score per category',
                    xaxis = dict(title = 'Category',),
                    yaxis = dict(title = 'Recall score'),
                    )


    graph_four=[]
    graph_four.append(
          go.Bar(
            x = df_scores['Unnamed: 0'].values.tolist(),
            y = df_scores['accuracy_score'].values.tolist(),
          )
        )


    layout_four = dict(title = 'Accuracy Score per category',
                    xaxis = dict(title = 'Category',),
                    yaxis = dict(title = 'Accuracy score'),
                    )

    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))
    graphs.append(dict(data=graph_four, layout=layout_four))
    return graphs
