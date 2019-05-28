import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.graph_objs as go

from sklearn.externals import joblib
from sqlalchemy import create_engine
from wrangling import return_graphs, return_names
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re

app = Flask(__name__)

class Clean_text(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):

        return pd.Series(X).apply(lambda x: " ".join(self.tokenize(x))).values

    def tokenize(self, sentence):
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
        return [stemmer.stem(lemmatizer.lemmatize(word).lower().strip()) for word in nltk.word_tokenize(sentence)\
            if not word in stop_words]

## load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    figures = return_graphs()
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                           ids=ids,
                           figuresJSON=figuresJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(return_names(), classification_labels))
    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
