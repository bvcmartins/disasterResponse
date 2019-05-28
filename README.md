# Disaster Response App Project 

Project 5 for Udacity's Data Scientist Nanodegree

## Overview

Figure Eight provided us with a annotated dataset containing
Twitter messages collected from disaster regions.
The dataset is classified in 36 categories (only 34 were
used). The project consisted of:

1. Implementing a NLP pipeline to clean and vectorize 
text data 
2. Classifying the messages according to the categories
3. Implement a Flask / Bootstrap app (on top of a
model provided by Udacity) that can be used to predict
the categories for any new Twitter messages 
 
## Methods

This project used a great variety of methods:

1. NLP
	* Pandas and Re: text processing tools
	* NLTK: Stemmer, Lemmatizer, Tokenizer 
	* Scikit-Learn: CountVectorizer, TfIdf,
AdaBoost classifier
	* Imblearn: SMOTE 
2. Web App
	* Flask
	* Bootstrap
	* Plotly

## Files
* NLP code 
	* code/data-pipeline.py
	* code/train_pipeline.py
* Web App
	* index.html
	* go.html 
	* run.py
	* wrangling.py

## Conclusions

Here is what we learned:

* Text classification depends on the quality of NLP
processing. 
* GridSearch is an important tool for pipeline
optmization, specially concerning NLP hyperparameters
* The dataset is imbalanced therefore accuracy is
not a good metric. F1 score was used as the most
important metric. Prediction assessment was done on a per-category basis
* SMOTE was used in an attempt to compensate for class
imbalances but the improvement was not relevant to
justify its use
