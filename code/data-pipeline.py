# import packages
import sys
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from sqlalchemy import create_engine

def load_data(data_file):
    # read in file
    try:
        file = pd.read_csv(data_file)
    except:
        file = None
    return file

def str_to_dict(x):
    list_ = re.split('\W+',x)
    return {key:value for key, value in zip(list_[0:][::2], list_[1:][::2])}

def clean_categories(categories):
    df_ = pd.DataFrame(list(categories.categories.map(lambda x : str_to_dict(x))))
    return pd.concat([categories['id'], df_], axis=1)

def merge_datasets(messages, categories_clean)
    return messages.merge(categories_clean, on=('id')).drop_duplicates()

def load_to_db(df, db_name):
    # load to database
    engine = create_engine('sqlite:///{}'.format(db_name), echo=False)
    df.to_sql('messages_categories',engine, index=False)

def generate_X(data_file):
    pass

def generate_y(data_file):
    pass
    


    # define features and label arrays


#    return X, y


def build_model():
    # text processing and model pipeline


    # define parameters for GridSearchCV


    # create gridsearch object and return as final model pipeline


    return model_pipeline


def train(X, y, model):
    # train test split


    # fit model


    # output model test results


    return model


def export_model(model):
    # Export model as a pickle file



def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline
