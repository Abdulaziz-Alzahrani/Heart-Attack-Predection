import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sqlalchemy
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, plot_confusion_matrix

def load_data():
    """
    load data from database and returns the dataframe.
    Parameters: None.
    
    Return: 
        - inputs.
        - targets.
    """
    engine = create_engine(f'sqlite:///Model/datasets.db')
    df = pd.read_sql_table('heart', engine)

    inputs = df.drop(['output'], axis=1)
    targets = df['output']
    return inputs, targets


def build_models():
    """
    build models using GridSearchCV using diffrent algorithms with diffrent hyperparameters
    and a cross validation of 4 K-folds.
    Parameters: None.
    Return: built models.
    """
    clfs = [
        LogisticRegression(),
        SVC(),
        KNeighborsClassifier(),
        RandomForestClassifier()
    ]
    params = [
        {'max_iter': [100, 200, 300], 'penalty':['l1', 'l2', 'none'], 'C':[0.1, 0.5, 1.0]},
        {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'C':[0.1, 0.5, 1, 5, 10]},
        {'n_neighbors':[*range(1,67)], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']},
        {'n_estimators':[100, 200, 300, 400, 500], 'criterion': ['gini', 'entropy']}
    ]
    models = []
    for clf in clfs:
        models.append(GridSearchCV(clf, param_grid=params[clfs.index(clf)], cv=4, verbose=3))
    return models

def split_train_test(X, Y):
    """
    split dataset into train and test where test is 10% of the original.
    Parameters: 
        - X: inputs
        - Y: targets

    Return: splitted dataset.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    return X_train, X_test, Y_train, Y_test

def train_models(models, X, Y):
    """
    train the given models.
    Parameters: 
        - models: list of models.
        - X: training set inputs.
        - Y: training set targets.

    Return: None.
    """
    for mdl in models:
        mdl.fit(X, Y)
        pickle.dump(mdl, open(f"Model/model-{models.index(mdl)}/model.pickle", "wb"))

def evaluate_models(models, X_test, y_test):
    for mdl in models:
        plot_confusion_matrix(mdl, X_test, y_test,display_labels=['less chance', 'high chance'])  
        plt.savefig(f"Model/model-{models.index(mdl)}/model.png")
        plt.close()
        y_pred = mdl.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        with  open(f"Model/model-{models.index(mdl)}/metrics.txt", "w") as f:
            f.write(f'accuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}\nF1: {f1}')


def main():
    X, Y = load_data()
    print('data loaded!')
    X_train, X_test, Y_train, Y_test = split_train_test(X, Y)
    print('building models...')
    models = build_models()
    print('models built!')
    train_models(models, X_train, Y_train)
    models = []
    for i in range(4):
        models.append(pickle.load(open(f'Model/model-{i}/model.pickle','rb')))
    evaluate_models(models, X_test, Y_test)

if __name__ == '__main__':
    main()