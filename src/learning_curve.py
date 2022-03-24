import os
import numpy as np

import sklearn
from sklearn import svm  # svm classification
from sklearn.ensemble import RandomForestClassifier as RF  # random forest classifications
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve  # train and test dataset
from sklearn.metrics import confusion_matrix  # confusion matrix
from sklearn.metrics import classification_report  # classification report
from sklearn.decomposition import PCA  # PCA analysis, for features
import sklearn.model_selection as ms  # cross validation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import Pre_main as pre
import file_handling as fh
import ML_main as ml

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    axes.grid()
    axes.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Overall accuracy"
    )
    axes.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation accuracy"
    )
    axes.legend(loc="best")
    return plt

if __name__ == '__main__':

    path = os.getcwd() + r"\Dataset"
    dataset_file = path + r"\dataset_test.txt"
    label_file = path + r"\label_test.txt"
    dataset = np.loadtxt(dataset_file)
    label = np.loadtxt(label_file)

    # we can define the estimator by ourselves
    # use different estimator to generator different figures
    # estimator= svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    estimator=RF(n_estimators=50, n_jobs=2)

    fig, axes = plt.subplots(1, 1, figsize=(10, 15))

    X, y = dataset,label

    title = "Learning Curves"
    # Cross validation with 50 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)

    plot_learning_curve(
        estimator, title, X, y, axes=axes, ylim=(0.6, 1.01), cv=cv, n_jobs=4
    )

    plt.show()

