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

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

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

    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


if __name__ == '__main__':

    path = os.getcwd() + r"\Dataset"
    dataset_file = path + r"\dataset_test.txt"
    label_file = path + r"\label_test.txt"
    dataset = np.loadtxt(dataset_file)
    label = np.loadtxt(label_file)

    # we can define the estimator by ourselves
    estimator= svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    X, y = dataset,label

    title = "Learning Curves (SVM)"
    # Cross validation with 50 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)

    plot_learning_curve(
        estimator, title, X, y, axes=axes, ylim=(0.5, 1.01), cv=cv, n_jobs=4
    )

    plt.show()



    # load dataset and label from dataset folder
    # all_x, all_y, all_z, all_point_list, points_per_object = fh.read()
    # object_values, object_size = pre.design_features(points_per_object)
    # dataset, label = pre.get_normalized_dataset_label(object_values)

