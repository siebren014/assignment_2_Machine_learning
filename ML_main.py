import os
import numpy as np

from sklearn import svm # svm classification
from sklearn.ensemble import RandomForestClassifier as RF # random forest classifications
from sklearn.model_selection import train_test_split # train and test dataset 
from sklearn.metrics import confusion_matrix # confusion matrix
from sklearn.metrics import classification_report # classification report
from sklearn.decomposition import PCA # PCA analysis, for features

import matplotlib.pyplot as plt

# proform SVM
def ml_svm(dataset, label):
    # prepare the train and test data set
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.4, random_state=0)
   
    # perform svm
    # --------------------------------------------------------------------
    print("svm classification --------------------------------------------------------------------")
    clf_svm = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf_svm.fit(X_train, y_train.ravel())

    print("statistics: ")
    print("Training accuracy:" + str(clf_svm.score(X_train,y_train)))
    print("Test accuracy:" + str(clf_svm.score(X_test,y_test)))

    # get predicted label
    y_pred_svm = clf_svm.predict(X_test)

    # get confusion matrix
    Confusion_matrix_svm = confusion_matrix(y_test, y_pred_svm)
    print("confusion matrix of svm: ")
    print(Confusion_matrix_svm)
    print("classification report: ")
    print(classification_report(y_test, clf_svm.predict(X_test)))
    print()

# perofrm RF
def ml_RF(dataset, label):
    # prepare the train and test data set
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.4, random_state=0)

    print("Random Forest classification ------------------------------------------------------------")
    clf_rf = RF(n_estimators=100,n_jobs=2) # a little overfitting
    clf_rf.fit(X_train, y_train.ravel())

    print("statistics: ")
    print("Training accuracy:" + str(clf_rf.score(X_train,y_train)))
    print("Test accuracy:" + str(clf_rf.score(X_test,y_test)))

    # get predicted label
    y_pred_rf = clf_rf.predict(X_test)

    # get confusion matrix
    Confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)
    print("confusion matrix of Random Forest: ")
    print(Confusion_matrix_rf)
    print("feature importances: ")
    print(clf_rf.feature_importances_) # maybe helpful to select features
    print("classification report: ")
    print(classification_report(y_test, clf_rf.predict(X_test)))
    print()

# PCA analysis, return new dataset with 3 selected attributes
def pca_analysis(dataset):
    # PCA
    print("PCA analysis")
    pca = PCA(n_components=6) # original 6 features
    pca_result = pca.fit(dataset)
    print("variance ratio of 6 features: ")
    print(pca.explained_variance_ratio_)
    print("variance of 6 features: ")
    print(pca.explained_variance_) # the variance of the first feature is too big
    print()
   
    print("select 3 features: ")
    pca = PCA(n_components=3) # selecting features
    pca_result = pca.fit(dataset)
    print("variance ratio of 3 features: ")
    print(pca.explained_variance_ratio_)
    print("variance of 3 features: ")
    print(pca.explained_variance_)
    print()

    selected_dataset = pca.transform(dataset) # dataset with 3 selected features
    return selected_dataset

if __name__ == '__main__':
   
   # load dataset and label from dataset folder
   path = os.getcwd() + r"\dataset" 
   dataset_file = path + r"\dataset.txt" 
   label_file = path + r"\label.txt" 
   dataset = np.loadtxt(dataset_file)
   label = np.loadtxt(label_file)

   # SVM
   ml_svm(dataset, label)

   # RF
   ml_RF(dataset, label)
   
   # PCA analysis
   selected_dataset = pca_analysis(dataset)

   print("SVM and Random Forest for dataset with 3 selected attributes: ")
   print()

   # SVM
   ml_svm(selected_dataset, label)

   # RF
   ml_RF(selected_dataset, label)

   # plot dataset with 3 attributes
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   ax.scatter(selected_dataset[:, 0], selected_dataset[:, 1],  selected_dataset[:, 2], marker='o')
   plt.show()
  

