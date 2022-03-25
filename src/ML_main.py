import os
import numpy as np

from sklearn import svm # svm classification
from sklearn.ensemble import RandomForestClassifier as RF # random forest classifications
from sklearn.model_selection import train_test_split # train and test dataset 
from sklearn.metrics import confusion_matrix # confusion matrix
from sklearn.metrics import classification_report # classification report
from sklearn.decomposition import PCA # PCA analysis, for features
import sklearn.model_selection as ms # cross validation
from sklearn.manifold import TSNE


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import evaluation as e

# perform SVM
def ml_svm(dataset, label):
    # prepare the train and test data set
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.4, random_state=0)
   
    # perform svm
    # --------------------------------------------------------------------
    print("svm classification --------------------------------------------------------------------")

    # build the svm classifier
    clf_svm = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr') # classifier
   
    # cross validation
    # accuracy
    cross_ac_svm = ms.cross_val_score(clf_svm, X_train, y_train, cv=5, scoring='accuracy')
    print("cross validation of svm: ")
    print("cross validation accuracy (svm): ",  cross_ac_svm.mean())
    print()

    print("train the svm classifier: ")
    clf_svm.fit(X_train, y_train.ravel())  # train the classifier with train dataset

    print("statistics: ")
    print("Training accuracy:" + str(clf_svm.score(X_train,y_train)))
    print("Test accuracy:" + str(clf_svm.score(X_test,y_test)))
   
    # get predicted label
    y_pred_svm = clf_svm.predict(X_test)

    # get confusion matrix
    Confusion_matrix_svm = confusion_matrix(y_test, y_pred_svm)
    print("confusion matrix of svm: ")
    print(Confusion_matrix_svm)
    
    # overall accuracy and mean accuracy
    oa=e.OA(Confusion_matrix_svm)
    print({}{}.format("overall accuracy of svm: ",oa))
    ma=e.mA(Confusion_matrix_svm)
    print({}{}.format("mean accuracy of svm: ",ma))
    
    print("classification report: ")
    print(classification_report(y_test, clf_svm.predict(X_test)))
    print()

# perform RF
def ml_RF(dataset, label):
    # prepare the train and test data set
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.4, random_state=0)

    print("Random Forest classification ------------------------------------------------------------")
    clf_rf = RF(n_estimators=100,n_jobs=2) # a little overfitting

    cross_ac_rf = ms.cross_val_score(clf_rf, X_train, y_train, cv=5, scoring='accuracy')
    print("cross validation of RF: ")
    print("cross validation accuracy (RF): ",  cross_ac_rf.mean())
    print()

    print("train the RF classifier: ")
    clf_rf.fit(X_train, y_train.ravel()) # fit the data

    print("statistics: ")
    print("Training accuracy:" + str(clf_rf.score(X_train,y_train)))
    print("Test accuracy:" + str(clf_rf.score(X_test,y_test)))
    
    # get predicted label
    y_pred_rf = clf_rf.predict(X_test)

    # get confusion matrix
    Confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)
    print("confusion matrix of Random Forest: ")
    print(Confusion_matrix_rf)
    
    # overall accuracy and mean accuracy
    oa=e.OA(Confusion_matrix_rf)
    print("{}{}".format("overall accuracy of rf: ",oa))
    ma=e.mA(Confusion_matrix_rf)
    print("{}{}".format("mean accuracy of rf: ",ma))
    
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

# plot selected 3 attributes of dataset
def plot_dataset(selected_dataset):
    # plot dataset with 3 attributes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(selected_dataset[:, 0], selected_dataset[:, 1],  selected_dataset[:, 2], marker='o')
    plt.savefig(os.getcwd() + r"\Figure" + "\dataset_3_features.png")
    plt.show()

# plot correlation between attributes
def plot_correlation_check(dataset):
    Data = pd.DataFrame(dataset)
    data = Data.corr()
    sns.heatmap(data,cmap='Blues',annot=True, square=True, fmt='.2g')
    plt.savefig(os.getcwd() + r"\Figure" + "\correlation.png")
    plt.show()

# plot the violin plot to observe the selected features
def plot_violin_features(both, feature_id):
    Data = pd.DataFrame(both) # convert to df to get one column
    sns.violinplot(x=6,y=feature_id,data=Data) 
    plt.show()

# plot t-SNE
def plot_t_sne(dataset):
    tsne = TSNE(n_components=2)
    tsne.fit_transform(dataset)
    print("data after dimensionality reduction")
    # print(tsne.embedding_)
    plt.scatter(tsne.embedding_[:,0],tsne.embedding_[:,1])
    plt.show()



if __name__ == '__main__':

   # load dataset and label from dataset folder
   path = os.getcwd() + r"\Dataset" 
   dataset_file = path + r"\dataset.txt" 
   label_file = path + r"\label.txt" 
   dataset = np.loadtxt(dataset_file)
   label = np.loadtxt(label_file)
    
   selected_dataset_file = path + r"\dataset.txt" 
   selected_label_file = path + r"\label.txt" 
   we_selected_dataset = np.loadtxt(selected_dataset_file)
   we_selected_label = np.loadtxt(selected_label_file)
    

   # load dataset with labels
   both_file = path + r"\both.txt" 
   both =  np.loadtxt(both_file)
   
   # plot using t-SNE
   # plot_t_sne(dataset) # uncomment this to plot the t-SNE figure

   # SVM
   ml_svm(dataset, label)

   # RF
   ml_RF(dataset, label)
   
   # PCA analysis
   selected_dataset = pca_analysis(dataset)

   print("SVM and Random Forest for dataset with PCA selected attributes: ")
   print()

   # SVM
   ml_svm(selected_dataset, label)

   # RF
   ml_RF(selected_dataset, label)

   print("SVM and Random Forest for dataset with 3 selected attributes: ")
   print()

   # SVM
   ml_svm(we_selected_dataset, we_selected_label)

   # RF
   ml_RF(we_selected_dataset, we_selected_label)
   # plot dataset with 3 selected attributes
   # plot_dataset(selected_dataset) # uncomment this to plot the dataset with 3 attributes in 3D space

   # correlation check
   # plot_correlation_check(dataset) # uncomment this to plot the correlation matrix of the dataset

   # plot violin using feature_id : 0, 1, 2, 3, 4, 5 -- indicating 6 features
   # x-axis: 1, 2, 3, 4, 5 -- indicating five categories: building, car, fence, pole, tree
   # both: dataset with labels, the first 6 columns are features, the 7th column is the labels
   # plot_violin_features(both, 2) # uncomment this to plot the violin graph for a feature

 
   
  

