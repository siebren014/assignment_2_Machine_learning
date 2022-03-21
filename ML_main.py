import os
import numpy as np

from sklearn import svm # svm classification
from sklearn.ensemble import RandomForestClassifier as RF # random forest classifications
from sklearn.model_selection import train_test_split # train and test dataset 
from sklearn.metrics import confusion_matrix # confusion matrix

if __name__ == '__main__':
   
   # load dataset and label from dataset folder
   path = os.getcwd() + r"\dataset" 
   dataset_file = path + r"\dataset.txt" 
   label_file = path + r"\label.txt" 
   dataset = np.loadtxt(dataset_file)
   label = np.loadtxt(label_file)

   # prepare the train and test data set
   X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.4, random_state=0)
   
   # perform svm --------------------------------------------------------------------
   print("svm classification")
   clf_svm = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
   clf_svm.fit(X_train, y_train.ravel())

   print("statistics: ")
   print("Training accuracy:"+str(clf_svm.score(X_train,y_train)))
   print("Test accuracy:"+str(clf_svm.score(X_test,y_test)))

   # get predicted label
   y_pred_svm = clf_svm.predict(X_test)

   # get confusion matrix
   Confusion_matrix_svm = confusion_matrix(y_test, y_pred_svm)
   print("confusion matrix of svm: ")
   print(Confusion_matrix_svm)
   print()

   # perform Random Forest -----------------------------------------------------------
   print("Random Forest classification")
   clf_rf = RF(n_estimators=100,n_jobs=2) # a little overfitting
   clf_rf.fit(X_train, y_train.ravel())

   print("statistics: ")
   print("Training accuracy:"+str(clf_rf.score(X_train,y_train)))
   print("Test accuracy:"+str(clf_rf.score(X_test,y_test)))

   # get predicted label
   y_pred_rf = clf_rf.predict(X_test)

   # get confusion matrix
   Confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)
   print("confusion matrix of Random Forest: ")
   print(Confusion_matrix_rf)
   print("feature importances: ")
   print(clf_rf.feature_importances_)
   print()

