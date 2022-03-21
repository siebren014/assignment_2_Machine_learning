# Siebren Meines 4880412
# Fengyan
# Yitong

import features as f # design features
import file_handling as fh # read and write files
import numpy as np # to help store the objects
import os

# from sklearn import svm # svm classification
# from sklearn.ensemble import RandomForestClassifier as RF # random forest classifications
# from sklearn.model_selection import train_test_split # train and test dataset 
# from sklearn.metrics import confusion_matrix # confusion matrix

import ML_dataset as ML

# design features here
# return:
# object_values -- 500 objects, not normalized yet
# object_size -- object size of each object
def design_features(points_per_object):
    # values for each of the object
    object_values = []

    # object size of each object
    object_size = []
    
    # calculate features
    for object in points_per_object:
        # calculate different features
        area = f.convex_hull_area(object)
        volume = f.convex_hull_volume(object)
        height_diff = f.compute_height_difference(object)
        dens = f.density(object, area)
        leng = f.length(object)
        squared = f.squareness(object)
        vol_dens = f.vol_density(object, volume)
        top_ratio = f.points_ontop(object)
        bottom_ratio = f.points_onbottom(object)
        height_area_ratio = area / height_diff
        middle = f.points_inmiddle(object)

        # append the values to object_values to perform the clustering on
        object_values.append([dens, vol_dens, top_ratio, bottom_ratio, height_area_ratio, middle])
       
        # append the amount of points of the objects, later we can use this amount of points to link points to a cluster
        object_size.append(len(object))

    return object_values, object_size

# get normalized dataset and label
# return:
# dataset, label
def get_normalized_dataset_label(object_values):

    # 500 objects with 6 normalized attributes
    normalized_object_values = fh.object_normalized(object_values)

    # convert it to float 64
    dataset =  np.array(normalized_object_values).astype(np.float64)

    # get labels of 500 objects
    all_label = fh.get_label(normalized_object_values)

    #convert it to float 64
    label = np.array(all_label).astype(np.float64)

    return dataset, label


if __name__ == '__main__':
    # read the input file
    all_x, all_y, all_z, all_point_list, points_per_object = fh.read()

    # strings for output
    svm_string = "SVM_output.las"
    rf_string = "RF_output.las"

    # ovject_values -- values for each of the object(NOT normalized objects)
    # object_size -- object size of each object
    object_values, object_size = design_features(points_per_object)

    # get normalized dataset and label
    # dataset -- 500 x 6 2d array, contains 500 objects, each object has 6 attributes
    # label -- 500 x 1 1d array, contains 500 labels for each object      
    dataset, label = get_normalized_dataset_label(object_values)     

    # store dataset and label, load it directly to shorten the running time
    ML.store_dataset_as_txt(dataset) # ML: ML_dataset.py
    ML.store_label_as_txt(label)

    print("dataset constructed, stored in: ", end = " ")
    print(os.getcwd() + r"\Dataset" + r"\dataset.txt" )

    print("label constructed, stored in: ", end = " ")
    print(os.getcwd() + r"\Dataset" + r"\label.txt" )

    #-------------------------------------------------------------------
    
    # From president of GEOS --------------------------------------------

    # object values for each object
    # object_points = np.array(object_values).astype(np.float64)

    # split the data set into two data set and the label set:
    # training_set, label, testing_set, help_objects = fh.randomly_split(normalized_object_values)

    # training_set = np.array(training_set).astype(np.float64)
    # label = np.array(label).astype(np.float64)
    # testing_set = np.array(testing_set).astype(np.float64)

    # random forest training and classifying
    # clf = RF(random_state=0)
    # clf.fit(training_set, label)
    # rf_predict = clf.predict(testing_set)
    # print("random forest prediction: ")
    # print(rf_predict)

    # SVM training and classifying
    # support_vm = svm.SVC(kernel='linear')  # Linear Kernel
    # support_vm.fit(training_set, label)
    # svm_predict = support_vm.predict(testing_set)
    # print("SVM prediction: ")
    # print(svm_predict)

    # From president of GEOS --------------------------------------------


    # construct later
    # ##evaluate the classification
    # eval.overall_accuracy(svm_cluster_output)
    # eval.mean_per_class_accuracy(svm_cluster_output)
    # eval.confusion_matrix(svm_cluster_output)
    # eval.overall_accuracy(rf_cluster_output)
    # eval.mean_per_class_accuracy(rf_cluster_output)
    # eval.confusion_matrix(rf_cluster_output)
    #
    # ##to go from the objects back to the points.
    # svm_point_output = fh.from_object2point(object_size, svm_cluster_output)
    # rf_point_output = fh.from_object2point(object_size, rf_cluster_output)

    # ## to write the files
    # fh.file_write(all_x, all_y, all_z, svm_point_output, svm_string)
    # fh.file_write(all_x, all_y, all_z, rf_point_output, rf_string)