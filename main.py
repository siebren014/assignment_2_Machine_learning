import features as f
import file_handling as fh
import evaluation as eval
import os
import sklearn as skl
import numpy as np

if __name__ == '__main__':
    # read the input file
    all_x, all_y, all_z, all_point_list, points_per_object = fh.read()

    # strings for output
    svm_string = "SVM_output.las"
    rf_string = "RF_output.las"

    # values for each of the object
    object_values = []

    # object size of each object
    object_size = []
    normalized_object_values = []

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

        # calculate normalized features
        # area_norm = fh.normalize(area)
        # volume_norm = fh.normalize(volume)
        # height_diff_norm = fh.normalize(height_diff)
        # dens_norm = fh.normalize(dens)
        # leng_norm = fh.normalize(leng)
        # squared_norm = fh.normalize(squared)
        # vol_dens_norm = fh.normalize(vol_dens)
        # top_ratio_norm = fh.normalize(top_ratio)
        # bottom_ratio_norm = fh.normalize(bottom_ratio)
        # height_area_ratio_norm = fh.normalize(height_area_ratio)
        # middle_norm = fh.normalize(middle)

        # append the values to object_values to perform the clustering on
        # object_values.append([dens, vol_dens, top_ratio, bottom_ratio, height_area_ratio, middle])

        # normalized_object_values.append(
        #     [dens_norm, vol_dens_norm, top_ratio_norm, bottom_ratio_norm, height_area_ratio_norm, middle_norm])
        # # append the amount of points of the objects, later we can use this amount of points to link points to a cluster
        # object_size.append(len(object))
        
    normalized_object_values = fh.object_normalized(object_values)
    # object values for each object
    object_points = np.array(object_values).astype(np.float64)
    n_object_points = np.array(normalized_object_values).astype(np.float64)

    # split the data set into two data set and the label set:
    training_set, label, testing_set = fh.randomly_split(n_object_points)

    training_set = np.array(training_set).astype(np.float64)
    label = np.array(label).astype(np.float64)
    testing_set = np.array(testing_set).astype(np.float64)

    # random forest training and classifying
    clf = skl.ensemble.RandomForestClassifier(random_state=0)
    clf.fit(training_set, label)
    clf.predict(testing_set)
    print(clf.predict(testing_set))

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
