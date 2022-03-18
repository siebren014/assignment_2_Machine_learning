import laspy
import numpy as np
import os
import random

#type to hold info of our objects
class point_type:
    def __init__(self, point):
        self.point = (point[0], point[1], point[2], point[3], point[4], point[5])
        self.set = 0 #1 for training, 2 for test
        self.cluster = 0 #to hold cluster value of the point object.
        self.original_index = 0 #hold original index for points to later link back to the actual point object.
        self.original_value = '' #car, buidling etc.


# write to file
def file_write(all_x, all_y, all_z, clusters, output_file):
    # Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.float32))
    # header.offsets = np.min(accumulated_objects, axis=0)
    header.scales = np.array([0.1, 0.1, 0.1])

    # Create a Las
    las = laspy.LasData(header)

    # create new dimension for cluster values
    las.add_extra_dim(laspy.ExtraBytesParams(
        name="clusters",
        type=np.uint64,
    ))

    # add all coordinates to the las file
    las.x = np.array(all_x).astype(np.float64)
    las.y = np.array(all_y).astype(np.float64)
    las.z = np.array(all_z).astype(np.float64)
    las.clusters = clusters

    # write the output to the file
    las.write(output_file)

#function to read the file and give the info needed
def read():
    input_folder = (r"C:scene_objects/scene_objects/data")
    all_point_list = []
    points_per_object = []
    all_x = []
    all_y = []
    all_z = []
    # loop trough files and retreive objects info as well as loose point info for output
    for file in os.listdir(input_folder):
        file_points = []
        with open(os.path.join(input_folder, file), 'r') as f:
            for points in f:
                split = points.split()
                all_x.append(split[0])
                all_y.append(split[1])
                all_z.append(split[2])
                all_point_list.append(split)
                file_points.append(split)
            points_per_object.append(file_points)
    return all_x, all_y, all_z, all_point_list, points_per_object

#function to normalize features
def normalize(features):
    max = features[0]
    for i in features:
        if i[0] > max:
            max = i[0]

    for j in features:
        j[0] = j[0] / max
    return features

#function to go back from objects to points, for the output.
def from_object2point(object_size, cluster_output):
    # list to hold objects * cluster value, to make the output ponitcloud
    db_cluster_value = []
    # For the amount of points in each object, add that amount of points in the cluster_value list.
    for multiplier in object_size:
        for i in range(multiplier):
            db_cluster_value.append(cluster_output[object_size.index(multiplier)])
    return db_cluster_value

#function to randomly split the objects in training and testing set.
def randomlysplit(object_points):
    #make a set out of the objects, seperate them in a test and training set
    s = set(object_points)
    training_set = set(random.sample(s, (round(len(object_points) * 0.6))))

    #list for training and test objects
    training_list = []
    test_list = []
    i = 0
    #assign values to point_type accordingly
    for point in object_points:
        original_index = object_points.index(point)
        point = point_type(point)
        point.original_index = original_index
        if point in training_set:
            point.set = 1
            training_list.append(point)
        if point not in training_set:
            point.set = 2
            test_list.append(point)
        if i < 100:
            point.original_value = 'building'
        elif 99 < i < 200:
            point.original_value = 'car'
        elif 199 < i < 300:
            point.original_value = 'fence'
        elif 299 < i < 400:
            point.original_value = 'pole'
        elif 399 < i < 500:
            point.original_value = 'tree'
        i += 1

    return object_points, training_list, test_list
