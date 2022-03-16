import laspy
import numpy as np
import os

def file_write(all_x, all_y, all_z, clusters, output_file):
    # write to file
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