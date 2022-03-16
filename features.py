from scipy.spatial import ConvexHull
from statistics import median

def compute_height_difference(point_list):
    z_max, z_min = float(point_list[0][2]), float(point_list[0][2])
    for point in point_list:
        if float(point[2]) > z_max:
            z_max = float(point[2])
        if float(point[2]) < z_min:
            z_min = float(point[2])
    return z_max -  z_min

#function to compute ratio of points and points on top
def points_ontop(point_list):
    high_point = point_list[0][2]
    z_values = []
    for point in point_list:
        if point[2] > high_point:
            high_point = point[2]

    points_at_top =[]
    for point in point_list:
        if abs(float(high_point) - float(point[2])) < 0.4:
            points_at_top.append(point[2])

    ratio_top = len(points_at_top)/ len(point_list)
    return float(ratio_top)

#function to compute ratio of points and points in middle
def points_inmiddle(point_list):
    z_values = []
    for point in point_list:
        z_values.append(float(point[2]))

    res = median(z_values)
    points_at_middle =[]

    for point in point_list:
        if abs(float(res) - float(point[2])) < 0.4:
            points_at_middle.append(point[2])

    ratio_middle = len(points_at_middle)/ len(point_list)
    return float(ratio_middle)

#function to compute ratio of points and points on bottom
def points_onbottom(point_list):
    low_point = point_list[0][2]
    z_values = []
    for point in point_list:
        if point[2] > low_point:
            low_point = point[2]

    points_at_bottom =[]
    for point in point_list:
        if abs(float(low_point) - float(point[2])) < 0.4:
            points_at_bottom.append(point[2])

    ratio_bottom = len(points_at_bottom) / len(point_list)
    return float(ratio_bottom)

#function to compute convex hull area
def convex_hull_area(point_list):
    hull = ConvexHull(point_list)
    return hull.area

#function to compute convex hull volume
def convex_hull_volume(point_list):
    hull = ConvexHull(point_list)
    return hull.volume

#function to compute density of points in an object
def density (object, area):
    dens = (area / len(object))
    return dens

#function to compute density of points of the volume
def vol_density (object, volume):
    dens = (volume / len(object))
    return dens

#function to compute squareness of object
def squareness (object):
    x_max, x_min = float(object[0][0]), float(object[0][0])
    y_max, y_min = float(object[0][1]), float(object[0][1])

    for point in object:
        if float(point[0]) > x_max:
            x_max = float(point[0])
        if float(point[0]) < x_min:
            x_min = float(point[0])
        if float(point[1]) > y_max:
            y_max = float(point[1])
        if float(point[1]) < y_min:
            y_min = float(point[1])

    x_dis = x_max - x_min
    y_dis = y_max - y_min
    if x_dis > y_dis:
        return x_dis /y_dis
    else:
        return y_dis / x_dis

#function to compute length of object
def length(object):
    x_max, x_min = float(object[0][0]), float(object[0][0])
    y_max, y_min = float(object[0][1]), float(object[0][1])

    for point in object:
        if float(point[0]) > x_max:
            x_max = float(point[0])
        if float(point[0]) < x_min:
            x_min = float(point[0])
        if float(point[1]) > y_max:
            y_max = float(point[1])
        if float(point[1]) < y_min:
            y_min = float(point[1])

    x_dis = x_max - x_min
    y_dis = y_max - y_min
    if x_dis > y_dis:
        return x_dis
    else:
        return y_dis
