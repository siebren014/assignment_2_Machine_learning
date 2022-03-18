from statistics import mode

def overall_accuracy(pointlist):
    i = 0
    correctness = 0
    for point in pointlist:
        if i < 100:
            if point.original_value == 'building':
                correctness += 1
        elif 99 < i < 200:
            if point.original_value == 'car':
                correctness += 1
        elif 199 < i < 300:
            if point.original_value == 'fence':
                correctness += 1
        elif 299 < i < 400:
            if point.original_value == 'pole':
                correctness += 1
        elif 399 < i < 500:
            if point.original_value == 'tree':
                correctness += 1
        i += 1
    print("overall accuracy = ", correctness / len(pointlist) * 100, "%")

def mean_per_class_accuracy(pointlist):
    i = 0
    building_correct = 0
    car_correct = 0
    fence_correct = 0
    pole_correct = 0
    tree_correct = 0

    for point in pointlist:
        if i < 100:
            if point.original_value == 'building':
                building_correct += 1
        elif 99 < i < 200:
            if point.original_value == 'car':
                car_correct += 1
        elif 199 < i < 300:
            if point.original_value == 'fence':
                fence_correct += 1
        elif 299 < i < 400:
            if point.original_value == 'pole':
                pole_correct += 1
        elif 399 < i < 500:
            if point.original_value == 'tree':
                tree_correct += 1
        i += 1
    print("building accuracy = ", building_correct, "%")
    print("car accuracy = ", car_correct, "%")
    print("fence accuracy = ", fence_correct, "%")
    print("pole accuracy = ", pole_correct, "%")
    print("tree accuracy = ", tree_correct, "%")

def confusion_matric():
    return 0
