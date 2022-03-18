from statistics import mode

##this is the one i copied from previous assignment, probably not yet correct.
def evaluation(object, name):
    outliers = []
    non_outliers = []
    for point in object:
        if point == 0:
            outliers.append(point)
        else:
            non_outliers.append(point)

    most_freq = mode(non_outliers)
    amount = object.count(most_freq)
    percentage = amount / len(non_outliers) * 100
    print(name, "outliers = ", len(outliers), "most frequent = ", most_freq, "percentage correct = ", percentage, "\n")


def overall_accuracy():
    return 0

def mean_per_class_accuracy():
    return 0

def confusion_matric():
    return 0
