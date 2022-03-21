
# store the constructed objects and label as txt files to: dataset folder

import numpy as np
import os

def store_dataset_as_txt(dataset):
    path = os.getcwd() + r"\dataset" 
    file_name = path + r"\dataset.txt" 
    np.savetxt(file_name, dataset)

def store_label_as_txt(label):
    path = os.getcwd() + r"\dataset" 
    file_name = path + r"\label.txt" 
    np.savetxt(file_name, label)

