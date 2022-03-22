
# store the constructed objects and label as txt files to: dataset folder

import numpy as np
import os

def store_dataset_as_txt(dataset):
    path = os.getcwd() + r"\dataset" 
    file_name = path + r"\dataset.txt" 
    np.savetxt(file_name, dataset) # save as txt file
    print("dataset constructed, stored in: ", end = " ")
    print(file_name)

def store_label_as_txt(label):
    path = os.getcwd() + r"\dataset" 
    file_name = path + r"\label.txt" 
    np.savetxt(file_name, label) # save as txt file
    print("label constructed, stored in: ", end = " ")
    print(file_name)

def store_both_as_txt(dataset, label):
    dataset = dataset.tolist() # convert to list to use append
    for i in range(len(dataset)):
        dataset[i].append(label[i])
    dataset = np.array(dataset) # convert to array to save txt
    path = os.getcwd() + r"\dataset" 
    file_name = path + r"\both.txt" 
    np.savetxt(file_name, dataset)
    print("dataset with label constructed, stored in: ", end = " ")
    print(file_name)


