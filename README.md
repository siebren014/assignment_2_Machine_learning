# assignment_2_Machine_learning
Machine learning environment for Siebren, Yitong & Fengyan;

## Basic Info
Each run needs to establish `objects` from more than one million points of [point cloud files](https://github.com/siebren014/assignment_2_Machine_learning/tree/master/scene_objects/scene_objects/data), and then train and test the objects dataset, which is very time-consuming. Therefore, store the generated `500 objects with normalized features` as `dataset.txt` file and the corresponding `ground truth label` as `label.txt` file. 

Thus we can directly read these two files for `SVM` and `Random Forest` algorithm.

All the `.py` files are in the [src](https://github.com/siebren014/assignment_2_Machine_learning/tree/master/src) folder.

## HOW TO USE

* In folder [Dataset](https://github.com/siebren014/assignment_2_Machine_learning/tree/master/dataset), 
[dataset.txt](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/dataset/dataset.txt) and 
[label.txt](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/dataset/label.txt) files should already exist.

* If not, run
[src\Pre_main.py](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/src/Pre_main.py) to build the `dataset.txt` and `label.txt`.

* If files already exist, run [src\ML_main.py](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/src/ML_main.py) to perform `SVM` and `Random Forest` classification.

* --> Features need to be changed?

* --> Change features in [src\Pre_main.py](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/src/Pre_main.py) and run it, `.txt` files will be updated.
Then use [src\ML_main.py](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/src/ML_main.py) to perform classifications.

## Dataset folder:

[dataset.txt](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/dataset/dataset.txt)
-- store the 500 objects file(with 6 normalized features)

[label.txt](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/dataset/label.txt)
-- store the ground truth labels of 500 objects file

## Figure folder

Store the pictures/screenshots which may be used in report.

[correlation.png](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/Figure/correlation.png) -- indicates the correlations between 6 attributes, as a reference for selecting attributes.

[dataset_3_features.png](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/Figure/dataset_3_features.png) -- shows the dataset with selected 3 attributes in 3D (attributes are selected using PCA).

## python files for performing algorithms

Relevant python files are entitled with "ML".

[ML_dataset.py](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/ML_dataset.py) -- functions to store 500 objects and labels as `.txt` files.

[ML_main.py](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/ML_main.py) -- set this as startup project and run.

## Tips

Relative path is used thus this project can be cloned and run directly without any modifications.

Before you run this project, you can find the packages needed: 
[requirements.txt](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/requirements.txt)

Use `pip install -r requirements.txt` to install appropriate versions of all dependent packages if you haven't got them.

