# assignment_2_Machine_learning
Machine learning environment for Siebren, Yitong & Fengyan;

# Info
Each run needs to establish `objects` from more than one million points of [point cloud files](https://github.com/siebren014/assignment_2_Machine_learning/tree/master/scene_objects/scene_objects/data), and then train and test the objects dataset, which is very time-consuming. Therefore, store the generated `500 objects with normalized features` as `dataset.txt` file and the corresponding `ground truth label` as `label.txt` file. 

Thus we can directly read these two files for `SVM` and `Random Forest` algorithm.

--> Features need to be changed?

Run [main.py](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/main.py) and txt files will be updated.

## [dataset](https://github.com/siebren014/assignment_2_Machine_learning/tree/master/dataset) folder:

[dataset.txt](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/dataset/dataset.txt)
-- store the 500 objects file(with 6 normalized features)

[label.txt](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/dataset/label.txt)
-- store the ground truth labels of 500 objects file

## [Figure](https://github.com/siebren014/assignment_2_Machine_learning/tree/master/Figure) folder

Store the pictures/screenshots which may be used in report.

## python files for performing algorithms

Relevant python files are entitled with "ML".

[ML_dataset.py](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/ML_dataset.py) -- functions to store 500 objects and labels as `.txt` files.

[ML_main.py](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/ML_main.py) -- set this as startup project and run.

## Tips

Relative path is used thus this project can be cloned and run directly without any modifications.

Before you run this project, you can find the packages needed: 
[requirements.txt](https://github.com/siebren014/assignment_2_Machine_learning/blob/master/requirements.txt)

Use `pip install -r requirements.txt` to install appropriate versions of all dependent packages if you haven't got them.

