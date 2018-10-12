import utils.filehandler as filehandler
import utils.learning as lrn
import time
import numpy as np

#for k nearest neighbor
from sklearn.neighbors import KNeighborsClassifier

#for feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# for confusion matrix
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

#for diagrams
import matplotlib.pyplot as plt

"""
IMPORTING DATA FROM CSV FILE
"""
x_train, y_train, x_validation, y_validation = filehandler.get_data('../fer2018/fer2018.csv')


"""
CLASSIFICATION WITH NEAREST NEIGHBOR ALGORITHM
"""
# Classifier initialization
knc = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3)

preds = lrn.get_knn_preds(knc, x_train, y_train, x_validation)


"""
CLASSIFICATION ACCURACY CALCULATION
"""
lrn.overall_accuracy(preds, y_validation)
lrn.accuracy_plots(preds, y_validation)


"""
BEST FEATURE COLLECTION
"""
# Importing data from each individual emotion file
x_angry, y_angry = filehandler.get_data('../fer2018/fer2018angry.csv', sep=False)
x_disgust, y_disgust = filehandler.get_data('../fer2018/fer2018disgust.csv', sep=False)
x_fear, y_fear = filehandler.get_data('../fer2018/fer2018fear.csv', sep=False)
x_happy, y_happy = filehandler.get_data('../fer2018/fer2018happy.csv', sep=False)
x_sad, y_sad = filehandler.get_data('../fer2018/fer2018sad.csv', sep=False)
x_surprise, y_surprise = filehandler.get_data('../fer2018/fer2018surprise.csv', sep=False)
x_neutral, y_neutral = filehandler.get_data('../fer2018/fer2018neutral.csv', sep=False)

# Search for top TEN attributes for each emotion
print("Searching for top 10 attributes...")
x_angry_10 = lrn.get_k_best(x_angry, y_angry, 10)
x_disgust_10 = lrn.get_k_best(x_disgust, y_disgust, 10)
x_fear_10 = lrn.get_k_best(x_fear, y_fear, 10)
x_happy_10 = lrn.get_k_best(x_happy, y_happy, 10)
x_sad_10 = lrn.get_k_best(x_sad, y_sad, 10)
x_surprise_10 = lrn.get_k_best(x_surprise, y_surprise, 10)
x_neutral_10 = lrn.get_k_best(x_neutral, y_neutral, 10)

# Search for top FIVE attributes for each emotion
print("Searching for top 5 attributes...")
x_angry_5 = lrn.get_k_best(x_angry, y_angry, 5)
x_disgust_5 = lrn.get_k_best(x_disgust, y_disgust, 5)
x_fear_5 = lrn.get_k_best(x_fear, y_fear, 5)
x_happy_5 = lrn.get_k_best(x_happy, y_happy, 5)
x_sad_5 = lrn.get_k_best(x_sad, y_sad, 5)
x_surprise_5 = lrn.get_k_best(x_surprise, y_surprise, 5)
x_neutral_5 = lrn.get_k_best(x_neutral, y_neutral, 5)

# Search for top TWO attributes for each emotion
print("Searching for top 2 attributes...")
x_angry_2 = lrn.get_k_best(x_angry, y_angry, 2)
x_disgust_2 = lrn.get_k_best(x_disgust, y_disgust, 2)
x_fear_2 = lrn.get_k_best(x_fear, y_fear, 2)
x_happy_2 = lrn.get_k_best(x_happy, y_happy, 2)
x_sad_2 = lrn.get_k_best(x_sad, y_sad, 2)
x_surprise_2 = lrn.get_k_best(x_surprise, y_surprise, 2)
x_neutral_2 = lrn.get_k_best(x_neutral, y_neutral, 2)


"""
DATASET REDUCTION
"""
# 14, 35, and 70 non-class attribute reduction:
top_attrs = {2 : np.array([x_angry_2, x_disgust_2, x_fear_2, x_happy_2, x_sad_2, x_surprise_2, x_neutral_2]).flatten(),
        	5 : np.array([x_angry_5, x_disgust_5, x_fear_5, x_happy_5, x_sad_5, x_surprise_5, x_neutral_5]).flatten(),
        	10 : np.array([x_angry_10, x_disgust_10, x_fear_10, x_happy_10, x_sad_10, x_surprise_10, x_neutral_10]).flatten()}

print("Reducing datasets...")
x_train_70 = lrn.reduce_attr(x_train, top_attrs[10])
x_train_35 = lrn.reduce_attr(x_train, top_attrs[5])
x_train_14 = lrn.reduce_attr(x_train, top_attrs[2])
x_validation_70 = lrn.reduce_attr(x_validation, top_attrs[10])
x_validation_35 = lrn.reduce_attr(x_validation, top_attrs[5])
x_validation_14 = lrn.reduce_attr(x_validation, top_attrs[2])


"""
ATTEMPT TO IMPROVE CLASSIFICATION WITH NEW DATASETS
"""
print("70 attribute classification")
preds = lrn.get_knn_preds(knc, x_train_70, y_train, x_validation_70)
lrn.overall_accuracy(preds, y_validation)
lrn.accuracy_plots(preds, y_validation)

print("35 attribute classification")
preds = lrn.get_knn_preds(knc, x_train_35, y_train, x_validation_35)
lrn.overall_accuracy(preds, y_validation)
lrn.accuracy_plots(preds, y_validation)

print("14 attribute classification")
preds = lrn.get_knn_preds(knc, x_train_14, y_train, x_validation_14)
lrn.overall_accuracy(preds, y_validation)
lrn.accuracy_plots(preds, y_validation)


"""
DATASET REDUCTION BASED ON CLASS
"""
# 2, 5, and 10 non-class attribute reduction:
# the top attributes of each specific emotion
top_attrs_emo = {2 : [x_angry_2, x_disgust_2, x_fear_2, x_happy_2, x_sad_2, x_surprise_2, x_neutral_2],
                5 : [x_angry_5, x_disgust_5, x_fear_5, x_happy_5, x_sad_5, x_surprise_5, x_neutral_5],
                10 : [x_angry_10, x_disgust_10, x_fear_10, x_happy_10, x_sad_10, x_surprise_10, x_neutral_10]}

# transforming the overall emotion dataset
x_train_10 = lrn.reduce_data_emo(x_train, y_train, 10)
x_train_5 = lrn.reduce_data_emo(x_train, y_train, 5)
x_train_2 = lrn.reduce_data_emo(x_train, y_train, 2)
x_validation_10 = lrn.reduce_data_emo(x_validation, y_validation, 10)
x_validation_5 = lrn.reduce_data_emo(x_validation, y_validation, 5)
x_validation_2 = lrn.reduce_data_emo(x_validation, y_validation, 2)


"""
2nd ATTEMPT TO IMPROVE CLASSIFICATION WITH NEW DATASETS
"""
print("10 attribute classification")
preds = lrn.get_knn_preds(knc, x_train_10, y_train, x_validation_10)
lrn.overall_accuracy(preds, y_validation)
lrn.accuracy_plots(preds, y_validation)

print("5 attribute classification")
preds = lrn.get_knn_preds(knc, x_train_5, y_train, x_validation_5)
lrn.overall_accuracy(preds, y_validation)
lrn.accuracy_plots(preds, y_validation)

print("2 attribute classification")
preds = lrn.get_knn_preds(knc, x_train_2, y_train, x_validation_2)
lrn.overall_accuracy(preds, y_validation)
lrn.accuracy_plots(preds, y_validation)