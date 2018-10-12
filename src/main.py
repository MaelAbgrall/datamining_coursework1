import utils.filehandler as filehandler
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

def get_knn_preds(knc, x_train, y_train, x_test):
    """
    Runs the KNN algorithm and returns predictions
    """
    print("Fitting data...")
    start_time = time.time()
    knc.fit(x_train, y_train)
    print(time.time() - start_time, "seconds to fit data")
    
    print("Making predictions...")
    start_time = time.time()
    predictions = knc.predict(x_test)
    print(time.time() - start_time, "seconds to predict the classes")
    
    return predictions


def overall_accuracy(predictions, actual):
    """
    Overall accuracy calculation
    """
    total = len(predictions)
    correct_count = (predictions == actual).sum()
    print(correct_count, " out of ", total, " correctly predicted")
    accuracy = (correct_count / total) * 100
    print(accuracy, "% correctly predicted")
    

def disp_confusion_matrix(predictions, actual):
    """
    displays the confusion matrix
    """
    # confusion matrix
    confusion_mat = confusion_matrix(actual, predictions)
    
    df_cm = pd.DataFrame(confusion_mat, index = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
                  columns = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
    plt.figure(figsize = (8, 6))
    sn.heatmap(df_cm, annot=True, fmt="d", cmap='Greys')
    plt.show()
    plt.clf()
    return confusion_mat

    
def disp_accuracy_hist(cm, predictions, actual):
    """
    displays the accuracy histogram
    """
    # accuracy histogram
    acc = []
    for i in range(len(cm)):
        total_emote = cm[i].sum()
        acc.append(cm[i][i] / total_emote)

    bar_width = .5
    positions = np.arange(7)
    plt.bar(positions, acc, bar_width)
    plt.xticks(positions + bar_width / 2, ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'))
    plt.show()
    plt.clf()
    

def accuracy_plots(predictions, actual):
    """
    displays the confusion matrix and the accuracy histogram
    """
    disp_accuracy_hist(disp_confusion_matrix(predictions, actual), predictions, actual)


def get_k_best(X, y, k):
    """
    returns the inidices of the k best attributes for each emotion type
    """
    kbest = SelectKBest(f_regression, k)
    kbest.fit_transform(X, y)
    best_attributes = kbest.get_support(True)
    return best_attributes


def reduce_attr(data, attr_indices):
    """
    saves the data at certain indices (which are in attr_indices) into new
    """
    new = []
    for i in range(len(data)):
        new.append([data[i][index] for index in attr_indices])
    return new


"""
IMPORTING DATA FROM CSV FILE
"""
x_train, y_train, x_validation, y_validation = filehandler.get_data('../fer2018/fer2018.csv')


"""
CLASSIFICATION WITH NEAREST NEIGHBOR ALGORITHM
"""
# Classifier initialization
knc = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3)

preds = get_knn_preds(knc, x_train, y_train, x_validation)


"""
CLASSIFICATION ACCURACY CALCULATION
"""
overall_accuracy(preds, y_validation)
accuracy_plots(preds, y_validation)


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
x_angry_10 = get_k_best(x_angry, y_angry, 10)
x_disgust_10 = get_k_best(x_disgust, y_disgust, 10)
x_fear_10 = get_k_best(x_fear, y_fear, 10)
x_happy_10 = get_k_best(x_happy, y_happy, 10)
x_sad_10 = get_k_best(x_sad, y_sad, 10)
x_surprise_10 = get_k_best(x_surprise, y_surprise, 10)
x_neutral_10 = get_k_best(x_neutral, y_neutral, 10)

# Search for top FIVE attributes for each emotion
print("Searching for top 5 attributes...")
x_angry_5 = get_k_best(x_angry, y_angry, 5)
x_disgust_5 = get_k_best(x_disgust, y_disgust, 5)
x_fear_5 = get_k_best(x_fear, y_fear, 5)
x_happy_5 = get_k_best(x_happy, y_happy, 5)
x_sad_5 = get_k_best(x_sad, y_sad, 5)
x_surprise_5 = get_k_best(x_surprise, y_surprise, 5)
x_neutral_5 = get_k_best(x_neutral, y_neutral, 5)

# Search for top TWO attributes for each emotion
print("Searching for top 2 attributes...")
x_angry_2 = get_k_best(x_angry, y_angry, 2)
x_disgust_2 = get_k_best(x_disgust, y_disgust, 2)
x_fear_2 = get_k_best(x_fear, y_fear, 2)
x_happy_2 = get_k_best(x_happy, y_happy, 2)
x_sad_2 = get_k_best(x_sad, y_sad, 2)
x_surprise_2 = get_k_best(x_surprise, y_surprise, 2)
x_neutral_2 = get_k_best(x_neutral, y_neutral, 2)


"""
DATASET REDUCTION
"""
# 14, 35, and 70 non-class attribute reduction:
top_attrs = {2 : np.array([x_angry_2, x_disgust_2, x_fear_2, x_happy_2, x_sad_2, x_surprise_2, x_neutral_2]).flatten(),
        	5 : np.array([x_angry_5, x_disgust_5, x_fear_5, x_happy_5, x_sad_5, x_surprise_5, x_neutral_5]).flatten(),
        	10 : np.array([x_angry_10, x_disgust_10, x_fear_10, x_happy_10, x_sad_10, x_surprise_10, x_neutral_10]).flatten()}

print("Reducing datasets...")
x_train_70 = reduce_attr(x_train, top_attrs[10])
x_train_35 = reduce_attr(x_train, top_attrs[5])
x_train_14 = reduce_attr(x_train, top_attrs[2])
x_validation_70 = reduce_attr(x_validation, top_attrs[10])
x_validation_35 = reduce_attr(x_validation, top_attrs[5])
x_validation_14 = reduce_attr(x_validation, top_attrs[2])


"""
ATTEMPT TO IMPROVE CLASSIFICATION WITH NEW DATASETS
"""
print("70 attribute classification")
preds = get_knn_preds(knc, x_train_70, y_train, x_validation_70)
overall_accuracy(preds, y_validation)
accuracy_plots(preds, y_validation)

print("35 attribute classification")
preds = get_knn_preds(knc, x_train_35, y_train, x_validation_35)
overall_accuracy(preds, y_validation)
accuracy_plots(preds, y_validation)

print("14 attribute classification")
preds = get_knn_preds(knc, x_train_14, y_train, x_validation_14)
overall_accuracy(preds, y_validation)
accuracy_plots(preds, y_validation)