import filehandler
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


def get_knn_preds(knc, X, y, X_test):
    """
    Runs the KNN algorithm and returns predictions
    """
    print("Fitting data...")
    start_time = time.time()
    knc.fit(X, y)
    print(time.time() - start_time, "seconds to fit data")
    
    print("Making predictions...")
    start_time = time.time()
    predictions = knc.predict(x_test)
    print(time.time() - start_time, "seconds to predict the classes")
    
    return predictions


def overall_accuracy(dct, num_attr, predictions, actual):
    """
    Calculates overall accuracy and adds it to the dictionary
    """
    total = len(predictions)
    correct_count = (predictions == actual).sum()
    print(correct_count, " out of ", total, " correctly predicted")
    accuracy = (correct_count / total)
    print(accuracy * 100, "% correctly predicted")
    dct[num_attr] = accuracy
    return accuracy
    

def disp_confusion_matrix(num_attr, predictions, actual):
    """
    displays the confusion matrix
    """
    # confusion matrix
    confusion_mat = confusion_matrix(actual, predictions)
    
    df_cm = pd.DataFrame(confusion_mat, index = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
                  columns = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
    plt.figure(figsize = (8, 6))
    sn.heatmap(df_cm, annot=True, fmt="d", cmap='Greys')
    plt.title('Confusion Matrix for %s Attributes', num_attr)
    plt.show()
    plt.clf()
    return confusion_mat

    
def disp_accuracy_hist(num_attr, cm, predictions, actual):
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
    plt.title('Histogram of Prediction Accuracy for %s Attributes', num_attr)
    plt.xlabel('Emotions')
    plt.ylabel('Accuracy')
    plt.show()
    plt.clf()


def disp_acc_summary(dct):
    """
    displays a histogram that summarizes the prediction accuracies based on
    the number of attributes
    """
    bar_width = 1
    plt.bar(dct.keys(), dct.values(), bar_width, color='r')
    plt.title('Histogram of Prediction Accuracy by Number of Attributes')
    plt.xlabel('Number of Attributes')
    plt.ylabel('Accuracy')
    plt.show()
    plt.clf()
    

def accuracy_plots(num_attr, predictions, actual):
    """
    displays the confusion matrix and the accuracy histogram
    """
    disp_accuracy_hist(num_attr, disp_confusion_matrix(num_attr, predictions, actual), predictions, actual)


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
    saves the data at certain indices (which are in attr_indices)
    into the new dataset
    """
    new = []
    for i in range(len(data)):
        new.append([data[i][index] for index in attr_indices])
    return new


def reduce_attr_emo(new, data, attr_indices):
    """
    saves the data at certain indices (which are in attr_indices) into new
    """
    new.append([data[index] for index in attr_indices])


def reduce_data_emo(x, y, num):
    """
    transforms the given dataset to keep num amount of attributes
    """
    new_data = []
    for i in range(len(x)):
        if (y[i] == 0):
            reduce_attr_emo(new_data, x[i], top_attrs_emo[num][0]) # num best attributes for angry emotion
        if (y[i] == 1):
            reduce_attr_emo(new_data, x[i], top_attrs_emo[num][1]) # num best attributes for disgust emotion
        if (y[i] == 2):
            reduce_attr_emo(new_data, x[i], top_attrs_emo[num][2]) # num best attributes for fear emotion
        if (y[i] == 3):
            reduce_attr_emo(new_data, x[i], top_attrs_emo[num][3]) # num best attributes for happy emotion
        if (y[i] == 4):
            reduce_attr_emo(new_data, x[i], top_attrs_emo[num][4]) # num best attributes for sad emotion
        if (y[i] == 5):
            reduce_attr_emo(new_data, x[i], top_attrs_emo[num][5]) # num best attributes for surprise emotion
        if (y[i] == 6):
            reduce_attr_emo(new_data, x[i], top_attrs_emo[num][6]) # num best attributes for neutral emotion
    return new_data