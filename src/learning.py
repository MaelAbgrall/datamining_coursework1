import time
import numpy as np

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
    predictions = knc.predict(X_test)
    print(time.time() - start_time, "seconds to predict the classes")
    
    return predictions


def overall_accuracy(dct, num_attr, predictions, actual):
    """
    Calculates overall accuracy and adds it to the dictionary
    """
    total = len(predictions)
    correct_count = (predictions == actual).sum()
    print("%d out of %d" % (correct_count, total))
    accuracy = (correct_count / total)
    print("%0.03f%% correctly predicted" % (accuracy * 100))
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
    title = 'Confusion Matrix for ' + num_attr + ' Attributes'
    plt.title(title)
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
    title = 'Histogram of Prediction Accuracy for ' + num_attr + ' Attributes'
    plt.title(title)
    plt.xlabel('Emotions')
    plt.ylabel('Accuracy')
    plt.show()
    plt.clf()


def accuracy_plots(num_attr, predictions, actual):
    """
    displays the confusion matrix and the accuracy histogram
    """
    disp_accuracy_hist(num_attr, disp_confusion_matrix(num_attr, predictions, actual), predictions, actual)



def edit_hist_bar(bar_list, color, label):
    """
    Changes the color and label of each bar in bar_list
    """
    # to avoid duplicate labels
    bar_list[0].set_label(label) 
    bar_list[0].set_facecolor(color)
    i = 1
    for i in range(len(bar_list)):
        bar_list[i].set_facecolor(color)


def disp_acc_summary(dct):
    """
    displays a histogram that summarizes the prediction accuracies based on
    the number of attributes
    ! this function is very specific to the experiments that have been carried out
    """
    plt.figure(figsize=(8,6))
    bars = plt.bar(list(dct.keys()), tuple(list(dct.values())))
    plt.title('Histogram of Prediction Accuracy by Number of Attributes')
    plt.xlabel('Number of Attributes')
    plt.ylabel('Accuracy')
    
    edit_hist_bar([bars[0]], 'r', 'all attributes')
    edit_hist_bar(bars[1:4], 'b', 'top overall attributes')
    edit_hist_bar(bars[4:7], 'g', 'top attributes depending on class')
    plt.legend(loc='upper left')
    plt.show()
    plt.clf()
    

def get_k_best(X, y, k):
    """
    returns the inidices of the k best attributes for each emotion type
    """
    kbest = SelectKBest(f_regression, k)
    kbest.fit(X, y)
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


def reduce_data_emo(top_attrs, x, y, num):
    """
    transforms the given dataset to keep num amount of attributes
    """
    new_data = []
    for i in range(len(x)):
        if (y[i] == 0):
            reduce_attr_emo(new_data, x[i], top_attrs[num][0]) # num best attributes for angry emotion
        if (y[i] == 1):
            reduce_attr_emo(new_data, x[i], top_attrs[num][1]) # num best attributes for disgust emotion
        if (y[i] == 2):
            reduce_attr_emo(new_data, x[i], top_attrs[num][2]) # num best attributes for fear emotion
        if (y[i] == 3):
            reduce_attr_emo(new_data, x[i], top_attrs[num][3]) # num best attributes for happy emotion
        if (y[i] == 4):
            reduce_attr_emo(new_data, x[i], top_attrs[num][4]) # num best attributes for sad emotion
        if (y[i] == 5):
            reduce_attr_emo(new_data, x[i], top_attrs[num][5]) # num best attributes for surprise emotion
        if (y[i] == 6):
            reduce_attr_emo(new_data, x[i], top_attrs[num][6]) # num best attributes for neutral emotion
    return new_data