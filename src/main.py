import utils.filehandler as filehandler
from sklearn.neighbors import KNeighborsClassifier
import time

# for confusion matrix
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix

#import the csv file
dataset = filehandler.import_csv('../fer2018/fer2018.csv')


#prepare 10 subsets for cross validation
set_list = filehandler.split_dataset(10, dataset)

for position in range(10):
   (x_train, y_train, x_validation, y_validation) = set_list[position]



# Classification: Performance of the Nearest Neighbour Algorithm on the given data set

"""
Classifier initialization
"""
knc = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3)

print("Fitting data...")
start_time = time.time()
knc.fit(x_train, y_train)
print(time.time() - start_time, "seconds to fit data")

"""
Running the algorithm. preds will hold the class predictions
"""
print("Making predictions...")
start_time = time.time()
preds = knc.predict(x_validation)
print(time.time() - start_time, "seconds to predict the classes")


"""
Overall accuracy calculation
"""
correct_count = (preds == y_validation).sum()
print(correct_count)
accuracy = correct_count / len(preds)
print(accuracy)


# Classification Accuracy Calculation

"""
Computing confusion matrix to evaluate classification accuracy
"""
confusion_matrix = confusion_matrix(y_validation, preds)


"""
Plotting the confusion matrix
"""
df_cm = pd.DataFrame(confusion_matrix, index = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
                  columns = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'])
plt.figure(figsize = (8, 6))
sn.heatmap(df_cm, annot=True, fmt="d", cmap='Greys')

"""
Plotting the accuracy histogram
"""
acc = []
for i in range(len(confusion_matrix)):
    total_emote = confusion_matrix[i].sum()
    acc.append(confusion_matrix[i][i] / total_emote)
    
bar_width = .5
positions = np.arange(7)
plt.bar(positions, acc, bar_width)
plt.xticks(positions + bar_width / 2, ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'))
plt.show()

