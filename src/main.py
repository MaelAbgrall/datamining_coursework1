import utils.filehandler as filehandler
from sklearn.neighbors import NearestNeighbors

import time

#import the csv file
dataset = filehandler.import_csv('../fer2018/fer2018.csv')

#prepare 10 subsets for cross validation
set_list = filehandler.split_dataset(10, dataset)

#learn
for position in range(10):
   (x_train, y_train, x_validation, y_validation) = set_list[position]
   #call learn using x_train, x_validation, etc
   

start_time = time.time()

nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(x_train[0:1000])
_, indices = nbrs.kneighbors(x_train)

print(time.time() - start_time)