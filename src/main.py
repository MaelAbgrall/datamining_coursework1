import utils.filehandler as filehandler

#import the csv file
dataset = filehandler.import_csv('../fer2018/fer2018.csv')

#prepare 10 subsets for cross validation
set_list = filehandler.split_dataset(10, dataset)

#learn
for position in range(10):
   (x_train, y_train, x_validation, y_validation) = set_list[position]
   
   #call learn using x_train, x_validation, etc

