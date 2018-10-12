import time

import pandas
import sklearn
import numpy


def import_csv(path):
    """import a csv file and shuffle it

    Arguments:
      path {string} -- path to the csv file

    Returns:
      data_frame -- pandas.dataframe
    """
    print('importing', path)
    start_time = time.time()

    data_frame = pandas.read_csv(path)
    data_frame = sklearn.utils.shuffle(data_frame)
    data = data_frame.as_matrix()

    #replacing a long string to a numpy array
    for line in range(data_frame.shape[0]):
        pixels = data[line, 1].split(" ")
        
        pixels = numpy.array(pixels, dtype=numpy.uint8)
        data[line, 1] = pixels

    print("\nimportation done in:", time.time() - start_time)
    return data



def split_dataset(number_subset, dataset):
    """take a dataframe and split it in x number of subset for cross validation
    
    Arguments:
        number_subset {int} -- number of subsets
        dataset {numpy.array} -- [description]
    
    Returns:
        list of subsets
        the list of sets is segmented as following:
        [(x_train, y_train, x_val, y_val), ...]
    """
    subset_list =  numpy.array_split(dataset, number_subset)

    #for position in range(number_subset):
    
    #element = [subset for position, subset in subset_list if position!=3]
    

    set_list = []
    for set_number in range(number_subset):
        x_train = []
        y_train = []
        x_validation = []
        y_validation = []
        for position in range(number_subset):
            if(position == set_number):
                x_validation.append(subset_list[position][:, 1].tolist())
                y_validation.append(subset_list[position][:, 0].tolist())

            if(position != set_number):
                x_train.append(subset_list[position][:, 1].tolist())
                y_train.append(subset_list[position][:, 0].tolist())
        
        set_list.append( 
            (numpy.concatenate(x_train), numpy.concatenate(y_train), numpy.concatenate(x_validation), numpy.concatenate(y_validation))
            )
    return set_list

def get_data(path, number_subset=10, sep=True):
    """
    gets the data from the csv file
    if sep = False the data won't be split into training and testing data
    """
    data_set = import_csv(path)
    data_list = split_dataset(number_subset, data_set)
    for position in range(number_subset):
        (x_train, y_train, x_validation, y_validation) = data_list[position]
    if (sep):
        return (x_train, y_train, x_validation, y_validation)
    return (numpy.concatenate((x_train, x_validation)), numpy.concatenate((y_train, y_validation)))