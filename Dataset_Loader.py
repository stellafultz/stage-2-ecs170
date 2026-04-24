'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')

        X_train, y_train = [], []
        X_test, y_test = [], []

        # ----- load training data -----
        f = open(self.dataset_source_folder_path + 'train.csv', 'r')
        for line in f:
            line = line.strip()
            elements = [int(x) for x in line.split(',')]
            y_train.append(elements[0])
            X_train.append(elements[1:])
        f.close()

        # ----- load testing data -----
        f = open(self.dataset_source_folder_path + 'test.csv', 'r')
        for line in f:
            line = line.strip()
            elements = [int(x) for x in line.split(',')]
            y_test.append(elements[0])
            X_test.append(elements[1:])
        f.close()

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }