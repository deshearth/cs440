import os
import numpy as np
def get_data():
    #set path
    bin_path = os.getcwd()
    mp_path = os.path.dirname(bin_path)
    data_path = mp_path + '/database/' 
    train_images_file = data_path + 'trainingimages' 
    train_labels_file = data_path + 'traininglabels'
    test_images_file = data_path + 'testimages'
    test_labels_file = data_path + 'testlabels'
    #get data
    f = open(train_images_file, 'r')
    train_images = np.array([list(eachline.strip('\n').split() 
                            for eachline in f)])
    f.close()
    f = open(train_labels_file, 'r')
    train_labels = np.genfromtxt(train_labels_file, dtype=int)
    f.close()
    f = open(test_images_file, 'r')
    test_images = np.array([list(eachline.strip('\n').split() 
                            for eachline in f)])
    f.close()
    f = open(test_labels_file, 'r')
    test_labels = np.genfromtxt(test_labels_file, dtype=int)
    f.close()
    data = {'train': {'image': train_images, 'label': train_labels},
            'test': {'image': test_images, 'label': test_labels}}
    return data 
