import numpy as np
import matplotlib.pyplot as pl
import os
WHITE = ' '
GRAY = '+'
BLACK = '#'
FG = 1
BG = 0
IM_SIZE = 28
NUM_CLASSES = 10

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
    train_images = np.array([list(eachline.strip('\n')) 
                            for eachline in f])
    f.close()
    f = open(train_labels_file, 'r')
    train_labels = np.genfromtxt(train_labels_file, dtype=int)
    f.close()
    f = open(test_images_file, 'r')
    test_images = np.array([list(eachline.strip('\n')) 
                            for eachline in f])
    f.close()
    f = open(test_labels_file, 'r')
    test_labels = np.genfromtxt(test_labels_file, dtype=int)
    f.close()
    data = {'train': {'images': train_images, 'labels': train_labels},
            'test': {'images': test_images, 'labels': test_labels}}
    return data 

def extract_feature(data):
    feature = np.zeros(data['images'].shape)
    fg_idx = np.logical_or(data['images']==GRAY, data['images']==BLACK)
    bg_idx = data['images'] == WHITE
    feature[fg_idx] = 1
    feature[bg_idx] = 0
    return feature.reshape((-1, IM_SIZE, IM_SIZE))

def train(feature, train_data):
    """
    train to get the feature, the feature would be a dictionary,
    the key would be the three possible chars in the image.
    the dimension of the value for each key is 28x28x10, 10 
    represents the class
    """
    #initialization
    model = {}
    model['likelihoods'] = {pixel_value: est_likelihood(pixel_value, feature,
                            train_data) for pixel_value in [FG, BG]} 
    num_train_samples = train_data['images'].shape[0] / IM_SIZE 
    model['priors'] = np.array([(train_data['labels'] == class_).sum()
            for class_ in range(NUM_CLASSES)]) / float(num_train_samples)

    return model

def est_likelihood(pixel_value, feature, train_data):
    """
    numerator represents the # of times pixel has specific value
    from training example from this class
    denominator represents the # of training example from this class
    """
    likelihood = np.zeros((NUM_CLASSES, IM_SIZE, IM_SIZE))
    for class_ in range(NUM_CLASSES):
        class_idx = train_data['labels'] == class_
        numerator = (feature[class_idx, :, :] == pixel_value).sum(axis=0) 
        denominator = class_idx.sum() 
        likelihood[class_, :, :] = smooth(numerator, denominator)
    return likelihood

def smooth(numerator, denominator):
    k, V = 1, 2
    return (numerator + k) / (float(denominator) + k*V)


def predict(feature, model):
    num_test_samples = feature.shape[0]
    posteriors = np.zeros((num_test_samples,NUM_CLASSES))
    for i in range(num_test_samples):
        posteriors[i] = np.array([(np.log(model['likelihoods'] \
                [pixel_val][:, feature[i]==pixel_val])).sum(axis=1)
                for pixel_val in model['likelihoods'].keys()]).sum(axis=0)  
    return np.argmax((posteriors + model['priors']), axis=1)

def evaluation(predict_values, ground_truth_data):
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for true_val in range(NUM_CLASSES):
        for pred_val in range(NUM_CLASSES):
            class_idx = ground_truth_data['labels'] == true_val
            conf_mat[true_val][pred_val] = \
                    (predict_values[class_idx] == pred_val).sum() \
                     / float(class_idx.sum())
    return np.round(conf_mat, 2), \
           (predict_values == ground_truth_data['labels']).sum() \
            / float(predict_values.shape[0])
