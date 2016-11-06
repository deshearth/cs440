import numpy as np
import matplotlib.pyplot as pl
WHITE = ' '
GRAY = '+'
BLACK = '#'
IM_SIZE = 28
def train(train_data):
    """
    train to get the feature, the feature would be a dictionary,
    the key would be the three possible chars in the image.
    the dimension of the value for each key is 28x28x10, 10 
    represents the class
    """
    #initialization
    likelihood = {WHITE: None, GRAY: None, BLACK: None}
    feature = {pixel_value: est_likelihood(pixel_value, train_data)
               for pixel_value in [WHITE, GRAY, BLACK]} 
    
def est_likelihood(pixel_value, train_data):
    """
    numerator represents the # of times pixel has specific value
    from training example from this class
    denominator represents the # of training example from this class
    """
    for class_ in range(10):
        numerator = (train_data.reshape(-1, IM_SIZE, IM_SIZE)
                     == pixel_value).sum(axis=0)
        denominator = (train_data[label] == class_).sum() 
        likelihood[pixel_value][:, :, class_] = numerator / denominator
    return likelihood

def test():
    pass
def evaluation():
    pass 
