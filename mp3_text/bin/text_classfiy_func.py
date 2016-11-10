"""
the two models are Multinomial NB and Bernoulli NB
in MNB the feature is the frequency of that term occurring in the
class while BNB cares the frequency of the doc from the class that 
contains the term.
MNB:
P(w_i|class) = (# occurrance of w_i from class + k) 
                / (# total words from that class + k*(# unique words))
BNB:
P(w_i|class) = (# docs that contains w_i from class + k)
                / (# total docs from class + k*2) (2 is the # of class)
"""
import os
import pandas as pd
import numpy as np
LIFE_PART, MIN_WAGE = -1, 1
FISHER_CLASSES = [LIFE_PART, MIN_WAGE]
NEGATIVE, POSTIVE = -1, 1
MOVIE_CLASSES = [NEGATIVE, POSTIVE]
def get_data():
    #set path
    bin_path = os.getcwd()
    mp_path = os.path.dirname(bin_path)
    data_path = mp_path + '/database/' 
    movie_dir = data_path + '/movie_review/' 
    fisher_dir = data_path + '/fisher_2topic/' 
    movie_train_file = movie_dir + 'rt-train.txt'
    fisher_train_file = fisher_dir + 'fisher_train_2topic.txt'
    movie_test_file = movie_dir + 'rt-test.txt'
    fisher_test_file = fisher_dir + 'fisher_test_2topic.txt'
    #get data
    fisher = {'train': formatted_data(fisher_train_file), 
              'test': formatted_data(fisher_test_file)}
    movie = {'train': formatted_data(movie_train_file), 
              'test': formatted_data(movie_test_file)}
    return fisher, movie 

def formatted_data(fName):
    f = open(fName, 'r')
    data = np.array([eachLine.strip().split(' ', 1) for eachLine in f])
    f.close()
    return data

def unpack_stats(s, i):
    stat = np.array([stat.split(':')+[str(i)] for stat in s.split(' ')])
    cols = ['count', 'doc']
    return pd.DataFrame(stat[:, 1:], columns=cols, index=stat[:, 0]) 


def train(train_data, option):
     
    #table is a dictionary, keys are class name, value is a pandas frame
    #row is the word (not unique), 
    #cols are word_count and doc
    dfs = get_df(train_data) 
    model = MNB(dfs) if option is 'MNB' else BNB(dfs)
        
def get_df(train_data):
    "try using multi-index"
    dfs = {}
    for class_ in CLASSES:
        idx = np.where(train_data[:, 0] == class_)[0]
        dfs [class_] = []
        for i in idx:
            dfs[class_].append(unpack_stats(train_data[i][1], i))
        dfs[class_] = pd.concat(dfs[class_])
    return dfs 

            
def MNB(dfs):
    cols = ['neg_class', 'pos_class']
    model = pd.DataFrame(columns=cols)

    


        
        

