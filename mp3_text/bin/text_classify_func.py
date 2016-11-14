# %load text_classfiy_func.py
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
import sys
COLS = ['count', 'doc']
LIFE_PART, MIN_WAGE = -1, 1
FISHER_CLASSES = [LIFE_PART, MIN_WAGE]
NEGATIVE, POSTIVE = -1, 1
MOVIE_CLASSES = [NEGATIVE, POSTIVE]
CLASSES = [-1, 1]
def get_path():
    bin_path = os.getcwd()
    mp_path = os.path.dirname(bin_path)
    files = {}
    files['fisher'] = {}
    files['movie'] = {}
    fisher = os.path.join(mp_path, 'database', 'fisher_2topic')
    movie = os.path.join(mp_path, 'database', 'movie_review')
    files['fisher']['train'] = os.path.join(fisher, 'fisher_train_2topic.txt')
    files['fisher']['test'] = os.path.join(fisher, 'fisher_test_2topic.txt')
    files['movie']['train'] = os.path.join(movie, 'rt-train.txt')
    files['movie']['test'] = os.path.join(movie, 'rt-test.txt')
    return files

def get_data():
    #get path
    files = get_path()
    #get data
    fisher = {'train': formatted_data(files['fisher']['train']), 
              'test': formatted_data(files['fisher']['test'])}
    movie = {'train': formatted_data(files['movie']['train']), 
              'test': formatted_data(files['movie']['test'])}
    return fisher, movie 

def formatted_data(fName):
    f = open(fName, 'r')
    data = np.array([eachLine.strip().split(' ', 1) for eachLine in f])
    f.close()
    return data

def unpack_stats(s, i, class_):
    stat = np.array([stat.split(':')+[str(i)+','] for stat in s.split(' ')])
    cols = ['count', 'doc']
    stat = pd.DataFrame(stat[:, 1:], columns=cols , index=stat[:, 0])
    stat['count'] = stat['count'].apply(pd.to_numeric)
    return stat


def train(train_data, option):
     
    #table is a dictionary, keys are class name, value is a pandas frame
    #row is the word (not unique), 
    #cols are word_count and doc
    df = get_df(train_data)
    return mnb(df, train_data) if option is 'mnb' else bnb(df, train_data)
    
        
def get_df(train_data):
    "try using multi-index"
    df = pd.DataFrame()
    df_tmp = []
    for class_ in CLASSES:
        idx = np.where(train_data[:, 0] == str(class_))[0]
        tmp = pd.DataFrame()
        for i in idx:
            tmp = tmp.append(unpack_stats(train_data[i, 1], i, class_))
        df_tmp.append(combine_dup(tmp))
    df = pd.concat(df_tmp, axis=1, keys=CLASSES)
    return df

def combine_dup(df):
    return pd.concat([df.groupby(df.index)['count'].sum(),
                      df.groupby(df.index)['doc'].sum()],
                      axis=1)

def mnb(df, raw_data):
    model = {}
    model['likelihoods'] = pd.DataFrame(index=df.index.values, 
                                        columns=CLASSES)
    model['priors'] = np.zeros(2)
    for i in xrange(2):
        class_ = CLASSES[i]
        count = np.nan_to_num(df.loc[:, (class_, 'count')].values)
        print count
        model['likelihoods'][class_] = smooth(count)
        model['priors'][i] = (raw_data[:, 0].astype('int') 
                           == class_).sum() \
                          / float(raw_data.shape[0])
    return model

def smooth(count, *args):
    k = 1
    c = (count+k) / (count+k).sum() if not args \
            else ((count+k) / float((args[0]+2*k)))
    return c

def bnb(df, raw_data):
    model = {}
    model['likelihoods'] = pd.DataFrame(index=df.index.values, 
                                        columns=CLASSES)
    model['priors'] = np.zeros(2)
    for i in xrange(2):
        class_ = CLASSES[i]
        n_doc = (raw_data[:, 0].astype('int') == class_).sum()
        n_doc_ww = df[(class_, 'doc')].apply(count_doc).values
        model['likelihoods'][class_] = smooth(n_doc_ww, n_doc)
        model['priors'][i] = (raw_data[:, 0].astype('int') 
                           == class_).sum() \
                          / float(raw_data.shape[0])
    return model

def count_doc(obj):
    obj = str(obj)
    if obj is 'nan':
        return 0
    else:
        s = obj.split(',')
        return len(set(filter(None, s)))
        
def predict(test_data, model, option):
    n_doc = test_data.shape[0]
    pred_val = np.zeros(n_doc)
    for i in xrange(n_doc):
        doc = [pair.split(':') for pair in test_data[i, 1].split(' ')]
        pred_val[i] = map_decision(doc, model, option)
    return pred_val

def map_decision(doc, model, *args):
    posteri = sum(map(lambda pair: cal_post(pair, model, *args), doc))
   
    return CLASSES[(posteri*model['priors']).argmax()]
    
def cal_post(pair, model, option):
    word = pair[0]
    if not any(model['likelihoods'].index.values==word):
        return np.zeros(2)
    else:
        return np.log(model['likelihoods'].loc[word, :].values) \
                   * int(pair[1]) if option is 'mnb' \
               else np.log(model['likelihoods'].loc[word, :].values)
    
    
def evaluation(pred_vals, ground_truth_data):
    ground_truth_label = ground_truth_data[:, 0].astype('int')
    conf_mat = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            class_idx = ground_truth_label == CLASSES[i]
            conf_mat[i, j] = \
                    (pred_vals[class_idx] == CLASSES[j]).sum() \
                    / float(class_idx.sum())
    return np.round(conf_mat, 2), \
        (pred_vals == ground_truth_label).sum() \
        / float(pred_vals.shape[0])

def write_data(fname, data, **kwargs):
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if type(data) is pd.core.frame.DataFrame:
        data.to_pickle(fname)
    if type(data) is np.ndarray:
        if kwargs:
            np.savetxt(fname, data, fmt=kwargs['fmt'])
        else:
            np.savetxt(fname, data)

