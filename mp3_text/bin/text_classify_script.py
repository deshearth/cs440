from text_classify_func import get_data, train, predict, evaluation, write_data, get_df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def script():
    fisher, movie = get_data()
    option = {}
    #option['data'] = 'fisher'
    option['model'] = 'bnb'
    option['data'] = 'movie'
    #option['model'] = 'mnb'

    data = fisher if option['data'] is 'fisher' else movie
    model = train(data['train'], option['model'])
    pred_vals = predict(data['test'], model, option['model'])

    out_dir = os.path.join(os.path.dirname(os.getcwd()), 'output')
    model_fname = os.path.join(out_dir, option['data']+'_'+option['model'])
    pred_fname = model_fname + '_pred'

    write_data(model_fname+'_likelihoods', model['likelihoods'])
    write_data(model_fname+'_priors', model['priors'])
    write_data(pred_fname, pred_vals, fmt='%d')

if __name__ == '__main__':
    script()
