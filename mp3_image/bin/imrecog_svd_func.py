import numpy as np
def train(train_data):
    #number of pixels
    n_p = 28 * 28
    model = {}
    model['im_mean'] = []
    model['usv'] = []
    for class_ in xrange(10):
        im = train_data['images'][train_data['labels']==class_, :, :]
        S = im.reshape(-1, n_p).T
        _, n_im = S.shape
        im_mean = (S.sum(axis=1) / n_im).reshape(-1, 1)
        model['im_mean'].append(im_mean)
        S_minus_mean = S - im_mean.repeat(n_im, axis=1)
        model['usv'].append(np.linalg.svd(S_minus_mean))
    return model

def predict(test_imgs, model):
    num_test_samples = test_imgs.shape[0]
    pred_vals = np.zeros(num_test_samples)
    for i in xrange(num_test_samples):
        pred_vals[i] = np.array(map(lambda param: msr(test_imgs[i], param),
                zip(model['im_mean'], model['usv']))).argmin()
    return pred_vals

def msr(test_im, param):
    im_mean = param[0]
    usv = param[1]
    return np.linalg.norm(np.dot(usv[0], (test_im.reshape(784, 1)-im_mean)), 2) ** 2


