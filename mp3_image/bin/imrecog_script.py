#! /usr/bin/env python
from imrecog_func import get_data, extract_feature, train, predict, evaluation
def mnist_classify():
    data = get_data()
    train_feature = extract_feature(data['train'])
    model = train(train_feature, data['train'])
    test_feature = extract_feature(data['test'])
    predict_values, h_post, l_post = predict(test_feature, model)
    confusion_matrix, accuracy = evaluation(predict_values, data['test'])
    #print np.round(confusion_matrix, 2)
    return data, model, predict_values, h_post, l_post, confusion_matrix, accuracy


if __name__ == '__main__':
    mnist_classify()
