#! /usr/bin/env python
from imrecog_func import get_data, extract_feature, train, predict, evaluation
def mnist_classify():
    data = get_data()
    train_feature = extract_feature(data['train'])
    model = train(train_feature, data['train'])
    test_feature = extract_feature(data['test'])
    predict_values = predict(test_feature, model)
    confusion_matrix, accuracy = evaluation(predict_values, data['test'])
    #print np.round(confusion_matrix, 2)
    print confusion_matrix
    print 'accuracy: ', accuracy 


if __name__ == '__main__':
    mnist_classify()
