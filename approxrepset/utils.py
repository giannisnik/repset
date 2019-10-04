import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.io import loadmat

def load_data(dataset, cv_fold):
    data = loadmat('../datasets/'+dataset+'_tr_te_split.mat')
    X = data['X']
    TR = data['TR'][cv_fold,:]-1
    TE = data['TE'][cv_fold,:]-1

    X_train = X[:,TR][0]
    X_test = X[:,TE][0]

    BOW_X = data['BOW_X']
    BOW_X_train = BOW_X[:,TR]
    BOW_X_test = BOW_X[:,TE]

    class_labels = np.array(data['Y'][0])
    le = LabelEncoder()
    class_labels = le.fit_transform(class_labels)
    n_classes = np.unique(class_labels).size
    y = np.zeros((class_labels.size, n_classes))
    for i in range(class_labels.size):
        y[i,class_labels[i]] = 1
    y_train = y[TR,:]
    y_test = y[TE,:]

    words = data['words']
    words_train = words[:,TR]
    words_test = words[:,TE]

    return X_train, X_test, y_train, y_test, n_classes, BOW_X_train, BOW_X_test, words_train, words_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count