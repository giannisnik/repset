import numpy as np
from math import ceil
from sklearn.metrics import accuracy_score,log_loss
from models import RepSet
from utils import load_data,AverageMeter

# Parameters
dataset = 'twitter'
epochs = 10 # number of iterations
lr = 0.01 # learning rate
n_hidden_sets = 10 # number of hidden sets
n_elements = 20 # cardinality of each hidden set
d = 300 # dimension of each vector
batch_size = 64  # batch size
cv_folds = 5 # number of folds for cross-validation

errs = list()

for it in range(cv_folds):
    X_train, X_test, y_train, y_test, n_classes, _, _, _ , _ = load_data(dataset, it)

    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    idx = np.random.permutation(n_train)
    n_train_batches = ceil(n_train/batch_size)
    train_batches = list()
    for i in range(n_train_batches):
        train_batches.append((X_train[idx[i*batch_size:min((i+1)*batch_size, n_train)]], y_train[idx[i*batch_size:min((i+1)*batch_size, n_train)]]))

    n_test_batches = ceil(n_test/batch_size)
    test_batches = list()
    for i in range(n_test_batches):
        test_batches.append((X_test[i*batch_size:min((i+1)*batch_size, n_test)], y_test[i*batch_size:min((i+1)*batch_size, n_test)]))

    model = RepSet(lr, n_hidden_sets, n_elements, d, n_classes)
 
    for epoch in range(epochs):

        train_loss = AverageMeter()
        train_err = AverageMeter()

        for X, y in train_batches:
            y_pred = model.train(X, y)

            train_loss.update(log_loss(y, y_pred), y_train.size)
            train_err.update(1-accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1)), y.shape[0])

        print("Cross-val iter:", '%02d' % (it+1), "epoch:", '%03d' % (epoch+1), "train_loss=", "{:.5f}".format(train_loss.avg),
            "train_err=", "{:.5f}".format(train_err.avg))

    test_loss = AverageMeter()
    test_err = AverageMeter()

    for X, y in test_batches:
        y_pred = model.test(X)

        test_loss.update(log_loss(y, y_pred), y_test.size)
        test_err.update(1-accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1)), y.shape[0])

    print("Cross-val iter:", '%02d' % (it+1), "train_loss=", "{:.5f}".format(train_loss.avg),
        "train_err=", "{:.5f}".format(train_err.avg), "test_loss=", "{:.5f}".format(test_loss.avg), "test_err=", "{:.5f}".format(test_err.avg))
    print()

    errs.append(test_err.avg)

print("Average error:", "{:.5f}".format(np.mean(errs)))
print("Standard deviation:", "{:.5f}".format(np.std(errs)))
