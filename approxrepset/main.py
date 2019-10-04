import torch
import numpy as np
import torch.nn.functional as F
from math import ceil
from models import ApproxRepSet
from utils import load_data,accuracy,AverageMeter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
dataset = 'twitter'
epochs = 30 # number of iterations
lr = 1e-3 # learning rate
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
        max_card = max([X_train[idx[j]].shape[1] for j in range(i*batch_size,min((i+1)*batch_size, n_train))])
        X = np.zeros((min((i+1)*batch_size, n_train)-i*batch_size, max_card, d))
        for j in range(i*batch_size,min((i+1)*batch_size, n_train)):
            X[j-i*batch_size,:X_train[idx[j]].shape[1],:] = X_train[idx[j]].T
        X = torch.FloatTensor(X).to(device)
        y = torch.LongTensor(np.where(y_train[idx[i*batch_size:min((i+1)*batch_size, n_train)]])[1]).to(device)
        train_batches.append((X, y))

    n_test_batches = ceil(n_test/batch_size)
    test_batches = list()
    for i in range(n_test_batches):
        max_card = max([X_test[j].shape[1] for j in range(i*batch_size,min((i+1)*batch_size, n_test))])
        X = np.zeros((min((i+1)*batch_size, n_test)-i*batch_size, max_card, d))
        for j in range(i*batch_size,min((i+1)*batch_size, n_test)):
            X[j-i*batch_size,:X_test[j].shape[1],:] = X_test[j].T
        X = torch.FloatTensor(X).to(device)
        y = torch.LongTensor(np.where(y_test[i*batch_size:min((i+1)*batch_size, n_test)])[1]).to(device)
        test_batches.append((X, y))

    model = ApproxRepSet(n_hidden_sets, n_elements, d, n_classes, device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    def train(X, y):
        optimizer.zero_grad()
        output = model(X)
        loss_train = F.cross_entropy(output, y)
        loss_train.backward()
        optimizer.step()
        return output, loss_train

    def test(X, y):
        output = model(X)
        loss_test = F.cross_entropy(output, y)
        return output, loss_test

    model.train()
    for epoch in range(epochs):

        train_loss = AverageMeter()
        train_err = AverageMeter()

        for X, y in train_batches:
            output, loss = train(X, y)

            train_loss.update(loss.item(), output.size(0))
            train_err.update(1-accuracy(output.data, y.data), output.size(0))

        print("Cross-val iter:", '%02d' % (it+1), "epoch:", '%03d' % (epoch+1), "train_loss=", "{:.5f}".format(train_loss.avg),
            "train_err=", "{:.5f}".format(train_err.avg))

    model.eval()

    test_loss = AverageMeter()
    test_err = AverageMeter()

    for X, y in test_batches:
        output, loss = test(X, y)
        
        test_loss.update(loss.item(), output.size(0))
        test_err.update(1-accuracy(output.data, y.data), output.size(0))

    print("Cross-val iter:", '%02d' % (it+1), "train_loss=", "{:.5f}".format(train_loss.avg),
        "train_err=", "{:.5f}".format(train_err.avg), "test_loss=", "{:.5f}".format(test_loss.avg), "test_err=", "{:.5f}".format(test_err.avg))
    print()

    errs.append(test_err.avg.cpu())

print("Average error:", "{:.5f}".format(np.mean(errs)))
print("Standard deviation:", "{:.5f}".format(np.std(errs)))