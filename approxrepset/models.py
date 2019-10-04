import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class ApproxRepSet(torch.nn.Module):

    def __init__(self, n_hidden_sets, n_elements, d, n_classes, device):
        super(ApproxRepSet, self).__init__()
        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements
        
        self.Wc = Parameter(torch.FloatTensor(d, n_hidden_sets*n_elements))
        self.fc1 = nn.Linear(n_hidden_sets, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.Wc.data.uniform_(-1, 1)

    def forward(self, X):
        t = self.relu(torch.matmul(X, self.Wc))
        t = t.view(t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets)
        t,_ = torch.max(t, dim=2)
        t = torch.sum(t, dim=1)
        t = self.relu(self.fc1(t))
        out = self.fc2(t)

        return F.log_softmax(out, dim=1)
