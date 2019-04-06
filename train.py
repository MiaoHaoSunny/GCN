from __future__ import division
from __future__ import print_function

import time
# import argparse
import numpy as np
import math

import torch
import torch.nn.functional as F
import torch.optim as optim

from Advanced.GCNN.utils import load_data, accuracy
from Advanced.GCNN.models import GCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
epochs = 200
fastmode = False
lr = 0.01
weight_decay = 5e-4
hidden = 16
dropout = 0.5

np.random.seed(seed)
torch.manual_seed(seed)
if device == 'cuda':
    torch.cuda.manual_seed(seed)


adj, features, labels, idx_train, idx_val, idx_test = load_data()

model = GCN(nfeat=features.shape[1],
            nhid=hidden,
            nclass=labels.max().item()+1,
            dropout=dropout)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
model = model.to(device)
features = features.to(device)
adj, labels, idx_train, idx_val, idx_test = adj.to(device), labels.to(device), idx_train.to(device),\
                                            idx_val.to(device),\
                                            idx_test.to(device)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    if not fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t)
          )
    # curr_lr = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    # # curr_lr = lr / 3
    # update_lr(optimizer, curr_lr)


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:", "loss={:.4f}".format(loss_test.item()),
          "accuracy={:.4f}".format(acc_test.item()))


t_total = time.time()
for epoch in range(epochs):
    train(epoch)
print("Optimization Finished")
print("Total time elapsed: {:.4f}s".format(time.time()-t_total))
test()
