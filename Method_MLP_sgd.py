'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # Input: 784 (28x28 flattened), Hidden: 128, Output: 10 classes
        self.fc_layer_1 = nn.Linear(784, 128)
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(128, 10)
        self.activation_func_2 = nn.Softmax(dim=1)

    def forward(self, x):
        '''Forward propagation'''
        h = self.activation_func_1(self.fc_layer_1(x))
        y_pred = self.activation_func_2(self.fc_layer_2(h))
        return y_pred

    def train(self, X, y):

        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # ✅ initialize ONCE (outside loop)
        self.loss_list = []
        self.acc_list = []

        for epoch in range(self.max_epoch):

            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            y_true = torch.LongTensor(np.array(y))

            train_loss = loss_function(y_pred, y_true)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                accuracy_evaluator.data = {
                    'true_y': y_true,
                    'pred_y': y_pred.max(1)[1]
                }

                acc = accuracy_evaluator.evaluate()

                print("Epoch:", epoch,
                      "Accuracy:", acc,
                      "Loss:", train_loss.item())

                self.loss_list.append(train_loss.item())
                self.acc_list.append(acc)

    def test(self, X):
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
            'loss_list': self.loss_list,
            'acc_list': self.acc_list
        }