import torch
import torch.nn as nn
import numpy as np

class Method_MLP:
    def __init__(self, mName=None, mDescription=None):
        self.method_name = mName
        self.method_description = mDescription
        self.learning_rate = 0.001
        self.max_epoch = 20
        self.batch_size = 64
        self.loss_history = []

        self.model = nn.Sequential(
            nn.Linear(784, 256), # hidden layer 1: 784 pixel inputs -> 256 neurons
            nn.ReLU(), # turns negative values to 0
            nn.Dropout(0.3), # randomly sets a fraction (30%) of neurons to 0 to prevent overfitting
            nn.Linear(256, 128), # hidden layer 2: 256 -> 128 neurons
            nn.ReLU(), # turns negative values to 0
            nn.Linear(128, 10) # output layer: 128 -> 10 neurons (one neuron per digit)
        )

        # Adam optimizer adjusts weights during training
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        # CrossEntropyLoss measures how wrong the predictions are
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, X, y):
        # convert numpy arrays to Pytorch tensors
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)

        # puts model in training mode
        self.model.train()

        for epoch in range(self.max_epoch):
            # shuffles the data every epoch
            perm = torch.randperm(X_t.size(0))
            epoch_loss = 0
            batches = 0

            # loop through the data in chunks of batch_size
            for i in range(0, X_t.size(0), self.batch_size):
                # grab the next batch using shuffled indices
                idx = perm[i:i+self.batch_size]
                X_b, y_b = X_t[idx], y_t[idx]

                # clear gradients from previous batch
                self.optimizer.zero_grad()

                # forward pass: run batch through model and compute loss
                loss = self.loss_fn(self.model(X_b), y_b)

                # backward pass: compute gradients
                loss.backward()

                # update weights based on gradients
                self.optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            # average loss across all batches for this epoch
            avg = epoch_loss / batches
            self.loss_history.append(avg)
            print(f'Epoch {epoch+1}/{self.max_epoch} | Loss: {avg:.4f}')

    def test(self, X):
        # puts model in eval mode
        self.model.eval()

        # no_grad means it doesn't track gradients
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X))

            # picks the class with the highest score as the prediction
            # _ discards the score values, pred keeps the class index (0-9)
            _, pred = torch.max(out, 1)
        return pred.numpy()