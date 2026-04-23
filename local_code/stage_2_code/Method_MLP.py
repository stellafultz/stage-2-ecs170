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
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, X, y):
        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)
        self.model.train()

        for epoch in range(self.max_epoch):
            perm = torch.randperm(X_t.size(0))
            epoch_loss = 0
            batches = 0
            for i in range(0, X_t.size(0), self.batch_size):
                idx = perm[i:i+self.batch_size]
                X_b, y_b = X_t[idx], y_t[idx]
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(X_b), y_b)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                batches += 1
            avg = epoch_loss / batches
            self.loss_history.append(avg)
            print(f'Epoch {epoch+1}/{self.max_epoch} | Loss: {avg:.4f}')

    def test(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X))
            _, pred = torch.max(out, 1)
        return pred.numpy()