import torch
import torch.nn as nn
import numpy as np
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.base_class.method import method


class Method_MLP(method, nn.Module):

    def __init__(self, mName=None, mDescription=None):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # -----------------------
        # Hyperparameters
        # -----------------------
        self.learning_rate = 1e-3
        self.max_epoch = 20
        self.batch_size = 64

        self.loss_list = []
        self.acc_list = []

        # -----------------------
        # Model architecture
        # -----------------------
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        # optimizer + loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    # -----------------------
    # forward pass
    # -----------------------
    def forward(self, x):
        return self.model(x)

    # -----------------------
    # training loop (mini-batch SGD)
    # -----------------------
    def train(self, X, y):

        X_t = torch.FloatTensor(np.array(X))
        y_t = torch.LongTensor(np.array(y))

        self.model.train()

        accuracy_evaluator = Evaluate_Accuracy('train_acc', '')

        for epoch in range(self.max_epoch):

            # shuffle indices
            perm = torch.randperm(X_t.size(0))

            epoch_loss = 0
            batches = 0

            for i in range(0, X_t.size(0), self.batch_size):

                idx = perm[i:i + self.batch_size]
                X_b, y_b = X_t[idx], y_t[idx]

                self.optimizer.zero_grad()

                outputs = self.model(X_b)
                loss = self.loss_fn(outputs, y_b)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            avg_loss = epoch_loss / batches
            self.loss_list.append(avg_loss)

            # training accuracy (full batch for monitoring only)
            with torch.no_grad():
                train_pred = self.model(X_t).max(1)[1]

            accuracy_evaluator.data = {
                'true_y': y_t,
                'pred_y': train_pred
            }

            acc = accuracy_evaluator.evaluate()
            self.acc_list.append(acc)

            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}/{self.max_epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    # -----------------------
    # testing
    # -----------------------
    def test(self, X):

        self.model.eval()

        with torch.no_grad():
            X_t = torch.FloatTensor(np.array(X))
            outputs = self.model(X_t)
            _, pred = torch.max(outputs, 1)

        return pred.numpy()

    # -----------------------
    # pipeline entry
    # -----------------------
    def run(self):

        print("method running...")
        print("--start training...")
        self.train(self.data['train']['X'], self.data['train']['y'])

        print("--start testing...")
        pred_y = self.test(self.data['test']['X'])

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
            'loss_list': self.loss_list,
            'acc_list': self.acc_list
        }