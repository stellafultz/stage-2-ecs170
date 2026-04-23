from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

np.random.seed(2)
torch.manual_seed(2)

# --- Load Data ---
train_loader = Dataset_Loader(dName='MNIST_train', dDescription='')
train_loader.dataset_source_folder_path = 'data/stage_2_data/'
train_loader.dataset_source_file_name = 'train.csv'
X_train, y_train = train_loader.load()

test_loader = Dataset_Loader(dName='MNIST_test', dDescription='')
test_loader.dataset_source_folder_path = 'data/stage_2_data/'
test_loader.dataset_source_file_name = 'test.csv'
X_test, y_test = test_loader.load()

print(f'Train: {X_train.shape}, Test: {X_test.shape}')

# --- Train ---
method_obj = Method_MLP(mName='MLP', mDescription='')
method_obj.train(X_train, y_train)

# --- Predict ---
y_pred = method_obj.test(X_test)

# --- Evaluate ---
print('\n===== Results =====')
print('Accuracy:            ', accuracy_score(y_test, y_pred))
print('F1 (weighted):       ', f1_score(y_test, y_pred, average='weighted'))
print('Recall (weighted):   ', recall_score(y_test, y_pred, average='weighted'))
print('Precision (weighted):', precision_score(y_test, y_pred, average='weighted'))

# --- Convergence Curve ---
plt.figure()
plt.plot(range(1, method_obj.max_epoch+1), method_obj.loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLP Training Convergence Curve')
plt.grid(True)
plt.savefig('../../result/stage_2_result/MLP_convergence_curve.png')
plt.show()

