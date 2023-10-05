import torch

from R2Ntab import R2Ntab
from datasets.dataset import transform_dataset, kfold_dataset
from sklearn.metrics import roc_auc_score

# Read datasets
name = 'chess'
X, Y, X_headers, Y_headers = transform_dataset(name, method='onehot-compare', negations=False, labels='binary')
datasets = kfold_dataset(X, Y, shuffle=1)
X_train, X_test, Y_train, Y_test = datasets[0]
train_set = torch.utils.data.TensorDataset(torch.Tensor(X_train.to_numpy()), torch.Tensor(Y_train))
test_set = torch.utils.data.TensorDataset(torch.Tensor(X_test.to_numpy()), torch.Tensor(Y_test))

# Train R2N-tab
net = R2Ntab(len(X_headers), 30, 1)
auc, rules, conds = net.fit(train_set, test_set, device='cpu', epochs=1000, batch_size=40, cancel_lam=1e-4, lr_cancel=1e-2, max_conditions=[50, 60, 110, 160, 210, 260, 300, 350])
print(auc)
print(rules)
print(conds)
