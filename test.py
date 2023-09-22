import torch

from R2Ntab import R2Ntab
from datasets.dataset import transform_dataset, kfold_dataset

# Read datasets
name = 'adult'
X, Y, X_headers, Y_headers = transform_dataset(name, method='onehot-compare', negations=False, labels='binary')
datasets = kfold_dataset(X, Y, shuffle=1)
X_train, X_test, Y_train, Y_test = datasets[0]
train_set = torch.utils.data.TensorDataset(torch.Tensor(X_train.to_numpy()), torch.Tensor(Y_train))
test_set = torch.utils.data.TensorDataset(torch.Tensor(X_test.to_numpy()), torch.Tensor(Y_test))

# Train R2N-tab
net = R2Ntab(len(X_headers), 50, 1)
net.fit(train_set, device='cuda', epochs=100, batch_size=400)
