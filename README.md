Interpretable machine learning model combining deep learning and rule learning. Developed by M.J. van der Zwart as MSc thesis project (c) 2023

## Installation

```
pip install r2ntab
```

## Preparing data using sample dataset

```python
import torch

from r2ntab import transform_dataset, kfold_dataset

name = 'adult.data'
X, Y, X_headers, Y_headers = transform_dataset(name, method='onehot-compare',
negations=False, labels='binary')
datasets = kfold_dataset(X, Y, shuffle=1)
X_train, X_test, Y_train, Y_test = datasets[0]
train_set = torch.utils.data.TensorDataset(torch.Tensor(X_train.to_numpy()),
torch.Tensor(Y_train))
```

## Creating and training the model

```python
from r2ntab import R2NTab

model = R2NTab(len(X_headers), 10, 1)
model.fit(train_set, epochs=1000)
Y_pred = model.predict(X_test)
```

## Extracting the results

```python
rules = model.extract_rules(X_headers, print_rules=True)
print(f'AUC: {model.score(Y_pred, Y_test, metric="auc")}')
print(f'# Rules: {len(rules)}')
print(f'# Conditions: {sum(map(len, rules))}')
```

## Contact

For any questions or problems, please open an issue <a href="https://github.com/mrvanderzwart/R2N-Tab">here</a> on GitHub.
