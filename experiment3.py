import torch
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from datasets.dataset import transform_dataset, kfold_dataset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from R2Ntab import R2Ntab
from sklearn.metrics import roc_auc_score


def run_selector(feature_selector, train_set, X_train, Y_train, X_test, Y_test, batch_size, lr_cancel, cancel_lam):
    start = time.time()

    if feature_selector == 'r2ntab':
        net = R2Ntab(train_set[:][0].size(1), 50, 1)
        net.fit(train_set, batch_size=batch_size, epochs=1000, cancel_lam=cancel_lam, lr_cancel=lr_cancel, fs=True)
        cancelled_features = list(torch.where(net.cancelout_layer.weight < 0)[0].numpy())
    elif feature_selector == 'gb':
        gb = GradientBoostingClassifier(n_estimators=50, random_state=0)
        gb.fit(X_train, Y_train)
        cancelled_features = np.where(gb.feature_importances_ == 0)[0]
    elif feature_selector == 'pca':
        pca = PCA(n_components=20)
        pca.fit(X_train)
        component = np.mean(pca.components_, axis=0)
        cancelled_features = list(torch.where(torch.tensor(component) < 0)[0].numpy())
    elif feature_selector == 'regression':
        model = LinearRegression()
        model.fit(X_train, Y_train)
        cancelled_features = list(torch.where(torch.tensor(model.coef_) < 0)[0].numpy())

    X_train_filtered = X_train.copy(deep=True)
    X_test_filtered = X_test.copy(deep=True)
    for index, ft_index in enumerate(cancelled_features):
        X_train = X_train.drop(X_train.columns[ft_index-index], axis=1)
        X_test = X_test.drop(X_test.columns[ft_index-index], axis=1)

    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    AUC = roc_auc_score(rf.predict(X_test), Y_test)

    end = time.time()
    runtime = end-start

    return AUC, len(cancelled_features)/train_set[:][0].size(1), runtime


def run():
    folds = 5
    runs = 5
    feature_selectors = ['r2ntab', 'gb', 'pca', 'regression']
    cancel_lams = {'heloc' : 1e-2, 'house' : 1e-4, 'adult' : 1e-2, 'magic' : 1e-2, 'diabetes' : 1e-2, 'chess' : 1e-4, 'backnote' : 1e-2, 'tictactoe' : 1e-6}
    for name in ['adult', 'heloc', 'house', 'magic', 'chess', 'diabetes', 'tictactoe', 'backnote']:
        aucs = {fs: [] for fs in feature_selectors}
        sparsity = {fs: [] for fs in feature_selectors}
        runtimes = {fs: [] for fs in feature_selectors}

        X, Y, X_headers, Y_headers = transform_dataset(name, method='onehot-compare', negations=False, labels='binary')
        datasets = kfold_dataset(X, Y, shuffle=1)

        batch_size = 400 if len(X) > 10e3 else 40
        
        if name in ['chess', 'heloc']:
            lr_cancel = 1e-2
        else:
            lr_cancel = 5e-3

        print(f'dataset: {name}')
        for fold in range(folds):
            print(f'  fold: {fold+1}') 
            X_train, X_test, Y_train, Y_test = datasets[fold]
            train_set = torch.utils.data.TensorDataset(torch.Tensor(X_train.to_numpy()), torch.Tensor(Y_train))
            
            for fs in feature_selectors:
                auc_values, sparsity_values, runtime_values = 0, 0, 0
                for run in range(runs):
                    print(f'    run: {run+1}')
                    new_auc, new_sparsity, new_runtime = run_selector(fs, train_set, X_train, Y_train, X_test, Y_test, batch_size, lr_cancel, cancel_lams[name])

                    auc_values += new_auc
                    sparsity_values += new_sparsity
                    runtime_values += new_runtime

                aucs[fs].append(auc_values/runs)
                sparsity[fs].append(sparsity_values/5)
                runtimes[fs].append(runtime_values/runs)

        with open(f'exp3-auc-{name}.json', 'w') as file:
            json.dump(aucs, file)

        with open(f'exp3-sparsities-{name}.json', 'w') as file:
            json.dump(sparsity, file)

        with open(f'exp3-runtimes-{name}.json', 'w') as file:
            json.dump(runtimes, file)
            
            
def plot():
    cancel_lams = {'heloc' : 1e-2, 'house' : 1e-4, 'adult' : 1e-2, 'magic' : 1e-2, 'diabetes' : 1e-2, 'chess' : 1e-4, 'backnote' : 1e-2, 'tictactoe' : 1e-6}
    for name in ['adult', 'heloc', 'house', 'magic', 'tictactoe', 'backnote', 'chess', 'diabetes']:

        print('dataset:', name)

        with open(f'exp3-auc-{name}.json') as file:
            aucs = json.load(file)

        with open(f'exp3-sparsities-{name}.json') as file:
            sparsities = json.load(file)
            
        with open(f'exp3-runtimes-{name}.json') as file:
            runtimes = json.load(file)


        for fs in ['r2ntab', 'gb', 'pca', 'regression']:
            print(f'AUC {fs}:', sum(aucs[f'{fs}']) / len(aucs[f'{fs}']))
            print(f'sparsity {fs}:', sum(sparsities[f'{fs}']) / len(sparsities[f'{fs}']))
            print(f'runtime {fs}:', sum(runtimes[f'{fs}']) / len(runtimes[f'{fs}']))
            
        print('\n')


if __name__ == "__main__":
    run()
