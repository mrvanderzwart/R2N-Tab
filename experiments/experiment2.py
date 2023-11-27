import torch
import json
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '..')
sys.path.insert(0, '../src')
sys.path.insert(0, '../src/include')

from datasets.dataset import transform_dataset, kfold_dataset

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from R2Ntab import R2Ntab
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

import math


def transform(X_train, X_test, cancelled_features):
    for index, ft_index in enumerate(cancelled_features):
        X_train = X_train.drop(X_train.columns[ft_index-index], axis=1)
        X_test = X_test.drop(X_test.columns[ft_index-index], axis=1)
        
    return X_train, X_test


def run_selector(feature_selector, train_set, X_train, Y_train, X_test, Y_test, batch_size, lr_cancel, cancel_lam, epochs):
    start = time.time()

    if feature_selector == 'r2ntab':
        model = R2Ntab(train_set[:][0].size(1), 50, 1)
        model.fit(train_set, batch_size=batch_size, epochs=epochs, cancel_lam=cancel_lam, lr_cancel=lr_cancel)
        cancelled_features = list(torch.where(model.cancelout_layer.weight < 0)[0].numpy())
    elif feature_selector == 'gb':
        model = GradientBoostingClassifier()
        model.fit(X_train, Y_train)
        cancelled_features = list(np.where(model.feature_importances_ == 0)[0])
        X_train, X_test = transform(X_train, X_test, cancelled_features)
        model.fit(X_train, Y_train)
    elif feature_selector == 'lda':
        model = LinearDiscriminantAnalysis()
        model.fit_transform(X_train, Y_train)
        cancelled_features = list(torch.where(torch.tensor(model.coef_[0]) < 0)[0].numpy())
        X_train, X_test = transform(X_train, X_test, cancelled_features)
        model.fit(X_train, Y_train)
    elif 'rf' in feature_selector:
        n_features = train_set[:][0].size(1)
        if '1' in feature_selector:
            k = math.floor(n_features*0.25)
        elif '2' in feature_selector:
            k = math.floor(n_features*0.5)
        elif '3' in feature_selector:
            k = math.floor(n_features*0.75)
        model = RandomForestClassifier()
        model.fit(X_train, Y_train)
        weights = list(model.feature_importances_)
        indices = sorted(range(len(weights)), key=lambda i: weights[i])
        cancelled_features = sorted([i for i in indices[:k]])
        X_train, X_test = transform(X_train, X_test, cancelled_features)
        model.fit(X_train, Y_train)
    elif feature_selector == 'svm':
        model = SVC(kernel='linear')
        model.fit(X_train, Y_train)
        cancelled_features = list(torch.where(torch.tensor(model.coef_[0]) < 0)[0].numpy())
        X_train, X_test = transform(X_train, X_test, cancelled_features)
        model.fit(X_train, Y_train)

    auc = roc_auc_score(model.predict(X_test), Y_test)
    sparsity = len(cancelled_features)/train_set[:][0].size(1)
    
    end = time.time()
    runtime = end-start

    return auc, sparsity, runtime


def run(fold):
    runs = 5
    feature_selectors = ['r2ntab']
    cancel_lams = {'heloc' : 1e-2, 'house' : 1e-4, 'adult' : 1e-2, 'magic' : 1e-2, 'diabetes' : 1e-2, 'chess' : 1e-4, 'backnote' : 1e-2, 'tictactoe' : 1e-6}
    for name in ['tictactoe']:
        aucs = {fs: [] for fs in feature_selectors}
        sparsity = {fs: [] for fs in feature_selectors}
        runtimes = {fs: [] for fs in feature_selectors}

        X, Y, X_headers, Y_headers = transform_dataset(name, method='onehot-compare', negations=False, labels='binary')
        datasets = kfold_dataset(X, Y, shuffle=1)

        batch_size = 400 if len(X) > 10e3 else 40
        
        if name == 'tictactoe':
            epochs=6000
        else:
            epochs = 1000
        
        if name in ['chess', 'heloc']:
            lr_cancel = 1e-2
        else:
            lr_cancel = 5e-3

        print(f'dataset: {name}')
        
        X_train, X_test, Y_train, Y_test = datasets[fold-1]
        train_set = torch.utils.data.TensorDataset(torch.Tensor(X_train.to_numpy()), torch.Tensor(Y_train))
            
        for fs in feature_selectors:
            auc_values, sparsity_values, runtime_values = 0, 0, 0
            for run in range(runs):
                print(f'    run: {run+1}')
                new_auc, new_sparsity, new_runtime = run_selector(fs, train_set, X_train, Y_train, X_test, Y_test, batch_size, lr_cancel, cancel_lams[name], epochs)

                auc_values += new_auc
                sparsity_values += new_sparsity
                runtime_values += new_runtime

            aucs[fs].append(auc_values/runs)
            sparsity[fs].append(sparsity_values/5)
            runtimes[fs].append(runtime_values/runs)

        with open(f'exp3-auc-{name}-fold-{fold}.json', 'w') as file:
            json.dump(aucs, file)

        with open(f'exp3-sparsities-{name}-fold-{fold}.json', 'w') as file:
            json.dump(sparsity, file)

        with open(f'exp3-runtimes-{name}-fold-{fold}.json', 'w') as file:
            json.dump(runtimes, file)
            
            
def plot():
    cancel_lams = {'heloc' : 1e-2, 'house' : 1e-4, 'adult' : 1e-2, 'magic' : 1e-2, 'diabetes' : 1e-2, 'chess' : 1e-4, 'backnote' : 1e-2, 'tictactoe' : 1e-6}
    folds = [1, 2, 3, 4]
    feature_selectors = ['rf1', 'rf2', 'rf3', 'gb', 'lda', 'svm', 'r2ntab']
    for name in ['adult', 'heloc', 'house', 'magic', 'tictactoe', 'backnote', 'chess', 'diabetes']:
    
        aucs = {fs: [] for fs in feature_selectors}
        sparsity = {fs: [] for fs in feature_selectors}
        runtimes = {fs: [] for fs in feature_selectors}

        print('dataset:', name)
        
        for fold in folds:

            with open(f'exp3-auc-{name}-fold-{fold}.json') as file:
                aucs = json.load(file)

            with open(f'exp3-sparsities-{name}-fold-{fold}.json') as file:
                sparsities = json.load(file)
            
            with open(f'exp3-runtimes-{name}-fold-{fold}.json') as file:
                run_times = json.load(file)


            for fs in feature_selectors:
                aucs[fs].append(np.mean(aucs[f'{fs}']))
                sparsity[fs].append(np.mean(sparsities[f'{fs}']))
                runtimes[fs].append(np.mean(run_times[f'{fs}']))
            
        for fs in feature_selectors:
            print(fs)
            print(np.mean(aucs[fs]))
            print(np.mean(sparsities[fs]))
            print(np.mean(runtimes[fs]))


if __name__ == "__main__":
    fold = int(sys.argv[1])
    run(fold)
