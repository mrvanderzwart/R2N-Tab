import torch
import json
import numpy as np
import matplotlib.pyplot as plt

from datasets.dataset import transform_dataset, kfold_dataset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from R2Ntab import R2Ntab


def fix_cancel_layer(net, features):
    with torch.no_grad():
        for feature in features:
            net.cancelout_layer.weight[feature] = -1


def run_selector(feature_selector, train_set, test_set, X_train, Y_train, X_test, Y_test, X_headers):
    if feature_selector == 'r2ntab':
        model = R2Ntab(train_set[:][0].size(1), 50, 1)
        model.fit(train_set, test_set, device='cpu', epochs=10, batch_size=400)
        accuracy = model.predict(X_test, Y_test)
        sparsity = sum(map(len, model.extract_rules(X_headers)))
    elif feature_selector == 'gb':
        gb = GradientBoostingClassifier(n_estimators=50, random_state=0)
        gb.fit(X_train, Y_train)
        cancelled_features = np.where(gb.feature_importances_ == 0)[0]
        model = R2Ntab(train_set[:][0].size(1), 50, 1)
        fix_cancel_layer(model, cancelled_features)
        model.fit(train_set, test_set, device='cpu', epochs=10, batch_size=400)
        accuracy = model.predict(X_test, Y_test)
        sparsity = sum(map(len, model.extract_rules(X_headers)))
    elif feature_selector == 'pca':
        pca = PCA(n_components=1)
        pca.fit(X_train)
        component = pca.components_[0]
        cancelled_features = list(torch.where(torch.tensor(component) < 0)[0].numpy())
        model = R2Ntab(train_set[:][0].size(1), 50, 1)
        fix_cancel_layer(model, cancelled_features)
        model.fit(train_set, test_set, device='cpu', epochs=10, batch_size=400)
        accuracy = model.predict(X_test, Y_test)
        sparsity = sum(map(len, model.extract_rules(X_headers)))
        
    return accuracy, sparsity


def run():
    folds = 5
    feature_selectors = ['r2ntab', 'gb', 'pca']
    for name in ['adult', 'heloc', 'house', 'magic']:
        accuracies = {}
        sparsities = {}
        for fs in feature_selectors:
            accuracies[fs] = []
            sparsities[fs] = []

        X, Y, X_headers, Y_headers = transform_dataset(name, method='onehot-compare', negations=False, labels='binary')
        datasets = kfold_dataset(X, Y, shuffle=1)

        print(f'dataset: {name}')
        for fold in range(folds):
            print(f'  fold: {fold+1}') 
            X_train, X_test, Y_train, Y_test = datasets[fold]
            train_set = torch.utils.data.TensorDataset(torch.Tensor(X_train.to_numpy()), torch.Tensor(Y_train))
            test_set = torch.utils.data.TensorDataset(torch.Tensor(X_test.to_numpy()), torch.Tensor(Y_test))

            for fs in feature_selectors:
                acc, sparsity = run_selector(fs, train_set, test_set, X_train, Y_train, X_test, Y_test, X_headers)

                accuracies[fs].append(acc)
                sparsities[fs].append(sparsity)

        with open(f'exp3-accuracies-{name}.json', 'w') as file:
            json.dump(accuracies, file)

        with open(f'exp3-sparsities-{name}.json', 'w') as file:
            json.dump(sparsities, file)
            
            
def plot():
    for name in ['adult', 'heloc', 'house', 'magic']:

        with open(f'exp3-accuracies-{name}.json') as file:
            accuracies = json.load(file)

        with open(f'exp3-sparsities-{name}.json') as file:
            sparsities = json.load(file)

        plt.style.use('seaborn-darkgrid')
        l1 = plt.scatter(sparsities['gb'], accuracies['gb'], c='red', label='Gradient Boosting + R2N-Tab $\eta$=0')
        l2 = plt.scatter(sparsities['pca'], accuracies['pca'], c='orange', label='PCA + R2N-Tab $\eta$=0')
        l3 = plt.scatter(sparsities['r2ntab'], accuracies['r2ntab'], c='blue', label='R2N-Tab $\lambda$=1e-4')

        mean_accuracy_gb = sum(accuracies['gb']) / len(accuracies['gb'])
        mean_accuracy_pca = sum(accuracies['pca']) / len(accuracies['pca'])
        mean_accuracy_r2ntab = sum(accuracies['r2ntab']) / len(accuracies['r2ntab'])
        mean_sparsity_gb = sum(sparsities['gb']) / len(sparsities['gb'])
        mean_sparsity_pca = sum(sparsities['pca']) / len(sparsities['pca'])
        mean_sparsity_r2ntab = sum(sparsities['r2ntab']) / len(accuracies['r2ntab'])

        plt.scatter(mean_sparsity_gb, mean_accuracy_gb, marker='X', edgecolors='black', s=120, c='red')
        plt.scatter(mean_sparsity_pca, mean_accuracy_pca, marker='X', edgecolors='black', s=120, c='orange')
        plt.scatter(mean_sparsity_r2ntab, mean_accuracy_r2ntab, marker='X', edgecolors='black', s=120, c='blue')

        combined_legend = plt.Line2D([0], [0], marker='X', color='w', label='Mean', markerfacecolor='black', markersize=12)

        plt.xlabel('# Conditions', fontsize=15)
        plt.ylabel('Test accuracy', fontsize=15)
        minimum = min(mean_accuracy_gb, mean_accuracy_pca, mean_accuracy_r2ntab)
        plt.ylim([minimum-0.1, mean_accuracy_gb+0.04])    
        plt.legend(handles=[l1, l2, l3, combined_legend], loc='lower right', fontsize=12)
        plt.savefig(f'exp3-{name}.pdf')
        plt.clf()