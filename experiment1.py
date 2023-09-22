import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from datasets.dataset import transform_dataset, kfold_dataset
from R2Ntab import R2Ntab
from DRNet import train as train, DRNet
from sklearn.metrics import roc_auc_score


networks = ['drnet', 'r2ntab2', 'r2ntab4', 'r2ntab6']
def run_network(network, train_set, test_set, X_test, Y_test, X_headers, batch_size):
    if network == 'drnet':
        net = DRNet(train_set[:][0].size(1), 50, 1)
        train(net, train_set, test_set=test_set, device='cpu', epochs=1000, batch_size=batch_size)
        auc = roc_auc_score(net.predict(np.array(X_test)), Y_test)
        sparsity = sum(map(len, net.get_rules(X_headers)))
    elif network == 'r2ntab2':
        net = R2Ntab(train_set[:][0].size(1), 50, 1)
        net.fit(train_set, device='cpu', epochs=1000, batch_size=batch_size, cancel_lam=1e-2)
        auc = net.score(net.predict(X_test), Y_test)
        sparsity = sum(map(len, net.extract_rules(X_headers)))
    elif network == 'r2ntab4':
        net = R2Ntab(train_set[:][0].size(1), 50, 1)
        net.fit(train_set, device='cpu', epochs=1000, batch_size=batch_size, cancel_lam=1e-4)
        auc = net.score(net.predict(X_test), Y_test)
        sparsity = sum(map(len, net.extract_rules(X_headers)))
    elif network == 'r2ntab6':
        net = R2Ntab(train_set[:][0].size(1), 50, 1)
        net.fit(train_set, device='cpu', epochs=1000, batch_size=batch_size, cancel_lam=1e-6)
        auc = net.score(net.predict(X_test), Y_test)
        sparsity = sum(map(len, net.extract_rules(X_headers)))
        
    return auc, sparsity


def run():
    folds = 5
    dataset_names = [f for f in os.listdir('./datasets') if os.path.isdir(os.path.join('./datasets', f))]
    for name in dataset_names:
        accuracies = {}
        sparsities = {}
        for network in networks:
            accuracies[network] = []
            sparsities[network] = []

        X, Y, X_headers, Y_headers = transform_dataset(name, method='onehot-compare', negations=False, labels='binary')
        datasets = kfold_dataset(X, Y, k=folds, shuffle=1)
        
        if len(X) > 10e3:
            batch_size = 400
        else:
            batch_size = 40

        print(f'dataset: {name}')
        for fold in range(folds):
            print(f'  fold: {fold+1}')
            X_train, X_test, Y_train, Y_test = datasets[fold]
            train_set = torch.utils.data.TensorDataset(torch.Tensor(X_train.to_numpy()), torch.Tensor(Y_train))
            test_set = torch.utils.data.TensorDataset(torch.Tensor(X_test.to_numpy()), torch.Tensor(Y_test))

            for network in networks:
                accuracy, sparsity = run_network(network, train_set, test_set, X_test, Y_test, X_headers, batch_size)
                accuracies[network].append(accuracy)
                sparsities[network].append(sparsity)

        with open(f'exp1-accuracies-{name}.json', 'w') as file:
            json.dump(accuracies, file)

        with open(f'exp1-sparsities-{name}.json', 'w') as file:
            json.dump(sparsities, file)
            
            
def plot():
    dataset_names = [f for f in os.listdir('./datasets') if os.path.isdir(os.path.join('./datasets', f))]
    for name in dataset_names:

        print('dataset:', name)

        with open(f'exp1-accuracies-{name}.json') as file:
            accs = json.load(file)

        with open(f'exp1-sparsities-{name}.json') as file:
            spars = json.load(file)

        for rate in ['2', '4', '6']:

            print('  rate: 1e-', rate)

            plt.style.use('seaborn-darkgrid')
            l1 = plt.scatter(spars['drnet'], accs['drnet'], c='red', label='DR-Net')
            l2 = plt.scatter(spars[f'r2ntab{rate}'], accs[f'r2ntab{rate}'], c='blue', label=f'R2N-Tab $\lambda$=1e-{rate}')

            mean_accuracy_drnet = sum(accs['drnet']) / len(accs['drnet'])
            mean_accuracy_r2ntab = sum(accs[f'r2ntab{rate}']) / len(accs[f'r2ntab{rate}'])
            mean_sparsity_drnet = sum(spars['drnet']) / len(spars['drnet'])
            mean_sparsity_r2ntab = sum(spars[f'r2ntab{rate}']) / len(spars[f'r2ntab{rate}'])

            print('    mean accuracy:', mean_accuracy_r2ntab, 'std:', np.std(accs[f'r2ntab{rate}']))
            print('    mean sparsity:', mean_sparsity_r2ntab, 'std:', np.std(spars[f'r2ntab{rate}']))

            plt.scatter(mean_sparsity_drnet, mean_accuracy_drnet, marker='X', edgecolors='black', s=120, c='red')
            plt.scatter(mean_sparsity_r2ntab, mean_accuracy_r2ntab, marker='X', edgecolors='black', s=120, c='blue')

            combi = plt.Line2D([0], [0], marker='X', color='w', label='Mean', markerfacecolor='black', markersize=12)

            plt.xlabel('# Conditions', fontsize=15)
            plt.ylabel('ROC-AUC', fontsize=15)
            plt.ylim([mean_accuracy_r2ntab-0.1, mean_accuracy_drnet+0.04])    
            plt.legend(handles=[l1, l2, combi], loc='lower right', fontsize=12)
            plt.savefig(f'{name}-{rate}.pdf')
            plt.clf()