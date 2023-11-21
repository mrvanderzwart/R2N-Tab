import torch
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../src')
sys.path.insert(0, '../src/include')
sys.path.insert(0, '../datasets')

from datasets.dataset import transform_dataset, kfold_dataset
from R2Ntab import R2Ntab
from DRNet import train as train, DRNet
from sklearn.metrics import roc_auc_score

networks = ['drnet', 'r2ntab2', 'r2ntab4', 'r2ntab6']
def run_network(network, train_set, test_set, X_train, Y_train, X_test, Y_test, X_headers, batch_size): 
    cancel_lam = {'r2ntab2': 1e-2, 'r2ntab4': 1e-4, 'r2ntab6': 1e-6}.get(network, 0)

    start = time.time()
    
    net = R2Ntab(train_set[:][0].size(1), 50, 1) if network.startswith('r2ntab') else DRNet(train_set[:][0].size(1), 50, 1)
    if network.startswith('r2ntab'):
        net.fit(train_set, device='cpu', epochs=1000, batch_size=batch_size, cancel_lam=cancel_lam, lr_cancel=5e-3)
    else:
        train(net, train_set, test_set, device='cpu', epochs=1000, batch_size=batch_size)
    auc = roc_auc_score(net.predict(np.array(X_test)), Y_test)
    train_auc = roc_auc_score(net.predict(np.array(X_train)), Y_train)
    rules = net.extract_rules(X_headers) if network.startswith('r2ntab') else net.get_rules(X_headers)
    conditions = sum(map(len, rules))
    end = time.time()
    runtime = end - start
        
    return auc, train_auc, len(rules), conditions, runtime


def run(dataset_name):
    folds = 1
    runs = 1
    results = {}
    
    results['aucs'] = {}
    results['aucs_train'] = {}
    results['rules'] = {}
    results['conditions'] = {}
    results['runtimes'] = {}
    
    for network in networks:
        results['aucs'][network] = []
        results['aucs_train'][network] = []
        results['rules'][network] = []
        results['conditions'][network] = []
        results['runtimes'][network] = []

    X, Y, X_headers, Y_headers = transform_dataset(dataset_name, method='onehot-compare', negations=False, labels='binary')
    datasets = kfold_dataset(X, Y, k=5, shuffle=1)
    
    batch_size = 400 if len(X) > 10e3 else 40

    for fold in range(folds):
        print(f'  fold: {fold+1}')
        X_train, X_test, Y_train, Y_test = datasets[fold]
        train_set = torch.utils.data.TensorDataset(torch.Tensor(X_train.to_numpy()), torch.Tensor(Y_train))
        test_set = torch.utils.data.TensorDataset(torch.Tensor(X_test.to_numpy()), torch.Tensor(Y_test))

        for network in networks:
            auc_values, train_aucs, n_rules, n_conds, run_times = 0, 0, 0, 0, 0
            for run in range(runs):
                print(f'  run: {run+1}')
                auc, train_auc, n_rule, conds, runtime = run_network(network, train_set, test_set, X_train, Y_train, X_test, Y_test, X_headers, batch_size)

                auc_values += auc
                train_aucs += train_auc
                n_rules += n_rule
                n_conds += conds
                run_times += runtime
                
            results['aucs'][network].append(auc_values/runs)
            results['aucs_train'][network].append(train_aucs/runs)
            results['rules'][network].append(n_rules/runs)
            results['conditions'][network].append(n_conds/runs)
            results['runtimes'][network].append(run_times/runs)

        with open(f'exp1-{dataset_name}.json', 'w') as file:
            json.dump(results, file)
            
            
def plot():
    dataset_names = ['heloc', 'house', 'magic', 'tictactoe', 'diabetes', 'chess', 'backnote']
    for name in dataset_names:

        print('dataset:', name)

        with open(f'exp1-{name}.json') as file:
            results = json.load(file)

        for rate in ['2', '4', '6']:

            print('  rate: 1e-', rate)

            plt.style.use('seaborn-darkgrid')
            l1 = plt.scatter(results['conditions']['drnet'], results['aucs']['drnet'], c='red', label='DR-Net')
            l2 = plt.scatter(results['conditions'][f'r2ntab{rate}'], results['aucs'][f'r2ntab{rate}'], c='blue', label=f'R2N-Tab $\lambda_3$=1e-{rate}')

            mean_auc_drnet = sum(results['aucs']['drnet']) / len(results['aucs']['drnet'])
            mean_auc_r2ntab = sum(results['aucs'][f'r2ntab{rate}']) / len(results['aucs'][f'r2ntab{rate}'])
            mean_sparsity_drnet = sum(results['conditions']['drnet']) / len(results['conditions']['drnet'])
            mean_sparsity_r2ntab = sum(results['conditions'][f'r2ntab{rate}']) / len(results['conditions'][f'r2ntab{rate}'])

            print(f'    mean AUC R2N-Tab {rate}:', mean_auc_r2ntab, 'std:', np.std(results['aucs'][f'r2ntab{rate}']))
            print(f'    mean rules R2N-Tab {rate}:', sum(results['rules'][f'r2ntab{rate}']) / len(results['rules'][f'r2ntab{rate}']), 'std:', np.std(results['rules'][f'r2ntab{rate}']))
            print(f'    mean conditions R2N-Tab {rate}:', mean_sparsity_r2ntab, 'std:', np.std(results['conditions'][f'r2ntab{rate}']))
            print(f'    mean runtimes R2N-Tab {rate}:', sum(results['runtimes'][f'r2ntab{rate}']) / len(results['runtimes'][f'r2ntab{rate}']), 'std:', np.std(results['runtimes'][f'r2ntab{rate}']))
            if rate == '2':
                print('    mean AUC DR-Net:', mean_auc_drnet, 'std:', np.std(results['aucs']['drnet']))
                print('    mean rules DR-Net:', sum(results['rules']['drnet']) / len(results['rules']['drnet']), 'std:', np.std(results['rules']['drnet']))
                print('    mean conditions DR-Net:', mean_sparsity_drnet, 'std:', np.std(results['conditions']['drnet']))
                print('    mean runtimes DR-Net:', sum(results['runtimes']['drnet']) / len(results['runtimes']['drnet']), 'std:', np.std(results['runtimes']['drnet']))

            plt.scatter(mean_sparsity_drnet, mean_auc_drnet, marker='X', edgecolors='black', s=120, c='red')
            plt.scatter(mean_sparsity_r2ntab, mean_auc_r2ntab, marker='X', edgecolors='black', s=120, c='blue')

            combi = plt.Line2D([0], [0], marker='X', color='w', label='Mean', markerfacecolor='black', markersize=12)

            plt.xlabel('# Conditions', fontsize=15)
            plt.ylabel('ROC-AUC', fontsize=15)
            plt.ylim([mean_auc_r2ntab-0.1, min(1.0, mean_auc_drnet+0.04)])    
            plt.legend(handles=[l1, l2, combi], loc='lower right', fontsize=12, frameon=True)
            plt.savefig(f'{name}-{rate}.pdf')
            plt.show()


if __name__ == "__main__":
    #dataset_name = sys.argv[1]
    #run(dataset_name)
    plot()
