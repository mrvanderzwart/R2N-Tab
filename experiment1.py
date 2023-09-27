import torch
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from datasets.dataset import transform_dataset, kfold_dataset
from R2Ntab import R2Ntab
from DRNet import train as train, DRNet
from sklearn.metrics import roc_auc_score


networks = ['drnet', 'r2ntab2', 'r2ntab4', 'r2ntab6']
def run_network(network, train_set, test_set, X_test, Y_test, X_headers, batch_size): 
    cancel_lam = {'r2ntab2': 1e-2, 'r2ntab4': 1e-4, 'r2ntab6': 1e-6}.get(network, 0)

    start = time.time()
    
    net = R2Ntab(train_set[:][0].size(1), 50, 1) if network.startswith('r2ntab') else DRNet(train_set[:][0].size(1), 50, 1)
    if network.startswith('r2ntab'):
        net.fit(train_set, device='cpu', epochs=1000, batch_size=batch_size, cancel_lam=cancel_lam, lr_cancel=5e-3)
    else:
        train(net, train_set, test_set, device='cpu', epochs=1000, batch_size=batch_size)
    auc = roc_auc_score(net.predict(np.array(X_test)), Y_test)
    rules = net.extract_rules(X_headers) if network.startswith('r2ntab') else net.get_rules(X_headers)
    conditions = sum(map(len, rules))

    end = time.time()
    runtime = end - start
        
    return auc, len(rules), conditions, runtime


def run():
    folds = 5
    runs = 5
    dataset_names = ['adult', 'backnote', 'diabetes', 'house', 'magic']
    for name in dataset_names:
        aucs = {network: [] for network in networks}
        rules = {network: [] for network in networks}
        conditions = {network: [] for network in networks}
        runtimes = {network: [] for network in networks}

        X, Y, X_headers, Y_headers = transform_dataset(name, method='onehot-compare', negations=False, labels='binary')
        datasets = kfold_dataset(X, Y, k=folds, shuffle=1)
        
        batch_size = 400 if len(X) > 10e3 else 40

        print(f'dataset: {name}')
        for fold in range(folds):
            print(f'  fold: {fold+1}')
            X_train, X_test, Y_train, Y_test = datasets[fold]
            train_set = torch.utils.data.TensorDataset(torch.Tensor(X_train.to_numpy()), torch.Tensor(Y_train))
            test_set = torch.utils.data.TensorDataset(torch.Tensor(X_test.to_numpy()), torch.Tensor(Y_test))

            for network in networks:
                auc_values, n_rules, n_conds, run_times = 0, 0, 0, 0
                for run in range(runs):
                    print(f'  run: {run+1}')
                    auc, n_rule, conds, runtime = run_network(network, train_set, test_set, X_test, Y_test, X_headers, batch_size)

                    auc_values += auc
                    n_rules += n_rule
                    n_conds += conds
                    run_times += runtime
                    
                aucs[network].append(auc_values/runs)
                rules[network].append(n_rules/runs)
                conditions[network].append(n_conds/runs)
                runtimes[network].append(run_times/runs)

        with open(f'exp1-aucs-{name}.json', 'w') as file:
            json.dump(aucs, file)

        with open(f'exp1-rules-{name}.json', 'w') as file:
            json.dump(rules, file)
            
        with open(f'exp1-conditions-{name}.json', 'w') as file:
            json.dump(conditions, file)

        with open(f'exp1-runtimes-{name}.json', 'w') as file:
            json.dump(runtimes, file)
            
            
def plot():
    dataset_names = ['adult', 'heloc', 'house', 'magic', 'tictactoe', 'chess', 'diabetes', 'backnote']
    for name in dataset_names:

        print('dataset:', name)

        with open(f'exp1-aucs-{name}.json') as file:
            aucs = json.load(file)

        with open(f'exp1-rules-{name}.json') as file:
            rules = json.load(file)

        with open(f'exp1-conditions-{name}.json') as file:
            conds = json.load(file)

        with open(f'exp1-runtimes-{name}.json') as file:
            runtimes = json.load(file)

        for rate in ['2', '4', '6']:

            print('  rate: 1e-', rate)

            plt.style.use('seaborn-darkgrid')
            l1 = plt.scatter(conds['drnet'], aucs['drnet'], c='red', label='DR-Net')
            l2 = plt.scatter(conds[f'r2ntab{rate}'], aucs[f'r2ntab{rate}'], c='blue', label=f'R2N-Tab $\lambda$=1e-{rate}')

            mean_auc_drnet = sum(aucs['drnet']) / len(aucs['drnet'])
            mean_auc_r2ntab = sum(aucs[f'r2ntab{rate}']) / len(aucs[f'r2ntab{rate}'])
            mean_sparsity_drnet = sum(conds['drnet']) / len(conds['drnet'])
            mean_sparsity_r2ntab = sum(conds[f'r2ntab{rate}']) / len(conds[f'r2ntab{rate}'])

            print(f'    mean AUC R2N-Tab {rate}:', mean_auc_r2ntab, 'std:', np.std(aucs[f'r2ntab{rate}']))
            print(f'    mean rules R2N-Tab {rate}:', sum(rules[f'r2ntab{rate}']) / len(rules[f'r2ntab{rate}']), 'std:', np.std(rules[f'r2ntab{rate}']))
            print(f'    mean conditions R2N-Tab {rate}:', mean_sparsity_r2ntab, 'std:', np.std(conds[f'r2ntab{rate}']))
            print(f'    mean runtimes R2N-Tab {rate}:', sum(runtimes[f'r2ntab{rate}']) / len(runtimes[f'r2ntab{rate}']), 'std:', np.std(runtimes[f'r2ntab{rate}']))
            if rate == '2':
                print('    mean AUC DR-Net:', mean_auc_drnet, 'std:', np.std(aucs['drnet']))
                print('    mean rules DR-Net:', sum(rules['drnet']) / len(rules['drnet']), 'std:', np.std(rules['drnet']))
                print('    mean conditions DR-Net:', mean_sparsity_drnet, 'std:', np.std(conds['drnet']))
                print('    mean runtimes DR-Net:', sum(runtimes['drnet']) / len(runtimes['drnet']), 'std:', np.std(runtimes['drnet']))

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
    run()
