import torch
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wittgenstein as rule
import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../src')
sys.path.insert(0, '../src/include')

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from datasets.dataset import transform_dataset, kfold_dataset, predefined_dataset
from R2Ntab import R2Ntab
from rulelist import RuleList
from sklearn.metrics import roc_auc_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_learner(rule_learner, X_train, X_test, Y_train, Y_test, train_set, test_set, X_headers, batch_size, lr_cancel, cancel_lam, epochs):
    aucs, n_rules, conditions = [], [], []
    
    RX_train = pd.DataFrame(X_train)
    RX_train = RX_train.sort_index(axis=1)
    RX_test = pd.DataFrame(X_test)
    RX_test = RX_test.sort_index(axis=1)

    if rule_learner == 'r2ntab':
        model = R2Ntab(train_set[:][0].size(1), 10, 1)
        model.fit(train_set, test_set, batch_size=batch_size, lr_cancel=lr_cancel, cancel_lam=cancel_lam, epochs=epochs)
        Y_pred = model.predict(X_test)
        aucs = roc_auc_score(Y_pred, Y_test)
        rules = model.extract_rules()
        n_rules = len(rules)
        conditions = sum(map(len, rules))
    elif rule_learner == 'ripper':
        for max_conditions in [50, 75, 100, 200, 300]:
            model = rule.RIPPER(max_total_conds=max_conditions)
            model.fit(RX_train, Y_train)
            aucs.append(roc_auc_score(model.predict(RX_test), Y_test))
            n_rules.append(len(model.ruleset_))
            conditions.append(sum(len(rule) for rule in model.ruleset_))
    elif rule_learner == 'cart':
        for max_depth in [2, 3, 5, 7, 8]:
            model = DecisionTreeClassifier(max_depth=max_depth)
            model.fit(X_train, Y_train)
            aucs.append(roc_auc_score(model.predict(X_test), Y_test))
            n_rules.append(export_text(model, feature_names=X_train.columns.tolist()).count('class'))
            conditions.append(export_text(model, feature_names=X_train.columns.tolist()).count('('))
    elif rule_learner == 'c4.5':
        for max_depth in [2, 3, 5, 7, 8]:
            model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
            model.fit(X_train, Y_train)
            aucs.append(roc_auc_score(model.predict(X_test), Y_test))
            n_rules.append(export_text(model, feature_names=X_train.columns.tolist()).count('class'))
            conditions.append(export_text(model, feature_names=X_train.columns.tolist()).count('('))
    elif rule_learner == 'classy':
        for max_depth in [1, 2, 5, 7, 9]:
            model = RuleList(task='prediction', target_model='categorical', max_depth=max_depth)
            rules = model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            aucs.append(roc_auc_score(Y_test, Y_pred))
            rules = str(rules)
            n_rules.append(rules.count("If") + rules.count("ELSE IF"))
            conditions.append(rules.count("AND") + rules.count("If") + rules.count("ELSE IF"))
        
    return aucs, n_rules, conditions

def run(dataset_name, fold):
    rule_learners = ['r2ntab']
    #conds = {'house' : [20, 35, 50, 65, 80], 'adult' : [25, 40, 55, 70, 95], 'heloc' : [10, 25, 40, 55, 70], 'magic' : [60, 75, 90, 105, 120], 'chess' : [120, 130, 140, 160, 180], 'diabetes' : [110, 120, 140, 160, 180], 'tictactoe' : [110, 120, 140, 160, 180], 'backnote' : [110, 120, 140, 160, 180]}
    cancel_lams = {'heloc' : 1e-2, 'house' : 1e-4, 'adult' : 1e-2, 'magic' : 1e-2, 'diabetes' : 1e-2, 'chess' : 1e-4, 'backnote' : 1e-2, 'tictactoe' : 1e-6}
    results = {}
    runs = 5

    results['aucs'] = {}
    results['rules'] = {}
    results['conditions'] = {}
    results['runtimes'] = {}
    
    for rl in rule_learners:
        results['aucs'][rl] = []
        results['rules'][rl] = []
        results['conditions'][rl] = []
        results['runtimes'][rl] = []

    X, Y, X_headers, Y_headers = transform_dataset(dataset_name, method='onehot-compare', negations=False, labels='binary')
    datasets = kfold_dataset(X, Y, shuffle=1)
    
    batch_size = 400 if len(X) > 10e3 else 40

    if dataset_name == 'heloc':
        epochs = 10000
    elif dataset_name == 'house' or dataset_name == 'magic':
        epochs = 8000
    else:
        epochs = 5000
    
    if dataset_name in ['chess', 'heloc']:
        lr_cancel = 1e-2
    else:
        lr_cancel = 5e-3
        
    X_train, X_test, Y_train, Y_test = datasets[fold]
    train_set = torch.utils.data.TensorDataset(torch.Tensor(X_train.to_numpy()), torch.Tensor(Y_train))
    test_set = torch.utils.data.TensorDataset(torch.Tensor(X_test.to_numpy()), torch.Tensor(Y_test))

    for learner in rule_learners:
        print(f'  rule learner: {learner}')
        learner_aucs, learner_rules, learner_conds, learner_runtimes = [], [], [], []
        for run in range(runs):
            print(f'    run: {run+1}')
            start = time.time()
            auc, n_rules, n_conds = run_learner(learner, X_train, X_test, Y_train, Y_test, train_set, test_set, X_headers, batch_size, lr_cancel, cancel_lams[dataset_name], epochs)
            end = time.time()
            runtime = end-start
            learner_aucs.append(auc)
            learner_rules.append(n_rules)
            learner_conds.append(n_conds)
            learner_runtimes.append(runtime)
            
        results['aucs'][learner] = np.mean(learner_aucs, axis=0).tolist()
        results['rules'][learner] = np.mean(learner_rules, axis=0).tolist()
        results['conditions'][learner] = np.mean(learner_conds, axis=0).tolist()
        results['runtimes'][learner] = np.mean(learner_runtimes, axis=0).tolist()
            
    with open(f'exp4-{dataset_name}-fold-{fold+1}.json', 'w') as file:
        json.dump(results, file)


def plot():
    colors = ['darkslategrey', 'mediumblue', 'orange', 'green', 'purple']
    plt.style.use('seaborn-darkgrid')
    for name in ['adult', 'house']:
    
        with open(f'{name}_graph_auc.json') as file:
            aucs = json.load(file)
            
        with open(f'{name}_graph_conds.json') as file:
            conds = json.load(file)
            
        if name == 'house':
            plt.scatter(conds['ripper'][1], aucs['ripper'][1], marker='x', c='mediumblue', label='_nolegend_')
            plt.scatter(conds['ripper'][2], aucs['ripper'][2], marker='x', c='mediumblue', label='_nolegend_')
            conds['ripper'] = [conds['ripper'][0], conds['ripper'][3], conds['ripper'][4]]
            aucs['ripper'] = [aucs['ripper'][0], aucs['ripper'][3], aucs['ripper'][4]]
            
        if name == 'adult':
            plt.scatter(conds['r2ntab'][3], aucs['r2ntab'][3], marker='x', c='darkslategrey', label='_nolegend_')
            conds['r2ntab'] = [conds['r2ntab'][0], conds['r2ntab'][1], conds['r2ntab'][2], conds['r2ntab'][4]]
            aucs['r2ntab'] = [aucs['r2ntab'][0], aucs['r2ntab'][1], aucs['r2ntab'][2], aucs['r2ntab'][4]]
            
        for i, fs in enumerate(['r2ntab', 'ripper', 'cart', 'c4.5', 'classy']):
            plt.plot(conds[f'{fs}'], aucs[f'{fs}'], '-x', color=colors[i])

        l = plt.legend(['R2N-Tab', 'RIPPER', 'CART', 'C4.5', 'Classy'], fontsize=12, frameon=True)
        for i in l.legendHandles:
            i.set_marker('')
            
        if name == 'house':
            plt.xlim([0,350])
        elif name == 'adult':
            plt.xlim([0,200])
        plt.xlabel('model complexity')
        plt.ylabel('AUC')
        plt.savefig(f'exp4-graph-{name}.pdf')
        plt.show()


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    fold = int(sys.argv[2])
    run(dataset_name, fold-1)
