import torch
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wittgenstein as rule
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from datasets.dataset import transform_dataset, kfold_dataset, predefined_dataset
from R2Ntab import R2Ntab
from rulelist import RuleList
from sklearn.metrics import roc_auc_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_learner(rule_learner, X_train, X_test, Y_train, Y_test, train_set, test_set, X_headers, batch_size, lr_cancel, cancel_lam, conds):
    aucs, n_rules, conditions = [], [], []
    
    RX_train = pd.DataFrame(X_train)
    RX_train = RX_train.sort_index(axis=1)
    RX_test = pd.DataFrame(X_test)
    RX_test = RX_test.sort_index(axis=1)

    if rule_learner == 'r2ntab':
        model = R2Ntab(train_set[:][0].size(1), 50, 1)
        aucs, n_rules, conditions = model.fit(train_set, test_set, batch_size=batch_size, lr_cancel=lr_cancel, cancel_lam=cancel_lam, max_conditions=conds)
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

def run():
    rule_learners = ['r2ntab', 'ripper', 'cart', 'c4.5', 'classy']
    folds = 5
    dataset_names = ['chess', 'diabetes' , 'backnote', 'tictactoe']
    conds = {'house' : [20, 35, 50, 65, 80], 'adult' : [25, 40, 55, 70, 95], 'heloc' : [10, 25, 40, 55, 70], 'magic' : [60, 75, 90, 105, 120], 'chess' : [120, 130, 140, 160, 180], 'diabetes' : [110, 120, 140, 160, 180], 'tictactoe' : [110, 120, 140, 160, 180], 'backnote' : [110, 120, 140, 160, 180]}
    cancel_lams = {'heloc' : 1e-2, 'house' : 1e-4, 'adult' : 1e-2, 'magic' : 1e-2, 'diabetes' : 1e-2, 'chess' : 1e-4, 'backnote' : 1e-2, 'tictactoe' : 1e-6}
    for name in dataset_names:
        print(f'dataset: {name}')
        aucs = {rl: [] for rl in rule_learners}
        rules = {rl: [] for rl in rule_learners}
        conditions = {rl: [] for rl in rule_learners}
        runtimes = {rl: [] for rl in rule_learners}

        X, Y, X_headers, Y_headers = transform_dataset(name, method='onehot-compare', negations=False, labels='binary')
        datasets = kfold_dataset(X, Y, shuffle=1)
        
        batch_size = 400 if len(X) > 10e3 else 40
        
        if name in ['chess', 'heloc']:
            lr_cancel = 1e-2
        else:
            lr_cancel = 5e-3
            
        for fold in range(folds):
            print(f'  fold: {fold+1}')
            X_train, X_test, Y_train, Y_test = datasets[fold]
            train_set = torch.utils.data.TensorDataset(torch.Tensor(X_train.to_numpy()), torch.Tensor(Y_train))
            test_set = torch.utils.data.TensorDataset(torch.Tensor(X_test.to_numpy()), torch.Tensor(Y_test))

            for learner in rule_learners:
                start = time.time()
                auc, n_rules, n_conds = run_learner(learner, X_train, X_test, Y_train, Y_test, train_set, test_set, X_headers, batch_size, lr_cancel, cancel_lams[name], conds[name].copy())
                end = time.time()
                runtime = end-start
                aucs[learner].append(auc)
                rules[learner].append(n_rules)
                conditions[learner].append(n_conds)
                runtimes[learner].append(runtime)
                
        for learner in rule_learners:
            aucs[learner] = np.mean(aucs[learner], axis=0).tolist()
            rules[learner] = np.mean(rules[learner], axis=0).tolist()
            conditions[learner] = np.mean(conditions[learner], axis=0).tolist()
            runtimes[learner] = np.mean(runtimes[learner], axis=0).tolist()
                
        with open(f'exp4-aucs-{name}.json', 'w') as file:
            json.dump(aucs, file)
            
        with open(f'exp4-rules-{name}.json', 'w') as file:
            json.dump(rules, file)
            
        with open(f'exp4-conditions-{name}.json', 'w') as file:
            json.dump(conditions, file)
            
        with open(f'exp4-runtimes-{name}.json', 'w') as file:
            json.dump(runtimes, file)

def plot():
    for name in ['heloc']:
    
        with open(f'exp4-aucs-{name}.json') as file:
            aucs = json.load(file)
            
        with open(f'exp4-sparsities-{name}.json') as file:
            sparsities = json.load(file)
            
        for fs in ['ripper', 'cart', 'c4.5', 'classy', 'r2ntab']:
            plt.plot(sparsities[f'{fs}'], aucs[f'{fs}'])

        plt.legend(['ripper', 'cart', 'c4.5', 'classy', 'r2ntab'])
        plt.xlim([0,120])
        plt.show()


if __name__ == "__main__":
    run()
