from AIX360.aix360.algorithms.rbm import BRCGExplainer, BooleanRuleCG
from AIX360.aix360.algorithms.rbm import FeatureBinarizer
from datasets.dataset import kfold_dataset, transform_dataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import sys
import time
import json

results = {}

results['auc'] = []
results['runtime'] = []
results['rules'] = []
results['conditions'] = []

name = sys.argv[1]

for fold in range(5):
    X_table, Y_table, _, _ = transform_dataset(name)
    datasets = kfold_dataset(X_table, Y_table)
    X_train, X_test, Y_train, Y_test = datasets[fold-1]

    fb = FeatureBinarizer(negations=True)
    X_train_fb = fb.fit_transform(X_train)
    X_test_fb = fb.fit_transform(X_test)

    start = time.time()

    boolean_model = BooleanRuleCG(timeMax=500)
    explainer = BRCGExplainer(boolean_model)
    explainer.fit(X_train_fb, Y_train)

    Y_pred = explainer.predict(X_test_fb)
    e = explainer.explain()

    end = time.time()
    results['auc'].append(roc_auc_score(Y_pred, Y_test))
    results['runtime'].append(end-start)
    conditions = 0
    for rule in e['rules']:
        conditions += (rule.count(' AND ')+1)
        
    results['rules'].append(len(e['rules']))
    results['conditions'].append(conditions)

with open(f'CG-{name}.json', 'w') as file:
    json.dump(results, file)
