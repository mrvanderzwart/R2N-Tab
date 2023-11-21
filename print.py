import json
import numpy as np

for name in ['backnote', 'diabetes']:
    with open(f'CG-{name}.json') as f:
        results = json.load(f)
        
    print(f'{name}\n')
    print(f'auc:{np.mean(results["auc"])}')
    print(f'rules:{np.mean(results["rules"])}')
    print(f'conditions:{np.mean(results["conditions"])}')
    print(f'runtime:{np.mean(results["runtime"])}\n')
