import json, os
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'

prediction = open(os.path.join(prediction_dir, 'prediction')).read().splitlines()
truth = pd.read_csv(os.path.join(reference_dir, 'test_labels.csv'))['name'].tolist()

f1 = f1_score(truth, prediction, average='macro')
accuracy = accuracy_score(truth, prediction)

with open(os.path.join(score_dir, 'scores.json'), 'w') as f:
    f.write(json.dumps({'f1_score': f1, 'accuracy': accuracy}))
