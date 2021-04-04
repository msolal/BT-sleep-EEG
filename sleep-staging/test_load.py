# %%
# 0a. Setting up the environment

import numpy as np
import os
import torch

from datasets.bids import BIDS
from braindecode.datautil.preprocess import (zscore, NumpyPreproc,
                                             MNEPreproc, preprocess)
from braindecode.datautil.windowers import create_windows_from_events
from visualisation.windows import view_nb_windows
from sklearn.metrics import (confusion_matrix, classification_report,
                             balanced_accuracy_score, cohen_kappa_score)
from visualisation.results import (save_score, plot_confusion_matrix,
                                   plot_history, plot_classification_report)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cmap = sns.cubehelix_palette(50)

# %%
mapping = {'Sleep stage W': 0,
           'Sleep stage 1': 1,
           'Sleep stage 2': 2,
           'Sleep stage 3': 3,
           'Sleep stage 3/4': 3,
           'Sleep stage 4': 3,
           'Sleep stage R': 4}
classes_mapping = {'0': 'W', '1': 'N1', '2': 'N2', '3': 'N3', '4': 'REM'}

# %%
train_valid = ['Clinical', '9ch', 48]

sfreq = 100
window_size_s = 30
lr = 5e-4           # lr can be 5e-4 or 1e-3
n_epochs = 10
batch_size = 8

plots_path = f'plots/report/'
train_desc = f'{train_valid[0]}_{train_valid[1]}-{train_valid[2]}-lr{lr}_batch{batch_size}_{n_epochs}epochs'
# models_path = '/storage/store2/work/msolal/trained_models/' + train_desc
models_path = '/storage/store2/work/msolal/trained_models/' + train_desc

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clf = torch.load(models_path, map_location=torch.device(device))

# %%
test = ['MASS', '9ch', 12]
plots_name = 'clin-mass'

dataset = BIDS(dataset=test[0],
               derivatives=test[1],
               subject_ids=test[2])

# Extracting windows
window_size_samples = window_size_s * sfreq
windows_dataset = create_windows_from_events(
    dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
    window_size_samples=window_size_samples, drop_last_window=True,
    window_stride_samples=window_size_samples, preload=True, mapping=mapping)
# Window preprocessing
preprocess(windows_dataset, [MNEPreproc(fn=zscore)])
test_set = windows_dataset

y_true = np.concatenate(
    tuple([test_set.datasets[i].windows.metadata['target'].values
           for i in range(len(test_set.datasets))]))
y_pred = clf.predict(test_set)

test_bal_acc = balanced_accuracy_score(y_true, y_pred)
test_kappa = cohen_kappa_score(y_true, y_pred)
print(f'Balanced accuracy: {test_bal_acc}\nKappa score: {test_kappa}')

conf_mat = confusion_matrix(y_true, y_pred, normalize='true')
confusion_df = pd.DataFrame(conf_mat, columns=classes_mapping.values(),
                            index=classes_mapping.values())
plt.figure()
ax = sns.heatmap(confusion_df, annot=True, fmt='.2f',
                 cmap=cmap, linewidths=.01, square=True, cbar=False,
                 annot_kws={"size": 14})
plt.yticks(rotation=0)
plt.savefig(plots_path + 'conf_mat/' + plots_name, facecolor='w')

class_report = classification_report(y_true, y_pred, zero_division=1)
class_report = class_report.replace('\n\n', '\n')
class_report = class_report.replace(' / ', '/')
lines = class_report.split('\n')

classes, values_matrix, support, mask_matrix = [], [], [], []
for line in lines[1:-1]:
    splitted_line = line.strip().split()
    support.append(int(splitted_line[-1]))
    if len(splitted_line) == 3:
        classes.append(splitted_line[0])
        values = [0, 0, float(splitted_line[1])]
        mask = [True, True, False]
    elif len(splitted_line) > 5:
        classes.append(splitted_line[0]+'_'+splitted_line[1])
        values = [float(x) for x in splitted_line[2: -1]]
        mask = [False, False, False]
    else:
        classes.append(splitted_line[0])
        values = [float(x) for x in splitted_line[1: -1]]
        mask = [False, False, False]
    values_matrix.append(values)
    mask_matrix.append(mask)

values_matrix = np.array(values_matrix)
mask_matrix = np.array(mask_matrix)
xlabels = ['Precision', 'Recall', 'F1-score']
ylabels = ['{} ({})'.format(classes_mapping[idx] if idx in classes_mapping else idx, sup)
                for idx, sup in zip(classes, support)]

report_df = pd.DataFrame(values_matrix, columns=xlabels, index=ylabels)
plt.figure()
ax = sns.heatmap(report_df, annot=True, cmap=cmap, linewidths=.01, square=True,
                 mask=mask_matrix, cbar=False)
ax.tick_params(left=False, bottom=False)

plt.savefig(plots_path + 'class_report/' + plots_name, facecolor='w')

# %%
