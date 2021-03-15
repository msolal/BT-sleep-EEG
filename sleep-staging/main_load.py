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


# %%
# 0b. Setting all the constants

mapping = {'Sleep stage W': 0,
           'Sleep stage 1': 1,
           'Sleep stage 2': 2,
           'Sleep stage 3': 3,
           'Sleep stage 3/4': 3,
           'Sleep stage 4': 3,
           'Sleep stage R': 4}
classes_mapping = {'0': 'W', '1': 'N1', '2': 'N2', '3': 'N3', '4': 'R'}

# datasets = ['Clinical', 'MASS_SS3']
# derivatives = ['6channels', '6channels']
datasets = ['MASS_SS3', 'Clinical']
derivatives = ['6channels', '6channels']
sizes = [48, 12]

sfreq = 100
window_size_s = 30
lr = 5e-4           # lr can be 5e-4 or 1e-3
n_epochs = 10
batch_size = 8

print_datasets = f'{datasets[0]}_{datasets[1]}'
print_sizes = f'{sizes[0]}_{sizes[1]}'

plots_path = f'plots/{print_datasets}-{print_sizes}-lr{lr}_batch{batch_size}_{n_epochs}epochs/'
train_desc = f'{datasets[0]}-{sizes[0]}-lr{lr}_batch{batch_size}_{n_epochs}epochs'
# models_path = '/storage/store2/work/msolal/trained_models/' + train_desc
models_path = '/media/pallanca/datapartition/maelys/trained_models/' + train_desc

# %%
# 1. Loading the data

try:
    os.mkdir(plots_path)
    print(f'Directory {plots_path} created\n')
except FileExistsError:
    print(f'Directory {plots_path} already exists\n')

dataset = BIDS(dataset=datasets[1],
               derivatives=derivatives[1],
               subject_ids=sizes[1])
print(dataset.description)

# %%
# 2. Preprocessing

preprocess(dataset, [NumpyPreproc(fn=lambda x: x * 1e6)])

# Extracting windows
window_size_samples = window_size_s * sfreq
windows_dataset = create_windows_from_events(
    dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
    window_size_samples=window_size_samples, drop_last_window=True,
    window_stride_samples=window_size_samples, preload=True, mapping=mapping)
# Window preprocessing
preprocess(windows_dataset, [MNEPreproc(fn=zscore)])

# %%
# 3. Making train, valid and test splits

test_set = windows_dataset
print(view_nb_windows(plots_path, None, None, test_set))

# %%
# 4. Loading the model

clf = torch.load(models_path)

# %%
# 5. Testing

y_true = np.concatenate(
    tuple([test_set.datasets[i].windows.metadata['target'].values
           for i in range(len(test_set.datasets))]))
y_pred = clf.predict(test_set)

# %%
# 6. Visualising results
test_bal_acc = balanced_accuracy_score(y_true, y_pred)
test_kappa = cohen_kappa_score(y_true, y_pred)
save_score(plots_path, test_bal_acc, test_kappa)

plot_history(plots_path, clf)

# Finally, we also display the confusion matrix and classification report
conf_mat = confusion_matrix(y_true, y_pred, normalize='true')
plot_confusion_matrix(plots_path, conf_mat, classes_mapping)
print(conf_mat)

class_report = classification_report(y_true, y_pred)
plot_classification_report(plots_path, class_report, classes_mapping)
print(class_report)
