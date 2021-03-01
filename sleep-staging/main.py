# %%
# 0a. Setting up the environment

import numpy as np
import os
import torch

from braindecode.datasets import BaseConcatDataset
from datasets.mass_bids import MASS_SS3
from datasets.sleep_physionet_bids import SleepPhysionet
from datasets.clinical import ClinicalDataset
from datautil.preprocess import zscore
from datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess
from datautil.windowers import create_windows_from_events
from datautil.split import split_by_events
from visualisation.windows import view_nb_windows
from util import set_random_seeds
from models.sleep_stager_chambon import SleepStagerChambon2018
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring
from braindecode import EEGClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from visualisation.results import (save_score, plot_confusion_matrix, 
                                   plot_history, plot_classification_report)


# %%
# 0b. Setting all the constants

mapping = {'Sleep stage W': 0,
           'Sleep stage 1': 1,
           'Sleep stage 2': 2,
           'Sleep stage 3': 3,
           'Sleep stage 4': 3,
           'Sleep stage R': 4}
classes_mapping = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}

train_test_diff = False
preprocessed = False
train_valid = 'MASS'
train_valid_size = 60
test = 'MASS'
test_size = 0
sfreq = 256
window_size_s = 30
lr = 5e-4
n_epochs = 10
batch_size = 8

print_train_test = f'{train_valid}_{test}' if train_test_diff else train_valid
print_size = (
    f'{train_valid_size}_{test_size}' if train_test_diff
    else str(train_valid_size))
train_test = [train_valid, test] if train_test_diff else [train_valid]
print_freq = 'preprocessed' if preprocessed else sfreq

plots_path = f'plots/{print_train_test}_{print_freq}-{print_size}-batch{batch_size}_{n_epochs}epochs/'

# %%
# 1. Loading the data

try:
    os.mkdir(plots_path)
    print(f'Directory {plots_path} created\n')
except FileExistsError:
    print(f'Directory {plots_path} already exists\n')

if train_valid == 'MASS':
    train_valid_dataset = MASS_SS3(subject_ids=train_valid_size,
                                   preprocessed=preprocessed)
elif train_valid == 'SleepPhysionet':
    train_valid_dataset = SleepPhysionet(subject_ids=train_valid_size,
                                         preprocessed=preprocessed)
elif train_valid == 'Clinical':
    train_valid_dataset = ClinicalDataset(subject_ids=train_valid_size)

if train_test_diff:
    if test == 'MASS':
        test_dataset = MASS_SS3(subject_ids=test_size)
    elif test == 'SleepPhysionet':
        test_dataset = SleepPhysionet(subject_ids=test_size,
                                      preprocessed=preprocessed)
    elif test == 'Clinical':
        test_dataset = ClinicalDataset(subject_ids=test_size)
    dataset = BaseConcatDataset([train_valid_dataset, test_dataset])
else:
    dataset = train_valid_dataset

print(dataset.description)


# %%
# 2. Preprocessing

# high_cut_hz = 30
# preprocessors = [
#     # convert from volt to microvolt, directly modifying the numpy array
#     NumpyPreproc(fn=lambda x: x * 1e6),
#     # bandpass filter
#     MNEPreproc(fn='filter', l_freq=None, h_freq=high_cut_hz),
# ]
# preprocess(dataset, preprocessors)

# Extracting windows
window_size_samples = window_size_s * sfreq
windows_dataset = create_windows_from_events(
    dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples, preload=True, mapping=mapping)

# Window preprocessing
preprocess(windows_dataset, [MNEPreproc(fn=zscore)])

# %%
# 3. Making train, valid and test splits

train_set, valid_set, test_set = split_by_events(windows_dataset, train_test)
print(view_nb_windows(plots_path, train_set, valid_set, test_set))

# %%
# 4. Creating the model

cuda = torch.cuda.is_available()
set_random_seeds(seed=87, cuda=cuda)
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

n_classes = 5
n_channels = train_set[0][0].shape[0]
input_size_samples = train_set[0][0].shape[1]

# Create model
model = SleepStagerChambon2018(
    n_channels,
    sfreq=sfreq,
    n_classes=n_classes,
    input_size_s=input_size_samples / sfreq
)

# Export model to device
print(f'Using device \'{device}\'.')
model = model.to(device)

# %%
# 5. Training

train_bal_acc = EpochScoring(
    scoring='balanced_accuracy', on_train=True, name='train_bal_acc',
    lower_is_better=False)
valid_bal_acc = EpochScoring(
    scoring='balanced_accuracy', on_train=False, name='valid_bal_acc',
    lower_is_better=False)
callbacks = [('train_bal_acc', train_bal_acc),
             ('valid_bal_acc', valid_bal_acc)]

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    train_split=predefined_split(valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    batch_size=batch_size,
    callbacks=callbacks,
    device=device
)

# Model training for a specified number of epochs. `y` is None as it is already
# supplied in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)

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
