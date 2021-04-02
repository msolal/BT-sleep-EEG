# %%
# 0a. Setting up the environment

import os
import torch

from braindecode.datasets import BaseConcatDataset
from datasets.bids import BIDS
from braindecode.datautil.preprocess import (zscore, NumpyPreproc,
                                             MNEPreproc, preprocess)
from braindecode.datautil.windowers import create_windows_from_events
from datautil.split import split_by_events
from visualisation.windows import view_nb_windows
from util import set_random_seeds
from braindecode.models.sleep_stager_chambon_2018 import SleepStagerChambon2018
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring
from braindecode import EEGClassifier


# %%
# 0b. Setting all the constants

mapping = {'Sleep stage W': 0,
           'Sleep stage 1': 1,
           'Sleep stage 2': 2,
           'Sleep stage 3': 3,
           'Sleep stage 3/4': 3,
           'Sleep stage 4': 3,
           'Sleep stage R': 4}
classes_mapping = {'0': 'W', '1': 'N1', '2': 'N2', '3': 'N3 / N4', '4': 'REM'}

datasets = ['MASS']
derivatives = ['9ch']
sizes = [48]

sfreq = 100
window_size_s = 30
lr = 5e-4           # lr can be 5e-4 or 1e-3
n_epochs = 10
batch_size = 8

desc = f'{datasets[0]}_{derivatives[0]}-{sizes[0]}-lr{lr}_batch{batch_size}_{n_epochs}epochs'
plots_path = f'plots/29-03/{desc}/'
models_path = f'/storage/store2/work/msolal/trained_models/{desc}'

# %%
# 1. Loading the data

try:
    os.mkdir(plots_path)
    print(f'Directory {plots_path} created\n')
except FileExistsError:
    print(f'Directory {plots_path} already exists\n')

dataset = BIDS(dataset=datasets[0],
               derivatives=derivatives[0],
               subject_ids=sizes[0])
print(dataset.description)

# %%
# 2. Preprocessing

# preprocess(dataset, [NumpyPreproc(fn=lambda x: x * 1e6)])

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

train_set, valid_set, _ = split_by_events(windows_dataset, None)
print(view_nb_windows(plots_path, train_set, valid_set, None))

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
torch.save(clf, models_path)
print(f'Model saved to {models_path}')
