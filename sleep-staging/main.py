# %%
# 0. Setting up the environment

import torch
import numpy as np


from datasets.mass_bids import MASS_SS3
from datasets.sleep_physionet import SleepPhysionet
from datasets.clinical import ClinicalDataset
from datautil.preprocess import zscore
from datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess
from datautil.windowers import create_windows_from_events
from datautil.split import train_valid_test_split
from util import set_random_seeds
from models.sleep_stager_chambon import SleepStagerChambon2018
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring
from braindecode import EEGClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from visualisation.visualisation import plot_confusion_matrix, plot_history


# %%
# 1. Loading the data

# dataset = SleepPhysionet(subject_ids=list(range(30)),
#                          recording_ids=[1],
#                          crop_wake_mins=30)
# dataset = MASS_SS3(subject_ids=None, crop_wake_mins=0)
dataset = ClinicalDataset(subject_ids=10, crop_wake_mins=0)

# dataset = [train_valid_ds, test_ds]

# %%
# 2. Preprocessing

high_cut_hz = 30

preprocessors = [
    # convert from volt to microvolt, directly modifying the numpy array
    NumpyPreproc(fn=lambda x: x * 1e6),
    # bandpass filter
    MNEPreproc(fn='filter', l_freq=None, h_freq=high_cut_hz),
]

# Transform the data
preprocess(dataset, preprocessors)

# preprocess(dataset[0], preprocessors)
# preprocess(dataset[1], preprocessors)

# Extracting windows

mapping = {  # We merge stages 3 and 4 following AASM standards.
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}

window_size_s = 30
sfreq = int(dataset.datasets[0].raw.info['sfreq'])
window_size_samples = window_size_s * sfreq

windows_dataset = create_windows_from_events(
    dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples, preload=True, mapping=mapping)

# sfreq = [int(dataset[0].datasets[0].raw.info['sfreq']),
#          int(dataset[1].datasets[0].raw.info['sfreq'])]
# window_size_samples = [window_size_s * sfreq[0], window_size_s * sfreq[1]]

# windows_dataset = [create_windows_from_events(
#                    dataset[0], trial_start_offset_samples=0,
#                    trial_stop_offset_samples=0,
#                    window_size_samples=window_size_samples[0],
#                    window_stride_samples=window_size_samples[0],
#                    preload=True, mapping=mapping),
#                   create_windows_from_events(dataset[1],
#                    trial_start_offset_samples=0,
#                    trial_stop_offset_samples=0,
#                    window_size_samples=window_size_samples[1],
#                    window_stride_samples=window_size_samples[1],
#                    preload=True, mapping=mapping)]


# Window preprocessing
preprocess(windows_dataset, [MNEPreproc(fn=zscore)])

# preprocess(windows_dataset[0], [MNEPreproc(fn=zscore)])
# preprocess(windows_dataset[1], [MNEPreproc(fn=zscore)])


# %%
# 3. Making train, valid and test splits
train_set, valid_set, test_set = train_valid_test_split(windows_dataset,
                                                        0.6, 0.2, 0.2)
# train_set, valid_set = train_valid_test_split(windows_dataset[0], 0.6, 0.2)
# test_set = train_valid_test_split(windows_dataset[1], 0, 0, 1)

print('Number of windows in each set:')
print(f'Training: {train_set.datasets[0].windows}')
print(f'Validation: {valid_set.datasets[0].windows}')
print(f'Test: {test_set.datasets[0].windows}')

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
    sfreq,
    n_classes=n_classes,
    input_size_s=input_size_samples / sfreq
)

# Export model to device
print(f'Using device \'{device}\'.')
model = model.to(device)

# %%
# 5. Training

lr = 5e-4
n_epochs = 5
batch_size = 16

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
print(f'Test balanced accuracy: {test_bal_acc:0.3f}')
print(f'Test Cohen\'s kappa: {test_kappa:0.3f}')

plot_history(clf)

# Finally, we also display the confusion matrix and classification report
classes_mapping = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}
conf_mat = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(conf_mat, classes_mapping)

print(classification_report(y_true, y_pred))
