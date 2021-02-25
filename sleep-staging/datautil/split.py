import random
import numpy as np
from braindecode.datasets import BaseConcatDataset

def train_valid_test_split(windows_dataset, shuffle=True, train=0.6, valid=0.2, test=0.2):
    assert(train+valid+test == 1)
    nb_ds = len(windows_dataset.datasets)
    train_size = int(nb_ds*train)
    valid_size = int(nb_ds*valid)
    if not shuffle:
        train_idx = list(range(train_size))
        valid_idx = list(range(train_size, train_size+valid_size))
        test_idx = list(range(train_size+valid_size, nb_ds))
    else:
        ds_idx = list(range(nb_ds))
        random.shuffle(ds_idx)
        train_idx = ds_idx[:train_size]
        valid_idx = ds_idx[train_size: train_size+valid_size]
        test_idx = ds_idx[train_size+valid_size: nb_ds]
    if len(train_idx) == 0:
        if len(valid_idx) == 0:
            splitted = windows_dataset.split(by=[test_idx])
            train_set = None
            valid_set = None
            test_set = splitted['0']
        else:
            splitted = windows_dataset.split(by=[valid_idx, test_idx])
            train_set = None
            valid_set = splitted['0']
            test_set = splitted['1']
    elif len(valid_idx) == 0:
        splitted = windows_dataset.split(by=[train_idx, test_idx])
        train_set = splitted['0']
        valid_set = None
        test_set = splitted['1']
    elif len(test_idx) == 0:
        splitted = windows_dataset.split(by=[train_idx, valid_idx])
        train_set = splitted['0']
        valid_set = splitted['1']
        test_set = None
    else:
        splitted = windows_dataset.split(by=[train_idx, valid_idx, test_idx])
        train_set = splitted['0']
        valid_set = splitted['1']
        test_set = splitted['2']
    return train_set, valid_set, test_set


def split_by_events(windows_dataset, train_test=None):
    if len(train_test) != 1:
        splitted_ds = windows_dataset.split(by='dataset')
        train_valid = splitted_ds[train_test[0]]
        n_events_per_subject = [len(ds.windows.events) for ds in train_valid.datasets]
        index_subjects = list(np.argsort(n_events_per_subject))
        folds = [index_subjects[i::4] for i in range(4)]
        splitted_folds = train_valid.split(by=folds)
        train_set = BaseConcatDataset([splitted_folds['0'], splitted_folds['1'], splitted_folds['2']])
        valid_set = BaseConcatDataset([splitted_folds['3']])
        test_set = splitted_ds[train_test[1]]
    else:
        n_events_per_subject = [len(ds.windows.events) for ds in windows_dataset.datasets]
        index_subjects = list(np.argsort(n_events_per_subject))
        folds = [index_subjects[i::5] for i in range(5)]
        splitted = windows_dataset.split(by=folds)
        train_set = BaseConcatDataset([splitted['0'], splitted['1'], splitted['2']])
        valid_set = BaseConcatDataset([splitted['3']])
        test_set = BaseConcatDataset([splitted['4']])
    return train_set, valid_set, test_set