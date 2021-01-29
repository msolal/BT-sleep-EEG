import os
import mne
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from sklearn.model_selection import LeavePGroupsOut
from mne.datasets.sleep_physionet.age import fetch_data


def get_mass_paths(nb_subjects=None):
    """ Get paths for the MASS dataset on drago.

    Parameters
    ----------
    nb_subjects : int or None
        number of subjects wanted.

    Returns
    -------
    raw_fnames : list
        list of paths for the .edf files containing the raw data.
    annot_fnames : list
        list of paths for the .edf files containing the annotations.
    """
    if nb_subjects is None:
        subjects = [x for x in list(range(1, 47)) if x not in {36, 40, 43, 45}]
    else:
        subjects = [x for x in list(range(1, 47)) if x not in
                    {36, 40, 43, 45}][:nb_subjects]
    rec_nb = []
    for i in subjects:
        if i < 10:
            rec_nb.append('01-03-000{}'.format(i))
        else:
            rec_nb.append('01-03-00{}'.format(i))
    raw_fnames = ['/storage/store/data/mass/SS3/{} PSG.edf'.format(i)
                  for i in rec_nb]
    annot_fnames = ["/storage/store/data/mass/SS3/annotations/{} Annotations.edf".
                    format(i) for i in rec_nb]
    return raw_fnames, annot_fnames


def load_mass_raw(raw_fname, annot_fname, load_eeg_only=True,
                  crop_wake_mins=0):
    """Load a recording into mne raw given file paths.

    Parameters
    ----------
    raw_fname : str
        path to the .edf file containing the raw data.
    annot_fname : str
        path to the .edf file containing the annotations.
    load_eeg_only : bool
        If True, only keep EEG channels and discard other modalities
        (speeds up loading).
    crop_wake_mins : float
        Number of minutes of wake events before and after sleep events.

    Returns
    -------
    mne.io.Raw :
        Raw object containing the EEG and annotations.
    """
    mapping = {'ECG I': 'ecg',
               'EOG Right Horiz': 'eog',
               'EOG Left Horiz': 'eog',
               'EMG Chin1': 'emg',
               'EMG Chin2': 'emg',
               'EMG Chin3': 'emg'}
    exclude = mapping.keys() if load_eeg_only else ()

    record_nb = raw_fname[29:39]
    assert(record_nb == annot_fname[41:51])
    # print('record_nb = annot_nb = ', record_nb)

    raw = mne.io.read_raw_edf(raw_fname, exclude=exclude)
    annots = mne.read_annotations(annot_fname)
    raw.set_annotations(annots, emit_warning=False)
    if not load_eeg_only:
        raw.set_channel_types(mapping)

    if crop_wake_mins > 0:  # Cut start and end Wake periods
        # Find first and last sleep stages
        mask = [x[-1] in ['1', '2', '3', '4', 'R']
                for x in annots.description]
        sleep_event_inds = np.where(mask)[0]

        # Crop raw
        tmin = annots[int(sleep_event_inds[0])]['onset'] - \
            crop_wake_mins * 60
        tmax = annots[int(sleep_event_inds[-1])]['onset'] + \
            crop_wake_mins * 60
        raw.crop(tmin=tmin, tmax=tmax)

    # Rename EEG channels
    ch_names = {i: i.replace('EEG ', '')
                for i in raw.ch_names if 'EEG' in i}
    mne.rename_channels(raw.info, ch_names)

    # works well with MASS, might require adaptation
    # Save subject and recording information in raw.info
    raw.info['subject_info'] = {'id': record_nb, 'rec_id': record_nb}

    return raw


def get_mass_dataset(nb_subjects=None, plot_idx=None):
    """Get MASS dataset.

    Parameters
    ----------
    nb_subjects : int or None
        number of subjects wanted.
    plot_idx : int or None
        record to plot.

    Returns
    -------
    Dataset :
        torch dataset.
    """
    # Get paths to .edf files
    raw_fnames, annot_fnames = get_mass_paths(nb_subjects)
    # Load recordings
    raws = [load_mass_raw(raw_fname, annot_fname) for (raw_fname, annot_fname)
            in zip(raw_fnames, annot_fnames)]
    sfreq = raws[0].info['sfreq']
    n_channels = raws[0].info['nchan']
    # Plot a recording as a sanity check
    if plot_idx is not None:
        raws[plot_idx].plot().savefig(plots_path + '1-rawplot')
    # Filtering
    filtering(raws)
    # Plot the power spectrum of a recording as sanity check
    if plot_idx is not None:
        raws[plot_idx].plot_psd().savefig(plots_path + '2-psd')
    # Apply windowing and move to pytorch dataset
    all_datasets = [EpochsDataset(*extract_epochs(raw),
                                  subj_nb=raw.info['subject_info']['id'],
                                  rec_nb=raw.info['subject_info']['rec_id'],
                                  transform=scale)
                    for raw in raws]
    # Concatenate into a single dataset
    dataset = ConcatDataset(all_datasets)
    return dataset, sfreq, n_channels


def get_physionet_paths(nb_subjects, recordings=[1]):
    """ Get paths for the sleep physionet dataset.

    Parameters
    ----------
    nb_subjects : int
        number of subjects wanted.
    recording : list
        list of recordings wanted.

    Returns
    -------
    paths : tuple
        tuple of paths for raw filenames and annotations filenames.
    """
    subjects = range(nb_subjects)
    return fetch_data(subjects=subjects,
                      recording=recordings,
                      on_missing='warning')


def load_physionet_raw(raw_fname, annot_fname, load_eeg_only=True,
                       crop_wake_mins=0):
    """Load a recording into mne raw given file paths.

    Parameters
    ----------
    raw_fname : str
        path to the .edf file containing the raw data.
    annot_fname : str
        path to the .edf file containing the annotations.
    load_eeg_only : bool
        If True, only keep EEG channels and discard other modalities
        (speeds up loading).
    crop_wake_mins : float
        Number of minutes of wake events before and after sleep events.

    Returns
    -------
    mne.io.Raw :
        Raw object containing the EEG and annotations.
    """
    mapping = {'EOG horizontal': 'eog',
               'Resp oro-nasal': 'misc',
               'EMG submental': 'misc',
               'Temp rectal': 'misc',
               'Event marker': 'misc'}
    exclude = mapping.keys() if load_eeg_only else ()

    raw = mne.io.read_raw_edf(raw_fname, exclude=exclude)
    annots = mne.read_annotations(annot_fname)
    raw.set_annotations(annots, emit_warning=False)
    if not load_eeg_only:
        raw.set_channel_types(mapping)

    if crop_wake_mins > 0:  # Cut start and end Wake periods
        # Find first and last sleep stages
        mask = [x[-1] in ['1', '2', '3', '4', 'R']
                for x in annots.description]
        sleep_event_inds = np.where(mask)[0]

        # Crop raw
        tmin = annots[int(sleep_event_inds[0])]['onset'] - \
            crop_wake_mins * 60
        tmax = annots[int(sleep_event_inds[-1])]['onset'] + \
            crop_wake_mins * 60
        raw.crop(tmin=tmin, tmax=tmax)

    # Rename EEG channels
    ch_names = {i: i.replace('EEG ', '')
                for i in raw.ch_names if 'EEG' in i}
    mne.rename_channels(raw.info, ch_names)

    basename = os.path.basename(raw_fname)
    subj_nb, rec_nb = int(basename[3:5]), int(basename[5])
    raw.info['subject_info'] = {'id': subj_nb, 'rec_id': rec_nb}

    return raw


def get_physionet_dataset(nb_subjects=None, plot_idx=None):
    """Get Sleep Physionet dataset.

    Parameters
    ----------
    nb_subjects : int or None
        number of subjects wanted.
    plot_idx : int or None
        record to plot.

    Returns
    -------
    Dataset :
        torch dataset.
    """
    # Get paths to .edf files
    raw_fnames, annot_fnames = get_physionet_paths(nb_subjects)
    # Load recordings
    raws = [load_physionet_raw(raw_fname, annot_fname)
            for (raw_fname, annot_fname) in zip(raw_fnames, annot_fnames)]
    sfreq = raws[0].info['sfreq']
    n_channels = raws[0].info['nchan']
    # Plot a recording as a sanity check
    if plot_idx is not None:
        raws[plot_idx].plot().savefig(plots_path + '1-rawplot')
    # Filtering
    filtering(raws)
    # Plot the power spectrum of a recording as sanity check
    if plot_idx is not None:
        raws[plot_idx].plot_psd().savefig(plots_path + '2-psd')
    # Apply windowing and move to pytorch dataset
    all_datasets = [EpochsDataset(*extract_epochs(raw),
                                  subj_nb=raw.info['subject_info']['id'],
                                  rec_nb=raw.info['subject_info']['rec_id'],
                                  transform=scale)
                    for raw in raws]
    # Concatenate into a single dataset
    dataset = ConcatDataset(all_datasets)
    return dataset, sfreq, n_channels


def filtering(raws, l_freq=None, h_freq=30):
    """Filtering.

    Parameters
    ----------
    raws : list of mne.io.Raw
        list of raw.
    l_freq : int
        low frequency, can be set to None.
    h_freq : int
        high frequency, can be set to None.

    Returns
    -------
    Happens in place, no return
    """
    for raw in raws:
        raw.load_data().filter(l_freq, h_freq)


def extract_epochs(raw, chunk_duration=30.):
    """Extract non-overlapping epochs from raw data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object to be windowed.
    chunk_duration : float
        Length of a window.

    Returns
    -------
    np.ndarray
        Epoched data, of shape (n_epochs, n_channels, n_times).
    np.ndarray
        Event identifiers for each epoch, shape (n_epochs,).
    """
    annotation_desc_2_event_id = {
        'Sleep stage W': 1,
        'Sleep stage 1': 2,
        'Sleep stage 2': 3,
        'Sleep stage 3': 4,
        'Sleep stage 4': 4,
        'Sleep stage R': 5}

    events, _ = mne.events_from_annotations(
        raw, event_id=annotation_desc_2_event_id,
        chunk_duration=chunk_duration)

    # create a new event_id that unifies stages 3 and 4
    event_id = {
        'Sleep stage W': 1,
        'Sleep stage 1': 2,
        'Sleep stage 2': 3,
        'Sleep stage 3/4': 4,
        'Sleep stage R': 5}

    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
    picks = mne.pick_types(raw.info, eeg=True, eog=True)
    epochs = mne.Epochs(raw=raw, events=events, picks=picks, preload=True,
                        event_id=event_id, tmin=0., tmax=tmax, baseline=None)

    return epochs.get_data(), epochs.events[:, 2] - 1


class EpochsDataset(Dataset):
    """Class to expose an MNE Epochs object as PyTorch dataset.

    Parameters
    ----------
    epochs_data : np.ndarray
        The epochs data, shape (n_epochs, n_channels, n_times).
    epochs_labels : np.ndarray
        The epochs labels, shape (n_epochs,)
    subj_nb: None | int
        Subject number.
    rec_nb: None | int
        Recording number.
    transform : callable | None
        The function to eventually apply to each epoch
        for preprocessing (e.g. scaling). Defaults to None.
    """

    def __init__(self, epochs_data, epochs_labels, subj_nb=None,
                 rec_nb=None, transform=None):
        assert len(epochs_data) == len(epochs_labels)
        self.epochs_data = epochs_data
        self.epochs_labels = epochs_labels
        self.subj_nb = subj_nb
        self.rec_nb = rec_nb
        self.transform = transform

    def __len__(self):
        return len(self.epochs_labels)

    def __getitem__(self, idx):
        X, y = self.epochs_data[idx], self.epochs_labels[idx]
        if self.transform is not None:
            X = self.transform(X)
        X = torch.as_tensor(X[None, ...])
        return X, y


def scale(X):
    """Standard scaling of data along the last dimention.

    Parameters
    ----------
    X : array, shape (n_channels, n_times)
        The input signals.

    Returns
    -------
    X_t : array, shape (n_channels, n_times)
        The scaled signals.
    """
    X -= np.mean(X, axis=1, keepdims=True)
    return X / np.std(X, axis=1, keepdims=True)


def pick_recordings(dataset, pick_idx):
    """Pick recordings using subject and recording numbers.

    Parameters
    ----------
    dataset : ConcatDataset
        The dataset to pick recordings from.
    pick_idx : list
        list of recordings in testing dataset.

    Returns
    -------
    ConcatDataset
        The picked recordings.
    ConcatDataset | None
        The remaining recordings. None if all recordings from
        `dataset` were picked.
    """
    remaining_idx = np.setdiff1d(
        range(len(dataset.datasets)), pick_idx)

    pick_ds = ConcatDataset([dataset.datasets[i] for i in pick_idx])
    if len(remaining_idx) > 0:
        remaining_ds = ConcatDataset(
            [dataset.datasets[i] for i in remaining_idx])
    else:
        remaining_ds = None

    return pick_ds, remaining_ds


def train_test_split(dataset, n_groups, split_by='subj_nb'):
    """Split dataset into train and test keeping n_groups out in test.

    Parameters
    ----------
    dataset : ConcatDataset
        The dataset to split.
    n_groups : int
        The number of groups to leave out.
    split_by : 'subj_nb' | 'rec_nb'
        Property to use to split dataset.

    Returns
    -------
    ConcatDataset
        The training data.
    ConcatDataset
        The testing data.
    """
    groups = [getattr(ds, split_by) for ds in dataset.datasets]
    train_idx, test_idx = next(
        LeavePGroupsOut(n_groups).split(X=groups, groups=groups))

    train_ds = ConcatDataset([dataset.datasets[i] for i in train_idx])
    test_ds = ConcatDataset([dataset.datasets[i] for i in test_idx])

    return train_ds, test_ds
