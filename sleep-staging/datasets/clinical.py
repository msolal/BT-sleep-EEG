import os
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from braindecode.datasets.base import BaseDataset, BaseConcatDataset


class ClinicalDataset(BaseConcatDataset):
    """MASS SS3 dataset.
    Contains overnight recordings from 62 healthy subjects.

    Parameters
    ----------
    subject_ids: list(str) | int | None
        list of str of subject(s) to be loaded. 
        If None, load all available subjects.
        If int, load first subject_ids subjects. 
    preload: bool
        If True, preload the data of the Raw objects.
    load_eeg_only: bool
        If True, only load the EEG channels and discard the others (EOG, EMG,
        temperature, respiration) to avoid resampling the other signals.
    crop_wake_mins: float
        Number of minutes of wake time to keep before the first sleep event
        and after the last sleep event. Used to reduce the imbalance in this
        dataset. Default of 30 mins.
    """
    def __init__(self, subject_ids=None, preload=False,
                 load_eeg_only=True, crop_wake_mins=30):

        all_sub = pd.read_csv('BIDS/participants.tsv', delimiter='\t',
                              skiprows=1,
                              names=['participant_id', 'age', 'sex', 'hand'],
                              engine='python')['participant_id'].transform(
                                  lambda x: x[4:]).tolist()

        if subject_ids is None:
            subject_ids = all_sub
        elif type(subject_ids) == int:
            subject_ids = all_sub[:subject_ids]
        elif len(set(subject_ids).intersection(all_sub)) != len(subject_ids):
            subject_ids = [x for x in subject_ids if x in all_sub]
            print('Warning: selected subject which doesn\'t exist')

        bids_root = '/media/pallanca/datapartition/maelys/data/BIDS'
        if load_eeg_only:
            datatype = 'eeg'
        bids_paths = [BIDSPath(subject=subject, datatype=datatype,
                               root=bids_root) for subject in subject_ids]

        all_base_ds = list()
        for path in bids_paths:
            raw, desc = self._load_raw(path, preload=preload,
                                       load_eeg_only=load_eeg_only,
                                       crop_wake_mins=crop_wake_mins)
            base_ds = BaseDataset(raw, desc)
            all_base_ds.append(base_ds)
        super().__init__(all_base_ds)

    @staticmethod
    def _load_raw(bids_path, preload, load_eeg_only=True,
                  crop_wake_mins=False):
        ch_mapping = {
            'ECG I': 'ecg',
            'EOG Right Horiz': 'eog',
            'EOG Left Horiz': 'eog',
            'EMG Chin1': 'emg',
            'EMG Chin2': 'emg',
            'EMG Chin3': 'emg'
        }
        exclude = ch_mapping.keys() if load_eeg_only else ()

        raw = read_raw_bids(bids_path=bids_path,
                            extra_params=dict(exclude=exclude),
                            verbose=False)
        annots = raw.annotations

        if crop_wake_mins > 0:
            # Find first and last sleep stages
            mask = [
                x[-1] in ['1', '2', '3', '4', 'R'] 
                for x in annots.description]
            sleep_event_inds = np.where(mask)[0]

            # Crop raw
            tmin = annots[int(sleep_event_inds[0])]['onset'] - \
                crop_wake_mins * 60
            tmax = annots[int(sleep_event_inds[-1])]['onset'] + \
                crop_wake_mins * 60
            raw.crop(tmin=tmin, tmax=tmax)

        # Rename EEG channels
        ch_names = {
            i: i.replace('EEG ', '') for i in raw.ch_names if 'EEG' in i}
        mne.rename_channels(raw.info, ch_names)

        if not load_eeg_only:
            raw.set_channel_types(ch_mapping)

        basename = bids_path.basename
        sub_nb = basename[4:8]
        ses_nb = basename[13:19]
        desc = pd.Series({'subject': sub_nb, 'recording': ses_nb}, name='')

        return raw, desc
