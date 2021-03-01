import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids
from braindecode.datasets.base import BaseDataset, BaseConcatDataset


bids_root = '/storage/store2/data/mass-bids/SS3/'
preproc_bids_root = bids_root + 'derivatives/preprocessed/'
channels = ['C3', 'C4', 'Cz', 'ECG I', 'EMG Chin1', 'EMG Chin2', 'EMG Chin3',
            'EOG Left Horiz', 'EOG Right Horiz', 'F3', 'F4', 'F7', 'F8', 'Fp1',
            'Fp2', 'Fz', 'O1', 'O2', 'Oz', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5',
            'T6']


class MASS_SS3(BaseConcatDataset):
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
    def __init__(self, subject_ids=None, preload=True,
                 load_eeg_only=True, crop_wake_mins=30, preprocessed=True):

        if preprocessed:
            bids_root = preproc_bids_root
        
        all_sub = pd.read_csv(bids_root + 'participants.tsv',
                              delimiter='\t', skiprows=1,
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
                  crop_wake_mins=30):

        raw = read_raw_bids(bids_path=bids_path)
        annots = raw.annotations
        raw.pick_channels(channels)
        if load_eeg_only:
            raw.pick_types(eeg=True)

        if crop_wake_mins > 0:
            # Find first and last sleep stages
            mask = [
                x[-1] in ['1', '2', '3', '4', 'R']
                for x in annots.description]
            sleep_event_inds = np.where(mask)[0]

            # Crop raw
            tmin = annots[int(sleep_event_inds[0])]['onset'] - \
                crop_wake_mins * 60
            tmin = max(raw.times[0], tmin)
            tmax = annots[int(sleep_event_inds[-1])]['onset'] + \
                crop_wake_mins * 60
            tmax = min(tmax, raw.times[-1])
            raw.crop(tmin=tmin, tmax=tmax)

        basename = bids_path.basename
        sub_nb = basename[4:]
        desc = pd.Series({'subject': sub_nb, 'dataset': 'MASS'}, name='MASS')

        return raw, desc
