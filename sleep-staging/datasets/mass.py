import os
import mne
import numpy as np
import pandas as pd

from mne.datasets.sleep_physionet.age import fetch_data

from braindecode.datasets.base import BaseDataset, BaseConcatDataset

class MASS_SS3(BaseConcatDataset):
    """MASS SS3 dataset.
    Contains overnight recordings from 62 healthy subjects.

    Parameters
    ----------
    subject_ids: list(int) | None
        list of int of subject(s) to be loaded. If None, load all available
        subjects.
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
    
        not_available = {36, 40, 43, 45}
        if subject_ids is None:
            subject_ids = [x for x in range(1, 47) if x not in not_available]
        if len(set(subject_ids).intersection(not_available)) != 0:
            subject_ids = subject_ids - not_available
            print('Warning: subjects 36, 40, 43 and 45 are not available.')

        rec_nb = ['01-03-000{}'.format(i) if i < 10 else '01-03-00{}'.format(i) for i in subject_ids]

        paths = [(f'/storage/store/data/mass/SS3/{i} PSG.edf', f'/storage/store/data/mass/SS3/annotations/{i} Annotations.edf') for i in rec_nb]

        all_base_ds = list()
        for p in paths:
            raw, desc = self._load_raw(
                p[0], p[1], preload=preload, load_eeg_only=load_eeg_only,
                crop_wake_mins=crop_wake_mins)
            base_ds = BaseDataset(raw, desc)
            all_base_ds.append(base_ds)
        super().__init__(all_base_ds)

    @staticmethod
    def _load_raw(raw_fname, annot_fname, preload, load_eeg_only=True,
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

        raw = mne.io.read_raw_edf(raw_fname, preload=preload, exclude=exclude)
        annots = mne.read_annotations(annot_fname)
        raw.set_annotations(annots, emit_warning=False)

        if crop_wake_mins > 0:
            # Find first and last sleep stages
            mask = [
                x[-1] in ['1', '2', '3', '4', 'R'] for x in annots.description]
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

        # record_nb = raw_fname[29:39]
        # assert(record_nb == annot_fname[41:51])
        # # print('record_nb = annot_nb = ', record_nb)
        basename = os.path.basename(raw_fname)
        subj_nb = basename[:10]
        sess_nb = basename[:10]
        desc = pd.Series({'subject': subj_nb, 'recording': sess_nb}, name='')

        return raw, desc