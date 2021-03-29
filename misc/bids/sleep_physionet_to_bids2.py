from os.path import basename
from tempfile import NamedTemporaryFile

import argparse
import pandas as pd
from tqdm import tqdm
import mne
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids

from braindecode.datasets.sleep_physionet import SleepPhysionet
from mne.datasets.sleep_physionet.age import fetch_data


def preprocess_and_save(
    raw_path_pair,
    preproc_bids_path,
    l_freq,
    h_freq,
    sfreq,
    to_microvolt=True,
    channels_to_keep=None,
    remove_ch_ref=False,
    crop_wake_mins=30,
    load_eeg_only=False
):
    raw, desc = SleepPhysionet._load_raw(
                    raw_path_pair[0], raw_path_pair[1], preload=True,
                    load_eeg_only=load_eeg_only, crop_wake_mins=crop_wake_mins)
    # Preprocessing
    if to_microvolt:
        raw.apply_function(lambda x: x * 1e6, channel_wise=False, verbose=False)
    if channels_to_keep is not None:
        raw.pick_channels(channels_to_keep)
    if sfreq != raw.info['sfreq']:
        raw.resample(sfreq=sfreq, npad='auto', verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    if remove_ch_ref:
        mapping = {name: name.split('-')[0] for name in raw.info['ch_names']}
        mne.rename_channels(raw.info, mapping)

    # Write new BIDS

    # Work around a limitation of MNE-BIDS: It won't allow us to save the
    # pre-loaded raw data to BIDS directly; so we're going to write the
    # data to a temporary file, which we are then going to pass to MNE-BIDS
    # for storage.
    # Use `_raw.fif` suffix to avoid MNE warnings.
    with NamedTemporaryFile(suffix='_raw.fif') as f:
        fname = f.name
        raw.save(fname, overwrite=True, verbose=False)
        raw = mne.io.read_raw_fif(fname, preload=False, verbose=False)
        write_raw_bids(raw, preproc_bids_path, overwrite=True, verbose=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess and save physionet sleep edf into bids.'
    )

    parser.add_argument(
        'destination',
        type=str,
        help='Directory to use as root for the bids files created.'
    )

    args = parser.parse_args()

    preproc_bids_root = args.destination

    l_freq, h_freq = None, 30
    sfreq = 100
    for subject_id in tqdm(range(49, 83)):
        for recording_id in [1, 2]:
            try:
                path_pair = fetch_data(subjects=[subject_id], recording=[recording_id], on_missing='warn', verbose=False)[0]
                subject = basename(path_pair[0]).split('-')[0][:5]
                session = basename(path_pair[1]).split('-')[0][5:]
                preproc_bids_path = BIDSPath(
                        subject=subject,
                        session=session,
                        root=preproc_bids_root,
                        suffix='eeg',
                        datatype='eeg'
                )
                preprocess_and_save(path_pair, preproc_bids_path, l_freq, h_freq, sfreq)
            except (IndexError, ValueError):
                print('error')