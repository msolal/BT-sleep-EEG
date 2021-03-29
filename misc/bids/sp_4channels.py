import os
import mne
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids
from tempfile import NamedTemporaryFile

bids_root = '/storage/store2/derivatives/Physionet/preprocessed/'
preproc_bids_root = '/storage/store2/derivatives/Physionet/4channels-eeg_eog_emg/'
datatype = 'eeg'

all_sub = pd.read_csv(bids_root + 'participants.tsv',
                      delimiter='\t', skiprows=1,
                      names=['participant_id', 'age', 'sex', 'hand'],
                      engine='python')['participant_id'].transform(
                        lambda x: x[4:]).tolist()
bids_paths = [BIDSPath(subject=subject, root=bids_root,
                       datatype=datatype) for subject in all_sub]


def preprocess_and_save(bids_path):
    raw = read_raw_bids(bids_path=bids_path)
    raw.pick_channels(['Fpz-Cz', 'Pz-Oz', 'EOG horizontal', 'EMG submental'])

    # Write new BIDS

    # Set output path
    preproc_bids_path = bids_path.copy().update(root=preproc_bids_root)

    # Work around a limitation of MNE-BIDS: It won't allow us to save the
    # pre-loaded raw data to BIDS directly; so we're going to write the
    # data to a temporary file, which we are then going to pass to MNE-BIDS
    # for storage.
    # Use `_raw.fif` suffix to avoid MNE warnings.
    with NamedTemporaryFile(suffix='_raw.fif') as f:
        fname = f.name
        raw.save(fname, overwrite=True)
        raw = mne.io.read_raw_fif(fname, preload=False)
        write_raw_bids(raw, preproc_bids_path, overwrite=True, verbose=False)


for bids_path in bids_paths:
    sessions = os.listdir(f'{bids_path.root}/sub-{bids_path.subject}')
    for session in sessions:
        bids_path.update(session=session[4:])
        preprocess_and_save(bids_path)
