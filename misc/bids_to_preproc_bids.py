import mne
import pandas as pd
from os.path import basename
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids 
from tempfile import NamedTemporaryFile

<<<<<<< HEAD
bids_root = '/storage/store2/data/mass-bids/SS3/'
preproc_bids_root = '/storage/store2/data/mass-bids/SS3/derivatives/preprocessed/'
=======
bids_root = '/storage/store2/data/SleepPhysionet-bids/'
preproc_bids_root = bids_root + 'derivatives/preprocessed/'
>>>>>>> f79dcda05819af89709b4fd4a38be556780e872d
datatype = 'eeg'

all_sub = pd.read_csv(bids_root + 'participants.tsv',
                    delimiter='\t', skiprows=1,
                    names=['participant_id', 'age', 'sex', 'hand'],
                    engine='python')['participant_id'].transform(
                        lambda x: x[4:]).tolist()
bids_paths = [BIDSPath(subject=subject, root=bids_root,
                       datatype=datatype) for subject in all_sub]

def preprocess_and_save(bids_path, l_freq, h_freq, sfreq):
    raw = read_raw_bids(bids_path=bids_path)
    # Preprocessing
    raw.load_data()
<<<<<<< HEAD
    raw.resample(sfreq=sfreq, npad='auto')
=======
    # raw.resample(sfreq=sfreq, npad='auto')
>>>>>>> f79dcda05819af89709b4fd4a38be556780e872d
    raw.filter(l_freq=l_freq, h_freq=h_freq)

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
        write_raw_bids(raw, preproc_bids_path, overwrite=True)

l_freq, h_freq = None, 30
sfreq = 100
for bids_path in bids_paths:
    preprocess_and_save(bids_path, l_freq, h_freq, sfreq)
