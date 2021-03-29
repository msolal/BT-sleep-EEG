import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids
from tempfile import NamedTemporaryFile

bids_root = '/storage/store2/derivatives/MASS/SS3/100Hz/'
preproc_bids_root = '/storage/store2/derivatives/MASS/SS3/4channels-eeg_eog_emg/'
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
    raw.pick_channels(['Fp1', 'Fp2', 'Cz', 'Pz', 'Oz',
                       'EOG Right Horiz', 'EOG Left Horiz',
                       'EMG Chin1'], ordered=True)
    sfreq = raw.info['sfreq']
    linefreq = raw.info['line_freq']

    data, times = raw[:]
    fpzcz = (data[0] + data[1])/2 - data[2]
    info_fpzcz = mne.create_info(['Fpz-Cz'], sfreq=sfreq, ch_types=datatype)
    info_fpzcz['line_freq'] = linefreq
    raw_fpzcz = mne.io.RawArray(fpzcz[np.newaxis, :], info_fpzcz)

    pzoz = data[3] - data[4]
    info_pzoz = mne.create_info(['Pz-Oz'], sfreq=sfreq, ch_types=datatype)
    info_pzoz['line_freq'] = linefreq
    raw_pzoz = mne.io.RawArray(pzoz[np.newaxis, :], info_pzoz)

    eog = (data[5] + data[6])/2
    info_eog = mne.create_info(['EOG Horiz avg'], sfreq=sfreq, ch_types='eog')
    info_eog['line_freq'] = linefreq
    raw_eog = mne.io.RawArray(eog[np.newaxis, :], info_eog)

    emg = data[7]
    info_emg = mne.create_info(['EMG Chin1'], sfreq=sfreq, ch_types='emg')
    info_emg['line_freq'] = linefreq
    raw_emg = mne.io.RawArray(emg[np.newaxis, :], info_emg)

    raw_final = raw_fpzcz.copy().add_channels([raw_pzoz, raw_eog, raw_emg])
    raw_final.info['meas_date'] = raw.info['meas_date']
    raw_final.set_annotations(raw.annotations)

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
        raw_final.save(fname, overwrite=True)
        raw = mne.io.read_raw_fif(fname, preload=False)
        write_raw_bids(raw, preproc_bids_path, overwrite=True)


for bids_path in bids_paths[42:]:
    preprocess_and_save(bids_path)
