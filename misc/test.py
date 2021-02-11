# %%
from mne_bids import BIDSPath, read_raw_bids

bids_root = '/storage/store2/data/mass-bids/SS3'
subject1 = '030001'
bids_path = BIDSPath(subject=subject1, root=bids_root)
channels = ['C3', 'C4', 'Cz', 'ECG I', 'EMG Chin1', 'EMG Chin2', 'EMG Chin3',
            'EOG Left Horiz', 'EOG Right Horiz', 'F3', 'F4', 'F7', 'F8', 'Fp1',
            'Fp2', 'Fz', 'O1', 'O2', 'Oz', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5',
            'T6']

# %%
raw = read_raw_bids(bids_path=bids_path)
print(raw.info['ch_names'])
raw.pick_channels(channels)
print(raw.info['ch_names'])
raw.pick_types(eeg=True)
print(raw.info['ch_names'], len(raw.info['ch_names']))


# %%

subject2 = '030047'
bids_path.update(subject=subject2)
raw2 = read_raw_bids(bids_path=bids_path)
raw2.drop_channels('A2')
print(raw2.info['ch_names'])
# %%
raw2.drop_channels('A2')
# %%
