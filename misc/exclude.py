from mne_bids import BIDSPath, read_raw_bids

bids_root = '/storage/store2/data/mass-bids/SS3'
bids_path = BIDSPath(subject='030001', datatype='eeg', root=bids_root)

raw1 = read_raw_bids(bids_path=bids_path)
print(raw1.info['ch_names'])

raw2 = read_raw_bids(bids_path=bids_path, extra_params=dict(exclude=['ECG I']))

raw3 = read_raw_bids(bids_path=bids_path, extra_params=dict(exclude=[]))


