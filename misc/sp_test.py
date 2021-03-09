# %%
from mne_bids import read_raw_bids, BIDSPath
from mne.datasets.sleep_physionet.age import fetch_data

bids_root = '/storage/store2/data/SleepPhysionet-bids/'
subject = '4011EH'

bids_path = BIDSPath(subject=subject, datatype='eeg', root=bids_root)
raw = read_raw_bids(bids_path)
raw.pick_types(eeg=True)
raw.plot()

# %%
path = fetch_data(subjects=[1], recording=[1], on_missing='warn')
print(path)
raw = mne.io.read_raw_edf(path[0][0])
annots = mne.read_annotations(path[0][1])
raw.set_annotations(annots)
raw.pick_types(eeg=True)
raw.plot()
# %%
