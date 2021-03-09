# %%
import mne
import os
from mne_bids import write_raw_bids, BIDSPath
from mne.datasets.sleep_physionet.age import fetch_data

# %%
all_sub = (list(range(13))+list(range(14, 36))+[37, 38]+list(range(40, 52))
           +list(range(53, 68))+list(range(69, 78))+list(range(80, 82)))
recording = [1, 2]

paths = fetch_data(all_sub, recording=recording, on_missing='warn')
bids_root = '/storage/store2/data/SleepPhysionet-bids/'

all_ch_types = {'EEG Fpz-Cz': 'eeg',
                'EEG Pz-Oz': 'eeg',
                'EOG horizontal': 'eog',
                'Resp oro-nasal': 'resp',
                'EMG submental': 'emg',
                'Temp rectal': 'misc',
                'Event marker': 'misc'}
 
# %%
for raw_path, annot_path in paths:
    raw = mne.io.read_raw_edf(raw_path)
    annots = mne.read_annotations(annot_path)
    raw.set_annotations(annots)
    subject = os.path.basename(annot_path).replace('-Hypnogram.edf', '').replace('SC', '')
    new_ch_names, new_ch_types = {}, {}
    for ch_name in raw.info['ch_names']:
        if ch_name.startswith('EEG '):
            new_ch_names[ch_name] = ch_name.replace('EEG ', '')
        if ch_name in all_ch_types.keys():
            new_ch_types[ch_name] = all_ch_types[ch_name]
    for old, new in new_ch_names.items():
        raw._orig_units[new] = raw._orig_units[old]
        del raw._orig_units[old]
        new_ch_types[new] = new_ch_types[old]
        del new_ch_types[old]
    raw.rename_channels(new_ch_names)
    raw.set_channel_types(new_ch_types)
    raw.info['line_freq'] = 50
    bids_path = BIDSPath(subject=subject, root=bids_root)
    write_raw_bids(raw, bids_path, overwrite=True)