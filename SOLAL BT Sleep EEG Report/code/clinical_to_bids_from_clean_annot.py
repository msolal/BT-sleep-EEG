import os
from clinical_annotations import clean_csv_to_annotation
import mne
from mne_bids import write_raw_bids, BIDSPath

raw_path = '/data/edf/'
annot_path = '/data/clean_annotations/'
bids_root = '/data/BIDS'

all_ch_names = {'EEG C3': 'C3',
                ...
                'EOG E2#1': 'EOG E2'}

all_ch_types = {'EMG Ment': 'emg',
                'EMG Chin1': 'emg',
                ...
                'EOG E1': 'eog',
                'EOG E2': 'eog',}

raw_files = os.listdir(raw_path)
annot_files = os.listdir(annot_path)
raw_names = [filename.strip('.edf') for filename in raw_files]
annot_names = [filename.strip('annot.csv') for filename in annot_files]
common = list(set(raw_names) & set(annot_names))

for fileref in common:
    raw_filepath = raw_path + fileref + '.edf'
    annot_filepath = annot_path + fileref + '.csv'
    subject = fileref[:10]
    annot = clean_csv_to_annotation(annot_filepath)
    raw = mne.io.read_raw_edf(raw_filepath)
    ch_names = raw.info['ch_names']
    new_ch_names = {ch_name : all_ch_names[ch_name] for ch_name in ch_names if ch_name in all_ch_names.keys()}
    new_ch_types = {ch_name : all_ch_types[ch_name] for ch_name in ch_names if ch_name in all_ch_types.keys()}
    for old, new in new_ch_names.items():
        raw._orig_units[new] = raw._orig_units[old]
        del raw._orig_units[old]
    raw.set_annotations(annot)
    raw.set_channel_types(new_ch_types)
    raw.rename_channels(new_ch_names)
    raw.info['line_freq'] = 50
    bids_path = BIDSPath(subject=subject, root=bids_root)
    write_raw_bids(raw, bids_path, overwrite=True)
