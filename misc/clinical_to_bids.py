# %%
import os
from clinical_annotations import csv_to_df, df_to_annotation
import mne
from mne_bids import write_raw_bids, BIDSPath

# %%
raw_path = '/media/pallanca/datapartition/maelys/data/edf/'
annot_path = '/media/pallanca/datapartition/maelys/data/csv_hypno/'
bids_root = '/media/pallanca/datapartition/maelys/data/BIDS'

all_ch_names = {'EEG C3': 'C3',
                'EEG C4': 'C4',
                'EEG F3': 'F3',
                'EEG F4': 'F4',
                'EEG M1': 'M1',
                'EEG M2': 'M2', 
                'EEG M1#2': 'M1',
                'EEG M2#2': 'M2', 
                'EEG O1': 'O1',
                'EEG O2': 'O2',
                'EEG O2*': 'O2',
                'EOG E1#1': 'EOG E1',
                'EOG E2#1': 'EOG E2'}

all_ch_types = {'BodyPos BodyPos': 'misc',
                'BodyPos Pos': 'misc',
                'ECG': 'ecg',
                'EMG Ment': 'emg',
                'EMG Chin1': 'emg',
                'EMG Tib-L': 'emg',
                'EMG Tib-R': 'emg',
                'EOG eog-l': 'eog',
                'EOG eog-r': 'eog',
                'EOG E1': 'eog',
                'EOG E2': 'eog',
                'EOG E1#1': 'eog',
                'EOG E2#1': 'eog',
                'Resp Abd': 'misc',
                'Resp Cann Raw': 'misc',
                'Resp Flux': 'misc',
                'Resp Flw2': 'misc',
                'Resp Ther': 'misc',
                'Resp Therm': 'misc',
                'Resp Thor': 'misc',
                'SaO2 HR': 'misc',
                'SaO2 Pulse': 'misc',
                'SaO2 SaO2': 'misc',
                'Sound Ronf': 'misc',
                'Sound Mic': 'misc',
                'Unspec BEAT': 'misc',
                'Unspec BP LVL': 'misc',
                'Unspec BP RAW': 'misc',
                'Unspec Flux': 'misc',
                'Unspec PULS': 'misc'}

# %%
raw_files = os.listdir(raw_path)
annot_files = os.listdir(annot_path)

raw_names = [filename.strip('.edf') for filename in raw_files]
annot_names = [filename.strip('annot.csv') for filename in annot_files]
common = list(set(raw_names) & set(annot_names))
common.sort()

# %%
for fileref in common:
    raw_filepath = raw_path + fileref + '.edf'
    annot_filepath = annot_path + fileref + 'annot.csv'
    subject = fileref[:10]
    annot_df = csv_to_df(annot_filepath, fileref)
    annot = df_to_annotation(annot_df)
    raw = mne.io.read_raw_edf(raw_filepath)
    ch_names = raw.info['ch_names']
    new_ch_names = {ch_name : all_ch_names[ch_name] for ch_name in ch_names if ch_name in all_ch_names.keys()}
    new_ch_types = {ch_name : all_ch_types[ch_name] for ch_name in ch_names if ch_name in all_ch_types.keys()}
    print(new_ch_types)
    for old, new in new_ch_names.items():
        raw._orig_units[new] = raw._orig_units[old]
        del raw._orig_units[old]
    raw.set_annotations(annot)
    raw.set_channel_types(new_ch_types)
    raw.rename_channels(new_ch_names)
    raw.info['line_freq'] = 50
    bids_path = BIDSPath(subject=subject, root=bids_root)
    write_raw_bids(raw, bids_path, overwrite=True)
