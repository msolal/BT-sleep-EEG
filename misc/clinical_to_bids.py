# %%
import os
from clinical_annotations import csv_to_df, df_to_annotation
import mne
from mne_bids import write_raw_bids, BIDSPath

# %%

# EEG O2*
# ['BodyPos BodyPos', 'BodyPos Pos', 'ECG', 'EEG C3', 'EEG C4', 'EEG F3',
#  'EEG F4', 'EEG M1', 'EEG M2', 'EEG O1', 'EEG O2*', 'EMG Ment',
#  'EMG Tib-L', 'EMG Tib-R', 'EOG eog-l', 'EOG eog-r', 'Resp Abd',
#  'Resp Flux', 'Resp Flw2', 'Resp Therm', 'Resp Thor', 'SaO2 HR',
#  'SaO2 Pulse', 'SaO2 SaO2', 'Sound Ronf']
naming_1 = {'EEG O2*': 'EEG O2'}
ch_types_1 = {'BodyPos BodyPos': 'misc',
              'BodyPos Pos': 'misc',
              'ECG': 'ecg',
              'EMG Ment': 'emg',
              'EMG Tib-L': 'emg',
              'EMG Tib-R': 'emg',
              'EOG eog-l': 'eog',
              'EOG eog-r': 'eog',
              'Resp Abd': 'resp',
              'Resp Flux': 'resp',
              'Resp Flw2': 'resp',
              'Resp Therm': 'resp',
              'Resp Thor': 'resp',
              'SaO2 HR': 'misc',
              'SaO2 Pulse': 'misc',
              'SaO2 SaO2': 'misc',
              'Sound Ronf': 'misc'}

# EEG M1#2 and M2#2
# ['BodyPos Pos', 'ECG', 'EEG C3', 'EEG C4', 'EEG F3',
#  'EEG F4', 'EEG M1#2', 'EEG M2#2', 'EEG O1', 'EEG O2',
#  'EMG Chin1', 'EMG Tib-L', 'EMG Tib-R', 'EOG E1', 'EOG E2',
#  'Resp Abd', 'Resp Cann Raw', 'Resp Ther', 'Resp Thor',
#  'SaO2 SaO2', 'Sound Ronf', 'Unspec BEAT', 'Unspec BP LVL',
#  'Unspec BP RAW', 'Unspec Flux', 'Unspec PULS']
naming_2 = {'EEG M1#2': 'EEG M1',
            'EEG M2#2': 'EEG M2'}
ch_types_2 = {'BodyPos Pos': 'misc',
              'ECG': 'ecg',
              'EMG Chin1': 'emg',
              'EMG Tib-L': 'emg',
              'EMG Tib-R': 'emg',
              'EOG E1': 'eog',
              'EOG E2': 'eog',
              'Resp Abd': 'resp',
              'Resp Cann Raw': 'resp',
              'Resp Ther': 'resp',
              'Resp Thor': 'resp',
              'SaO2 SaO2': 'misc',
              'Sound Ronf': 'misc',
              'Unspec BEAT': 'misc',
              'Unspec BP LVL': 'misc',
              'Unspec BP RAW': 'misc',
              'Unspec Flux': 'misc',
              'Unspec PULS': 'misc'}

# no EEG M1#2, no M2#2, no O2*
# ['BodyPos BodyPos', 'BodyPos Pos', 'ECG', 'EEG C3', 'EEG C4', 'EEG F3',
#  'EEG F4', 'EEG M1', 'EEG M2', 'EEG O1', 'EEG O2', 'EMG Chin1',
#  'EMG Tib-L', 'EMG Tib-R', 'EOG E1#1', 'EOG E2#1', 'Resp Abd',
#  'Resp Flux', 'Resp Flw2', 'Resp Therm', 'Resp Thor', 'SaO2 HR',
#  'SaO2 Pulse', 'SaO2 SaO2', 'Sound Mic']
naming_3 = {'EOG E1#1': 'EOG E1',
            'EOG E2#1': 'EOG E2'}
ch_types_3 = {'BodyPos BodyPos': 'misc',
              'BodyPos Pos': 'misc',
              'ECG': 'ecg',
              'EMG Chin1': 'emg',
              'EMG Tib-L': 'emg',
              'EMG Tib-R': 'emg',
              'EOG E1#1': 'eog',
              'EOG E2#1': 'eog',
              'Resp Abd': 'resp',
              'Resp Flux': 'resp',
              'Resp Flw2': 'resp',
              'Resp Thor': 'resp',
              'SaO2 HR': 'misc',
              'SaO2 Pulse': 'misc',
              'SaO2 SaO2': 'misc',
              'Sound Mic': 'misc'}

# %%
raw_path = 'data/edf/'
annot_path = 'data/csv_hypno/'

raw_files = os.listdir(raw_path)
annot_files = os.listdir(annot_path)

raw_names = [filename.strip('.edf') for filename in raw_files]
annot_names = [filename.strip('annot.csv') for filename in annot_files]
common = list(set(raw_names) & set(annot_names))
common.sort()

for fileref in common[:10]:
    raw_filepath = raw_path + fileref + '.edf'
    annot_filepath = annot_path + fileref + 'annot.csv'
    subject = fileref[:10]
    annot_df = csv_to_df(annot_filepath, fileref)
    annot = df_to_annotation(annot_df)
    raw = mne.io.read_raw_edf(raw_filepath)
    channels = raw.info['ch_names']
    if 'EEG O2*' in channels:
        channel_types = ch_types_1
        naming = naming_1
    elif 'EEG M1#2' in channels:
        channel_types = ch_types_2
        naming = naming_2
    else:
        channel_types = ch_types_3
        naming = naming_3
    raw.set_annotations(annot)
    raw.rename_channels(naming)
    raw.set_channel_types(channel_types)
    raw.info['line_freq'] = 50
    bids_path = BIDSPath(subject=subject, root='data/BIDS')
    write_raw_bids(raw, bids_path)
