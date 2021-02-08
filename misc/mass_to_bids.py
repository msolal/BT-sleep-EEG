# %%
import glob
import mne
from os.path import basename
from mne_bids import write_raw_bids, BIDSPath


raw_path = '/storage/store2/data/mass/SS3/'
annot_path = '/storage/store2/data/mass/SS3/annotations/'

raw_files = glob.glob(raw_path+'*.edf')
annot_files = glob.glob(annot_path+'*.edf')

raw_names = [basename(raw_file)[:10] for raw_file in raw_files]
annot_names = [basename(annot_file)[:10] for annot_file in annot_files]
common = list(set(raw_names) & set(annot_names))
common.sort()

# %%
# ['ECG I', 'EEG A2-CLE', 'EEG C3-CLE', 'EEG C4-CLE', 'EEG Cz-CLE',
#  'EEG F3-CLE', 'EEG F4-CLE', 'EEG F7-CLE', 'EEG F8-CLE',
#  'EEG Fp1-CLE', 'EEG Fp2-CLE', 'EEG Fz-CLE', 'EEG O1-CLE',
#  'EEG O2-CLE', 'EEG Oz-CLE', 'EEG P3-CLE', 'EEG P4-CLE',
#  'EEG Pz-CLE', 'EEG T3-CLE', 'EEG T4-CLE', 'EEG T5-CLE', 'EEG T6-CLE',
#  'EMG Chin1', 'EMG Chin2', 'EMG Chin3', 'EOG Left Horiz', 'EOG Right Horiz']
ch_types_1 = {'ECG I': 'ecg',
              'EMG Chin1': 'emg',
              'EMG Chin2': 'emg',
              'EMG Chin3': 'emg',
              'EOG Right Horiz': 'eog',
              'EOG Left Horiz': 'eog'}
ch_rename_1 = {}

# ['ECG I', 'EEG C3-LER', 'EEG C4-LER', 'EEG Cz-LER', 'EEG F3-LER',
#  'EEG F4-LER', 'EEG F7-LER', 'EEG F8-LER', 'EEG Fp1-LER', 'EEG Fp2-LER',
#  'EEG Fz-LER', 'EEG O1-LER', 'EEG O2-LER', 'EEG Oz-LER', 'EEG P3-LER', 'EEG P4-LER',
#  'EEG Pz-LER', 'EEG T3-LER', 'EEG T4-LER', 'EEG T5-LER', 'EEG T6-LER', 'EMG Chin1',
#  'EMG Chin2', 'EMG Chin3', 'EOG Left Horiz','EOG Right Horiz', 'Resp Belt Abdo', 'Resp Belt Thor']
ch_types_2 = {'ECG I': 'ecg',
              'EMG Chin1': 'emg',
              'EMG Chin2': 'emg',
              'EMG Chin3': 'emg',
              'EOG Right Horiz': 'eog',
              'EOG Left Horiz': 'eog',
              'Resp Belt Abdo': 'resp',
              'Resp Belt Thor': 'resp'}
ch_rename_2 = {}

# %%
for fileref in common[:1]:
    raw_filepath = raw_path + fileref + ' PSG.edf'
    annot_filepath = annot_path + fileref + ' Annotations.edf'
    subject = fileref[3:5]+fileref[6:]
    raw = mne.io.read_raw_edf(raw_filepath)
    annot = mne.read_annotations(annot_filepath)
    annot.description = ['Sleep stage W' if x=='Sleep stage ?' else x for x in annot.description]
    raw.set_annotations(annot, emit_warning=False)
    channels = raw.info['ch_names']
    if 'Resp Belt Abdo' in channels:
        ch_types = ch_types_2
        ch_rename = ch_rename_2
    else: 
        ch_types = ch_types_1
        ch_rename = ch_rename_1
    raw.set_channel_types(ch_types)
    raw.info['line_freq'] = 50
    bids_path = BIDSPath(subject=subject, root='/storage/store/data/mass-bids/SS3')
    write_raw_bids(raw, bids_path, overwrite=True)
