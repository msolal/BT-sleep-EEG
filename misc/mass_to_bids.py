import glob
import mne
from os.path import basename
from mne_bids import write_raw_bids, BIDSPath, update_sidecar_json
import xml.etree.ElementTree as ET

raw_path = '/storage/store2/data/mass/SS3/'
annot_path = '/storage/store2/data/mass/SS3/annotations/'
bids_root = '/storage/store2/data/mass-bids/SS3'

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

raw_files = glob.glob(raw_path+'*.edf')
annot_files = glob.glob(annot_path+'*.edf')

raw_names = [basename(raw_file)[:10] for raw_file in raw_files]
annot_names = [basename(annot_file)[:10] for annot_file in annot_files]
common = list(set(raw_names) & set(annot_names))
common.sort()

for fileref in common:
    raw_filepath = raw_path + fileref + ' PSG.edf'
    annot_filepath = annot_path + fileref + ' Annotations.edf'
    subject = fileref[3:5]+fileref[6:]
    raw = mne.io.read_raw_edf(raw_filepath)
    annots = mne.read_annotations(annot_filepath)
    for i, desc in enumerate(annots.description):
        if desc == 'Sleep stage ?':
            annots.description[i] = 'Sleep stage W'
        elif 'EMGArtefact' in desc:
            root = ET.fromstring(desc)
            channel = root.attrib["channel"].replace("-LER", "").replace("-CLE", "").split()[1]
            annots.description[i] = f"BAD_{root.attrib['groupName']}_{channel}"
        elif 'MicroArousal' in desc:
            root = ET.fromstring(desc)
            channel = root.attrib["channel"].replace("-LER", "").replace("-CLE", "").split()[1]
            annots.description[i] = f"{root.attrib['groupName']}_{channel}"
    raw.set_annotations(annots, emit_warning=False)
    ch_names = raw.info['ch_names']
    if 'Resp Belt Abdo' in ch_names:
        ch_types = ch_types_2
        ref = 'Linked Ear Reference'
    else: 
        ch_types = ch_types_1
        ref = 'Computed Linked Ear'
    new_ch_names = {}
    for old in ch_names:
        if old.startswith('EEG'):
            new = old.replace('EEG ', '').replace('-LER', '').replace('-CLE', '')
            new_ch_names[old] = new
            raw._orig_units[new] = raw._orig_units[old]
            del raw._orig_units[old]
    raw.rename_channels(new_ch_names)
    raw.set_channel_types(ch_types)
    raw.info['line_freq'] = 50
    bids_path = BIDSPath(subject=subject, root=bids_root)
    write_raw_bids(raw, bids_path, overwrite=True)
    bids_path.update(suffix='eeg', extension='.json')
    update_sidecar_json(bids_path, {'EEGReference': ref})
