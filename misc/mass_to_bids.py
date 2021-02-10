import glob
import mne
from os.path import basename
from mne_bids import write_raw_bids, BIDSPath, update_sidecar_json
import xml.etree.ElementTree as ET

raw_path = '/storage/store2/data/mass/SS3/'
annot_path = '/storage/store2/data/mass/SS3/annotations/'
bids_root = '/storage/store2/data/mass-bids/SS3'

all_ch_types = {'ECG I': 'ecg',
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
            channel = (root.attrib["channel"]
                           .replace("-LER", "")
                           .replace("-CLE", "").split()[1])
            annots.description[i] = f"BAD_{root.attrib['groupName']}_{channel}"
        elif 'MicroArousal' in desc:
            root = ET.fromstring(desc)
            channel = (root.attrib["channel"]
                           .replace("-LER", "")
                           .replace("-CLE", "").split()[1])
            annots.description[i] = f"{root.attrib['groupName']}_{channel}"
    raw.set_annotations(annots, emit_warning=False)
    ch_names = raw.info['ch_names']
    new_ch_names, new_ch_types = {}, {}
    for ch_name in ch_names:
        if ch_name.startswith('EEG '): 
            new_ch_names[ch_name] = ch_name.replace('EEG ', '').replace('-LER', '').replace('-CLE', '')
        if ch_name in all_ch_types.keys():
            new_ch_types[ch_name] = all_ch_types[ch_name]
    if 'Resp Belt Abdo' in ch_names:
        ref = 'Linked Ear Reference'
    else: 
        ref = 'Computed Linked Ear'
    for old, new in new_ch_names.items():
        raw._orig_units[new] = raw._orig_units[old]
        del raw._orig_units[old]
    raw.rename_channels(new_ch_names)
    raw.set_channel_types(new_ch_types)
    raw.info['line_freq'] = 50
    bids_path = BIDSPath(subject=subject, root=bids_root)
    write_raw_bids(raw, bids_path, overwrite=True)
    bids_path.update(suffix='eeg', extension='.json')
    update_sidecar_json(bids_path, {'EEGReference': ref})
