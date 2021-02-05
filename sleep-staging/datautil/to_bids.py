# %%
import os
from annotations import csv_to_df, df_to_annotation
import mne
from mne_bids import write_raw_bids, BIDSPath


raw_path = 'edf/'
annot_path = 'csv_hypno/'

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
    annot_df = csv_to_df(annot_filepath)
    annot = df_to_annotation(annot_df)
    raw = mne.io.read_raw_edf(raw_filepath)
    raw.set_annotations(annot)
    raw.info['line_freq'] = 50
    bids_path = BIDSPath(subject=subject, root='BIDS2')
    write_raw_bids(raw, bids_path)
