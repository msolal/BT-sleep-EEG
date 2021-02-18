#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:23:18 2021

@author: maelys.solal
"""

# %% 
import pandas as pd

annot_path = '/home/maelys/Documents/misc-bt/clinical-annot/'
fileref1 = 'AFju890504'
fileref2 = 'ALma600418'
annot_filepath1 = annot_path+fileref1+'annot.csv'
annot_filepath2 = annot_path+fileref2+'annot.csv'

# %%
df = pd.read_csv(annot_filepath2, sep='\t', encoding='UTF-16 LE')
column_labels = {i: c for (i, c) in enumerate(df.columns)}
full_df = df[[column_labels[0], column_labels[4], column_labels[5], 
                column_labels[7], column_labels[8]]]
full_df.columns = ['sequence', 'onset', 'duration', 'description', 'validated']
full_df.loc[:, 'onset'] = full_df.loc[:, 'onset'] / 1e6
full_df.loc[:, 'duration'] = full_df.loc[:, 'duration'] / 1e6

# %%
th_next_onset = list(full_df['onset'] + full_df['duration'])
full_df.insert(5, 'th_next_onset', th_next_onset)

# %%
yes_df = full_df[full_df['validated'] == 'Yes']
first_yes_idx = yes_df.iloc[0, 0] - 1

# %%
def get_first_annots(final_df, annots_df, first_yes_idx):
    yes_onset = annots_df.loc[first_yes_idx, 'onset']
    # yes_duration = annots_df.loc[first_yes_idx, 'duration']
    yes_description = annots_df.loc[first_yes_idx, 'description']
    yes_th_next_onset = annots_df.loc[first_yes_idx, 'th_next_onset']
    prev_idx = first_yes_idx - 1
    prev_onset = annots_df.loc[prev_idx, 'onset']
    # prev_duration = annots_df.loc[prev_idx, 'duration']
    prev_description = annots_df.loc[prev_idx, 'description']
    prev_th_next_onset = annots_df.loc[prev_idx, 'th_next_onset']
    if prev_onset == yes_onset:
        final_df = final_df.append(annots_df.loc[:prev_idx-1])
    elif prev_th_next_onset == yes_onset:
        final_df = final_df.append(annots_df.loc[:prev_idx])
    elif prev_th_next_onset > yes_onset:
        if prev_description != yes_description:
            annots_df.loc[prev_idx, 'duration'] = yes_onset - annots_df.loc[prev_idx, 'th_next_onset']
            final_df = final_df.append(annots_df.loc[:prev_idx])
        else:
            annots_df.loc[first_yes_idx, 'duration'] = yes_th_next_onset - prev_onset
            annots_df.loc[first_yes_idx, 'onset'] = prev_onset
            final_df = final_df.append(annots_df.loc[:prev_idx])
    final_df = final_df.append(annots_df.loc[first_yes_idx])
    print(final_df)   
    
# %%
def get_intermediate_annots(final_df, annots_df, yes_idx_a, yes_idx_b):
    

# %%
final_df = pd.DataFrame()
get_first_annots(final_df, full_df, first_yes_idx)


# %%
final_column_labels = ['Onset (s after orig_time)',
                       'Duration (s)',
                       'Description']

# %%
# check overlap function
# combine events function