#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:49:57 2021

@author: maelys.solal
"""

# %%
import numpy as np
import pandas as pd


# %%
annot_path = '/home/maelys/Documents/misc-bt/clinical-annot/'
fileref1 = 'AFju890504'
fileref2 = 'ALma600418'
annot_filepath1 = annot_path+fileref1+'annot.csv'
annot_filepath2 = annot_path+fileref2+'annot.csv'

df = pd.read_csv(annot_filepath2, sep='\t', encoding='UTF-16 LE')
# data is an array [sequence, onset, duration, description, validated]
data = df.values[:, [0, 4, 5, 7, 8]]
# change sequence number to start at 0
data[:, 0] = data[:, 0] - 1
# change onset and duration to seconds 
data[:, [1, 2]] = data[:, [1, 2]] / 1e6
# data: [sequence, onset, duration, description, validated, offset, keep]
offset = np.reshape(data[:, 1] + data[:, 2], (-1, 1))
keep = [[True] for _ in range(len(data))]
data = np.append(data, offset, axis=1)
data = np.append(data, keep, axis=1)
# yes_data contains only the events which have been validated by hand
yes_data = data[data[:, 4] == 'Yes']

# %%
def get_prev(data, yes):
    yes_idx, yes_onset, _, yes_desc, _, yes_offset, _ = yes
    prev_idx = yes_idx - 1
    while True:
        prev_offset = data[prev_idx][5]
        prev_onset = data[prev_idx][1]
        if prev_offset == yes_onset:
            break
        elif prev_onset < yes_onset:
            data[prev_idx][5] = yes_onset
            data[prev_idx][2] = data[prev_idx][5] - data[prev_idx][1]
            break  
        elif prev_onset == yes_onset:
            data[prev_idx][6] = False
            prev_idx -= 1          
        
def get_next(data, yes):
    yes_idx, yes_onset, _, _, _, yes_offset, _ = yes
    next_idx = yes_idx + 1
    while True:
        next_offset = data[next_idx][5]
        next_onset = data[next_idx][1]
        if next_onset == yes_offset:
            break
        elif next_offset > yes_offset:
            data[next_idx][1] = yes_offset
            data[next_idx][2] = data[next_idx][5] - data[next_idx][1]  
            break
        elif next_offset <= yes_offset:
            data[next_idx][6] = False
            next_idx += 1

# %%
for yes in yes_data:
    if yes[0] != 0: 
        get_prev(data, yes)
    if yes[0] != len(data):
        get_next(data, yes)

onset_ok = [[True if data[i+1, 1] == data[i, 5] else False] 
            for i in range(len(data)-1)]
data = np.append(data, [[None]] + onset_ok, axis=1)

for line in data:
    if line[6] == True and line[7] == False:
        idx, onset, _, desc, _, offset, _, _ = line
        prev_idx = idx - 1
        _, prev_onset, _, prev_desc, _, prev_offset, _, _ = data[prev_idx]
        if desc == prev_desc:
            data[prev_idx][5] = offset
            data[prev_idx][2] = data[prev_idx][5] - data[prev_idx][1]
            data[idx][6] = False

clean_data = data[data[:, 6] == True]
onset_ok = [[True if clean_data[i+1, 1] == clean_data[i, 5] else False] 
            for i in range(len(clean_data)-1)]
clean_data = np.append(clean_data, [[None]] + onset_ok, axis=1)
clean_data = np.delete(clean_data, [6, 7], axis=1)


# %%
final_column_labels = ['Onset (s after orig_time)',
                       'Duration (s)',
                       'Description']

final_df = pd.DataFrame(columns=final_column_labels)
save_final_df(fileref, final_df)

final_df = 

data[:100]
fdata[:100]