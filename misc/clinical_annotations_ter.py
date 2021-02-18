#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 18:24:14 2021

@author: maelys.solal
"""

# %%
import pandas as pd
import numpy as np

annot_path = '/home/maelys/Documents/misc-bt/clinical-annot/'
fileref1 = 'AFju890504'
fileref2 = 'ALma600418'
annot_filepath1 = annot_path+fileref1+'annot.csv'
annot_filepath2 = annot_path+fileref2+'annot.csv'

df = pd.read_csv(annot_filepath1, sep='\t', encoding='UTF-16 LE')
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
    yes_idx = yes[0]
    yes_onset = yes[1]
    yes_desc = yes[3]
    yes_offset = yes[5]
    prev_idx = yes_idx - 1
    while True:
        prev_offset = data[prev_idx][5]
        prev_onset = data[prev_idx][1]
        prev_desc = data[prev_idx][3]
        if prev_offset == yes_onset:
            break
        elif prev_onset < yes_onset:
            if prev_offset <= yes_offset:
                if prev_desc != yes_desc:
                    data[prev_idx][5] = yes_onset
                    data[prev_idx][2] = data[prev_idx][5] - data[prev_idx][1]  
                else:
                    data[prev_idx][6] = False
                    data[yes_idx][1] = prev_onset
                    data[yes_idx][2] = data[yes_idx][5] - data[yes_idx][1]
            else:
                duplicate = data[prev_idx]
                data[prev_idx][5] = yes_onset
                data[prev_idx][2] = data[prev_idx][5] - data[prev_idx][1]
                duplicate[1] = yes_offset
                duplicate[2] = prev_offset - yes_offset
                data = np.append(data, [duplicate], axis=0)
            break  
        elif prev_onset == yes_onset:
            if prev_offset <= yes_offset:
                data[prev_idx][6] = False
            else:
                if prev_desc != yes_desc:
                    data[prev_idx][1] = yes_offset
                    data[prev_idx][2] = data[prev_idx][5] - data[prev_idx][1]  
                    data[[prev_idx, yes_idx]] = data[[yes_idx, prev_idx]]
                else:
                    data[prev_idx][6] = False
                    data[yes_idx][5] = prev_offset
                    data[yes_idx][2] = data[yes_idx][5] - data[yes_idx][1]
            prev_idx -= 1              
    print(prev_idx)
        
def get_next(data, yes):
    yes_idx = yes[0]
    yes_onset = yes[1]
    yes_desc = yes[3]
    yes_offset = yes[5]
    next_idx = yes_idx + 1
    while True:
        next_offset = data[next_idx][5]
        next_onset = data[next_idx][1]
        next_desc = data[next_idx][3]
        if next_onset == yes_offset:
            break
        elif next_offset == yes_offset:
            data[next_idx][6] = False
            break
        elif next_offset > yes_offset:
            if next_desc != yes_desc:
                data[next_idx][1] = yes_offset
                data[next_idx][2] = data[next_idx][5] - data[next_idx][1]  
            else:
                data[next_idx][6] = False
                data[yes_idx][5] = next_offset
                data[yes_idx][2] = data[yes_idx][5] - data[yes_idx][1]
            break
        elif next_offset < yes_offset:
            data[next_idx][6] = False
            next_idx += 1
    print(next_idx)
# %%
for yes in yes_data[:20]:
    if yes[0] != 0: 
        get_prev(data, yes)
    if yes[0] != len(data):
        get_next(data, yes)

fdata = data[data[:, 6] == True]
fdata = np.delete(fdata, 6, axis=1)

onset_ok = [[True if fdata[i+1, 1] == fdata[i, 5] else False] 
            for i in range(len(fdata)-1)]
fdata = np.append(fdata, [[None]] + onset_ok, axis=1)

fdata[:100]

# %%
# get_prev(yes_data[0])
# get_next(yes_data[0])

# %%
# get_prev(yes_data[1])
# get_next(yes_data[1])

# %%
# get_prev(yes_data[2])
# get_next(yes_data[2])

# %%
# final_column_labels = ['Onset (s after orig_time)',
#                        'Duration (s)',
#                        'Description']

# %%
# check overlap function
# combine events function