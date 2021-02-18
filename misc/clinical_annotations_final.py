import numpy as np
import pandas as pd
import mne
import glob
import os

final_column_labels = ['Sequence', 
                       'Onset (s after orig_time)',
                       'Duration (s)',
                       'Offset (s)', 
                       'Onset ok',
                       'Description']

event_label = {'Éveil': 'Sleep stage W',
               'Stade N1': 'Sleep stage 1',
               'Stade N2': 'Sleep stage 2',
               'Stade N3': 'Sleep stage 3',
               'Stade N4': 'Sleep stage 4',
               'REM': 'Sleep stage R'}

map_labels = np.vectorize(
    lambda x: event_label[x] if x in event_label else 'misc')


annot_path = '/media/pallanca/datapartition/maelys/data/csv_hypno/'
files = glob.glob(annot_path+'*.csv')
filerefs = [os.path.basename(file).strip('annot.csv') for file in files]
clean_annot_path = (
    '/media/pallanca/datapartition/maelys/data/clean_annotations/')


def csv_to_df(filepath, fileref):
    """ Get csv file as pandas dataframe
    Select events marked as validated then add those missing
    Put everything together in final_df
    """
    df = pd.read_csv(filepath, sep='\t', encoding='UTF-16 LE')
    # data is an array [sequence, onset, duration, description, validated]
    data = df.values[:, [0, 4, 5, 7, 8]]
    data[:, 0] = data[:, 0] - 1                 # seq nb starts at 0
    data[:, [1, 2]] = data[:, [1, 2]] / 1e6     # onset&duration converted in s
    # data: [sequence, onset, duration, description, validated, offset, keep]
    data = compute_offset(data)
    keep = [[True] for _ in range(len(data))]
    data = np.append(data, keep, axis=1)
    data = select_events_clinical(data)
    data = compute_offset(data)
    data = merge_identical_events(data)
    data = compute_offset(data)
    clean_data = data[data[:, 7] == True]
    final_df = pd.DataFrame(columns=final_column_labels)
    final_df[final_column_labels[0]] = clean_data[:, 0]
    final_df[final_column_labels[1]] = clean_data[:, 1]
    final_df[final_column_labels[2]] = clean_data[:, 2]
    final_df[final_column_labels[3]] = clean_data[:, 5]
    final_df[final_column_labels[4]] = clean_data[:, 6]
    final_df[final_column_labels[5]] = map_labels(clean_data[:, 3])
    save_final_df(fileref, final_df)
    return final_df[final_df['Description'] != 'misc']


def select_events_clinical(data):
    yes_data = data[data[:, 4] == 'Yes']
    for yes in yes_data:
        if yes[0] != 0: 
            get_prev(data, yes)
        if yes[0] != len(data) - 1:
            get_next(data, yes)
    return data
    

def prev_keep_idx(data, idx):
    prev_idx = idx - 1
    while data[prev_idx][7] != True and prev_idx >= 0:
        prev_idx -= 1
    return prev_idx

def next_keep_idx(data, idx):
    next_idx = idx + 1
    while data[next_idx][7] != True and next_idx < len(data) - 1:
        next_idx += 1
    return next_idx

def compute_offset(data):
    if data.shape[1] <= 5:
        offset = np.reshape(data[:, 1] + data[:, 2], (-1, 1))
        data = np.append(data, offset, axis=1)
        onset_ok = [[True if data[i+1, 1] == data[i, 5] 
                     else False] for i in range(len(data)-1)]
        data = np.append(data, [[None]] + onset_ok, axis=1)
    else:
        offset = data[:, 1] + data[:, 2]
        data[:, 5] = offset
        onset_ok = [True if data[i+1, 1] == data[prev_keep_idx(data, i+1), 5] 
                    else False for i in range(len(data)-1)]
        data[:, 6] = [None] + onset_ok
    return data
        

def merge_identical_events(data):
    for line in data:
        if line[6] == False and line[7] == True:
            idx, desc, offset= line[0], line[3], line[5]
            prev_idx = prev_keep_idx(data, idx)
            prev_desc, prev_offset = data[prev_idx][3], data[prev_idx][5]
            next_idx = next_keep_idx(data, idx)
            next_onset = data[next_idx][1]
            if desc == prev_desc:
                if next_onset < len(data) - 1:
                    data[prev_idx][5] = min(max(offset, prev_offset), next_onset)
                else:
                    data[prev_idx][5] = max(offset, prev_offset)
                data[prev_idx][2] = data[prev_idx][5] - data[prev_idx][1]
                data[idx][7] = False
    return data

             
def df_to_annotation(final_df):
    onset = list(final_df.iloc[:, 0])
    duration = list(final_df.iloc[:, 1])
    description = list(final_df.iloc[:, 3])
    my_annot = mne.Annotations(onset, duration, description)
    return my_annot


def save_final_df(fileref, final_df):
    csv_filepath = clean_annot_path + fileref
    final_df.to_csv(csv_filepath+'.csv')


def get_prev(data, yes):
    yes_idx, yes_onset = yes[0], yes[1]
    prev_idx = yes_idx - 1
    while True and prev_idx >= 0:
        prev_offset = data[prev_idx][5]
        prev_onset = data[prev_idx][1]
        if prev_offset == yes_onset:
            break
        elif prev_onset < yes_onset:
            data[prev_idx][5] = yes_onset
            data[prev_idx][2] = data[prev_idx][5] - data[prev_idx][1]
            break  
        elif prev_onset == yes_onset:
            data[prev_idx][7] = False
            prev_idx -= 1          
        
def get_next(data, yes):
    yes_idx, yes_offset= yes[0], yes[5]
    next_idx = yes_idx + 1
    while True and next_idx < len(data):
        next_offset = data[next_idx][5]
        next_onset = data[next_idx][1]
        if next_onset == yes_offset:
            break
        elif next_offset > yes_offset:
            data[next_idx][1] = yes_offset
            data[next_idx][2] = data[next_idx][5] - data[next_idx][1]  
            break
        elif next_offset <= yes_offset:
            data[next_idx][7] = False
            next_idx += 1


for fileref in filerefs:
    print(fileref)
    if fileref in {'MEga720912', 'WAre750130', 'MOas940705', 'COma390828', 'OZch790212', 'PRma490730'}:
        annot_filepath = annot_path+fileref+'annot.csv'
        final_df = csv_to_df(annot_filepath, fileref)
        my_annot = df_to_annotation(final_df)

