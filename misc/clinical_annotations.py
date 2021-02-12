# %%

import pandas as pd
import mne

# %%
final_column_labels = ['Onset (s after orig_time)',
                       'Duration (s)',
                       'Description']
event_label = {'Éveil': 'Sleep stage W',
               'Stade N1': 'Sleep stage 1',
               'Stade N2': 'Sleep stage 2',
               'Stade N3': 'Sleep stage 3',
               'Stade N4': 'Sleep stage 4',
               'REM': 'Sleep stage R'}
clean_annot_path = (
    '/media/pallanca/datapartition/maelys/data/clean_annotations/annot')


def csv_to_df(filepath, fileref):
    """ Get csv file as pandas dataframe
    Select events marked as validated then add those missing
    Put everything together in final_df
    """
    df = pd.read_csv(filepath, sep='\t', encoding='UTF-16 LE')
    column_labels = {i: c for (i, c) in enumerate(df.columns)}
    seq_to_keep = select_events_olivier(df, column_labels)
    annot_df = df.loc[df['Sequence'].
                      isin(seq_to_keep)][[column_labels[0],
                                          column_labels[4],
                                          column_labels[5],
                                          column_labels[7]]]
    final_df = pd.DataFrame(columns=final_column_labels)
    final_df[final_column_labels[0]] = annot_df[column_labels[4]].transform(
                                lambda x: x / 1e6)
    final_df[final_column_labels[1]] = annot_df[column_labels[5]].transform(
                                lambda x: x / 1e6)
    final_df[final_column_labels[2]] = annot_df[column_labels[7]].transform(
                                lambda x: event_label[x] if x in event_label
                                else 'misc')
    save_final_df(fileref, final_df)
    return final_df[final_df['Description'] != 'misc']


def select_events_olivier(df, column_labels):
    seq_to_keep = list(df[df['Validated'] == 'Yes'][column_labels[0]])
    if seq_to_keep != []:
        val_df = df[df['Validated'] == 'Yes'][[column_labels[0],
                                               column_labels[4],
                                               column_labels[5], 
                                               column_labels[7], 
                                               column_labels[8]]]
        val_df.columns = ['sequence', 'onset', 'duration', 'description', 'validated']
        val_df[column_labels[4]].transform(lambda x: x / 1e6)
        val_df[column_labels[5]].transform(lambda x: x / 1e6)
        sequence = [val_df.iloc[0, 0], None]
        start_time = val_df.iloc[0, 1]
        duration = val_df.iloc[0, 2]
        th_start_time = start_time + duration
        for i in range(len(val_df)):
            start_time = val_df.iloc[i, 1]
            duration = val_df.iloc[i, 2]
            sequence[1] = val_df.iloc[i, 0]
            if start_time != th_start_time:
                seq_to_keep += list(range(sequence[0]+1, sequence[1]))
            th_start_time = start_time + duration
    else:
        seq_to_keep = list(df['Sequence'])
    print(val_df)
    print(seq_to_keep)
    return seq_to_keep


def df_to_annotation(final_df):
    onset = list(final_df.iloc[:, 0])
    duration = list(final_df.iloc[:, 1])
    description = list(final_df.iloc[:, 2])
    my_annot = mne.Annotations(onset, duration, description)
    return my_annot


def save_final_df(fileref, final_df):
    csv_filepath = clean_annot_path + fileref
    final_df.to_csv(csv_filepath+'.csv')

# %%

annot_path = '/media/pallanca/datapartition/maelys/data/csv_hypno/'
fileref = 'AFju890504'
annot_filepath = annot_path+fileref+'annot.csv'

# %%
df = pd.read_csv(annot_filepath, sep='\t', encoding='UTF-16 LE')
column_labels = {i: c for (i, c) in enumerate(df.columns)}
val_df = df[df['Validated'] == 'Yes'][[column_labels[0],
                                        column_labels[4],
                                        column_labels[5], 
                                        column_labels[7], 
                                        column_labels[8]]]
val_df.columns = ['sequence', 'onset', 'duration', 'description', 'validated']
seq_to_keep = list(val_df['sequence'])
val_df['onset'] = val_df['onset'].transform(lambda x: x / 1e6)
val_df['duration'] = val_df['duration'].transform(lambda x: x / 1e6)

# %%
th_onset = list(val_df['onset'] + val_df['duration'])
val_df['th_onset'] = [0] + th_onset[:-1]

# %%
onset_ok = []
for i in range(1, len(val_df)):
    onset_ok.append(True if val_df.iloc[i, 1] == val_df.iloc[i, 5] else False)
val_df['onset_ok'] = [None] + onset_ok


# %%
new_df = df[[column_labels[0],column_labels[4], column_labels[5], column_labels[7], column_labels[8]]]
new_df.columns = ['sequence', 'onset', 'duration', 'description', 'validated']
seq_to_keep = list(new_df['validated']=='Yes')
new_df['onset'] = new_df['onset'].transform(lambda x: x / 1e6)
new_df['duration'] = new_df['duration'].transform(lambda x: x / 1e6)

# %%
th_onset = list(new_df['onset'] + new_df['duration'])
new_df['th_onset'] = [0] + new_onset[:-1]

# %%
onset_ok = []
for i in range(1, len(new_df)):
    onset_ok.append(True if new_df.iloc[i, 1] == new_df.iloc[i, 5] else False)
new_df['onset_ok'] = [None] + onset_ok
# %%
annot_df = csv_to_df(annot_filepath, fileref)
annot = df_to_annotation(annot_df)

# %%
