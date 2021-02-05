import pandas as pd
import mne
import os


final_column_labels = ['Onset (s after orig_time)',
                       'Duration (s)',
                       'Description']
event_label = {'Éveil': 'Sleep stage W',
               'Stade N1': 'Sleep stage 1',
               'Stade N2': 'Sleep stage 2',
               'Stade N3': 'Sleep stage 3',
               'Stade N4': 'Sleep stage 4',
               'REM': 'Sleep stage R'}


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
                                               column_labels[5]]]
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
    return seq_to_keep


def df_to_annotation(final_df):
    onset = list(final_df.iloc[:, 0])
    duration = list(final_df.iloc[:, 1])
    description = list(final_df.iloc[:, 2])
    my_annot = mne.Annotations(onset, duration, description)
    return my_annot


def save_final_df(fileref, final_df):
    csv_filepath = 'data/clean_annotations' + fileref
    final_df.to_csv(csv_filepath)
