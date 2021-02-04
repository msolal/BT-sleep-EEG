#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:44:34 2021

@author: maelys.solal
"""

# %%
import pandas as pd
import mne


def csv_to_df(filepath):
    """ Get csv file as pandas dataframe
    """
    df = pd.read_csv(filepath, sep='\t', encoding='UTF-16 LE')
    column_labels = {i: c for (i, c) in enumerate(df.columns)}
    return df, column_labels


def select_seq(df, column_labels):
    """ Select events marked as validated then add those missing
    """
    seq_to_keep = list(df[df['Validated'] == 'Yes'][column_labels[0]])
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
    return seq_to_keep


def create_final_df(df, column_labels, seq_to_keep):
    """ Put everything together in final_df
    """
    annot_df = df.loc[df['Sequence'].
                      isin(seq_to_keep)][[column_labels[0],
                                          column_labels[4],
                                          column_labels[5],
                                          column_labels[7]]]
    final_column_labels = ['Onset (s after orig_time)',
                           'Duration (s)',
                           'Description']
    event_label = {'Éveil': 'Sleep stage W',
                   'Stade N1': 'Sleep stage 1',
                   'Stade N2': 'Sleep stage 2',
                   'Stade N3': 'Sleep stage 3',
                   'REM': 'Sleep stage R'}
    final_df = pd.DataFrame(columns=final_column_labels)
    final_df[final_column_labels[0]] = annot_df[column_labels[4]].transform(
                                lambda x: x / 1e6)
    final_df[final_column_labels[1]] = annot_df[column_labels[5]].transform(
                                lambda x: x / 1e6)
    final_df[final_column_labels[2]] = annot_df[column_labels[7]].transform(
                                lambda x: event_label[x])
    return final_df


df, column_labels = csv_to_df(filepath)
seq_to_keep = select_seq(df, column_labels)
final_df = create_final_df(df, column_labels, seq_to_keep)

# %%
onset = list(final_df.iloc[:, 0])
duration = list(final_df.iloc[:, 1])
description = list(final_df.iloc[:, 2])
my_annot = mne.Annotations(onset, duration, description)

# %%
raw = mne.io.read_raw_edf(rawfile)
raw.set_annotations(my_annot)
# %%
