#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 14:40:15 2021

@author: maelys.solal
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_stacked_bar(data, series_labels, category_labels=None, 
                     show_values=False, value_format="{}", y_label=None, 
                     x_label=None, colors=None, grid=False, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will 
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))



    data = np.array(data, dtype=int)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)



    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)
    
    if x_label: 
        plt.xlabel(x_label)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2, 
                         value_format.format(h), ha="center", 
                         va="center")

fig, ax = plt.subplots()

series_labels = ['Training set', 'Validation set', 'Testing set']
cat_labels = ('0', 'W', '1', '2', '3')
x_label = 'Sleep stages'
y_label = 'Number of samples'

training = np.array([1846, 1254, 9376, 2796, 3576], dtype=int)
validation = np.array([562, 385, 3109, 686, 881], dtype=int)
testing = np.array([1471, 477, 3589, 1083, 1417], dtype=int)
data = np.array([training, validation, testing], dtype=int)

nb_cat = len(cat_labels)
x = np.arange(nb_cat)
# categories = {n:cat for (n, cat) in zip(x, cat_labels)}

axes = []
cum_size = np.zeros(nb_cat)

for i, row_data in enumerate(data):
    axes.append(ax.bar(x, row_data, bottom=cum_size, label=series_labels[i]))
    cum_size += row_data

ax.set_xlabel(x_label)
# ax.set_xticks(x)
ax.set_xticklabels(cat_labels)
ax.set_ylabel(y_label)

ax.bar_label(axes[0], label_type='center')
ax.bar_label(axes[1], label_type='center')
ax.bar_label(axes[2], label_type='center')

ax.legend()

# plt.savefig('bar.png')
fig.tight_layout()
plt.title('Sleep stages imbalance in MASS dataset')
plt.show()
