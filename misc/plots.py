#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 20:17:13 2021

@author: maelys.solal
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cmap = sns.cubehelix_palette(50)
classes = ['W', 'N1', 'N2', 'N3', 'R'] 

columns = ['Precision', 'Recall', 'F1-score']
rows = ['W (4659)', 
        'N1 (2347)', 
        'N2 (5329)', 
        'N3 / N4 (673)', 
        'R (1993)', 
        'accuracy (15001)', 
        'macro_avg (15001)',
        'weighted_avg (15001)']

# MASS - SP
# conf_mat = [[1311, 56, 20, 1, 29], 
#             [27, 228, 127, 0, 95], 
#             [9, 81, 3178, 237, 84], 
#             [0, 0, 112, 971, 0], 
#             [9, 63, 74, 0, 1325]]
conf_mat = [[.93, .04, .01, 0, .02], 
            [.06, .48, .27, 0, .20], 
            [0, .02, .89, 0, .02], 
            [0, 0, .1, .9, 0],
            [.01, .04, .05, 0, .9]]

confusion_matrix = pd.DataFrame(conf_mat, columns=classes, index=classes)
plt.figure()
ax = sns.heatmap(confusion_matrix, annot=True, 
                 cmap=cmap, square=True, cbar=True)
ax.tick_params(left=False, bottom=False)
plt.title('Normalised confusion matrix')
plt.yticks(rotation=0)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()