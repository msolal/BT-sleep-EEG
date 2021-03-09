#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:40:11 2021

@author: maelys.solal
"""

# %%
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
classes_mapping = {'Sleep stage R': 0, 'Sleep stage 4': 1, 'Sleep stage 3': 2, 
                   'Sleep stage 2': 3, 'Sleep stage 1': 4, 'Sleep stage W': 5}

# %%
path = '/media/pallanca/datapartition/maelys/data/clean_annotations/'
files = glob.glob(path+'*.csv')
# filerefs = [os.path.basename(file).strip('annot.csv') for file in files]
fileref = 'AFju890504'

filepath = path+fileref+'.csv'
df = pd.read_csv(filepath)
data = df.values[:, [2, 3, 6]]

# %%
start = data[0][0]
end = data[-1][0]+data[-1][1]

# %%
# times = np.arange(start, end)
times = []
annot = []
for line in data:
    onset = int(line[0])
    duration = int(line[1])
    offset = onset+duration
    description = line[2]
    times += list(range(onset, offset))
    annot += [classes_mapping[description] if description in classes_mapping 
              else -1 for _ in range(onset, offset)]

# %%
sns.set(context='talk')
plt.figure(figsize=(20, 5))
plt.plot(times, annot)
# plt.plot(times, annot, marker='+', linestyle='None')