# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:27:36 2023

@author: ricca
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
from openpyxl import load_workbook
from textwrap import wrap


params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)


def cm2inch(cm):
    inc = cm / 2.54
    return inc

def median_calculation(e):
    return np.median(e)


#%% Directories and files

dir_input = os.path.join(os.path.dirname(os.getcwd()),'Data','Input','PublishPerish')

file_topics_list = os.path.join(dir_input,"List_topics.xlsx")

dir_output = os.path.join(os.path.dirname(os.getcwd()),'Data','Output')


#%% Read files

# Read the list of topics
df_topiclist = pd.read_excel(file_topics_list,'Sheet1')

# Create a list for reading the results
list_scilit_cit = []
list_topic_median = []

# For cycle across topics
for index, row in df_topiclist.iterrows():
    
    file_name = os.path.join(dir_input,'Topic_' + str(row['Topic number']) + '.csv')
    df_topic = pd.read_csv(file_name)
    list_scilit_cit.append(df_topic['CitesPerYear'].values)
    list_topic_median.append(df_topic['CitesPerYear'].median())


#%% Sort topics based on the median

# df_topic_median = pd.DataFrame(list_topic_median)
# df_topic_median = df_topic_median.rename(columns={0:'Median'})
# df_topic_median['Number'] = df_topiclist['Topic number']
# df_topic_median = df_topic_median.sort_values(by=['Median'],ascending=False)

df_topiclist['Median'] = list_topic_median
df_topiclist = df_topiclist.sort_values(by=['Median'],ascending=False)


#%% Settings for the plots

labels_size = 6
ticklabels_size = 6
markers_size = 4.5
line_width = 2
matplotlib.rcParams['axes.labelsize'] = labels_size
matplotlib.rcParams['axes.titlesize'] = labels_size
matplotlib.rcParams['xtick.labelsize'] = ticklabels_size
matplotlib.rcParams['ytick.labelsize'] = ticklabels_size
matplotlib.rcParams['legend.fontsize'] = labels_size
matplotlib.rcParams['lines.markersize'] = markers_size
matplotlib.rcParams['lines.linewidth'] = line_width


#%% Plot results

fig, ax = plt.subplots(1,1,dpi=600, figsize=(cm2inch(9.0),cm2inch(9.0*1.5)))

list_scilit_cit.sort(key=median_calculation,reverse=True)

bplot = ax.boxplot(list_scilit_cit, vert=False, notch=False, patch_artist = True,
           widths=0.7, showmeans=False, showfliers=False,
           medianprops={'color':'red'},meanprops={'marker':'x'})

for bb in bplot:
    for patch in bplot['boxes']:
        patch.set_facecolor('lightsteelblue')

ax.invert_yaxis()

ax.set_yticks(np.arange(1, len(df_topiclist['Topic'].tolist())+1))

labels = df_topiclist['Topic'].tolist()
labels = [ '\n'.join(wrap(l, 21)) for l in labels ]
ax.set_yticklabels(labels)

ax.set_xlabel('Number of citations per year',fontsize=6)

plt.grid(color='lightgray', linewidth=1, axis='x')

plt.tight_layout()

fig.savefig(os.path.join(dir_output,'Scilit_citations')+'.png',dpi=600)
