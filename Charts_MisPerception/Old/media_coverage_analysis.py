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


#%% Directories and files

dir_input = os.path.join(os.path.dirname(os.getcwd()),'Data','Input','MediaCoverageTitles')

file_media_coverage = os.path.join(dir_input,"media_coverage.xlsx")

dir_output = os.path.join(os.path.dirname(os.getcwd()),'Data','Output')


#%% Read files

# Read the percentages
df_mediacov = pd.read_excel(file_media_coverage,"Percentage")


#%% Sort topics based on their mean value and calculate max

# Calculate a row with the mean value
df_mediacov['All'] =  df_mediacov.iloc[:,1:].mean(axis=1)

# Order based on the mean value
df_mediacov = df_mediacov.sort_values(by=['All'],ascending=False)

# Calculate the maximum observed value
max_val = df_mediacov.iloc[:,1:].max().max()


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

fig, ax = plt.subplots(1,1,dpi=600,figsize=(cm2inch(9.0),cm2inch(9.0*1.5)))

ax.barh(df_mediacov['Topic'],width=df_mediacov['All'], height=0.7,
        color='lightsteelblue')

aus = ax.plot(df_mediacov['Austria'],df_mediacov['Topic'],linestyle='',color='green',
        marker='D')
ax.plot(df_mediacov['Denmark'],df_mediacov['Topic'],linestyle='',color='red',
        marker='D')
ax.plot(df_mediacov['France'],df_mediacov['Topic'],linestyle='',color='blue',
        marker='D')
ax.plot(df_mediacov['Italy'],df_mediacov['Topic'],linestyle='',color='green',
        marker='o')
ax.plot(df_mediacov['Ireland'],df_mediacov['Topic'],linestyle='',color='red',
        marker='o')
ax.plot(df_mediacov['Germany'],df_mediacov['Topic'],linestyle='',color='blue',
        marker='o')
ax.plot(df_mediacov['Norway'],df_mediacov['Topic'],linestyle='',color='green',
        marker='x')
ax.plot(df_mediacov['Switzerland'],df_mediacov['Topic'],linestyle='',
        color='red',marker='x')

ax.invert_yaxis()

ax.set_yticks(np.arange(0, len(df_mediacov['Topic'].tolist())))

labels = df_mediacov['Topic'].tolist()
labels = [ '\n'.join(wrap(l, 21)) for l in labels ]
ax.set_yticklabels(labels)

vals = ax.get_xticks()
ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])

ax.set_xlabel('Incidence of topics in newspaper headlines',fontsize=6)

ax.legend(labels=['Austria','Denmark','France','Italy','Ireland',
                  'Germany','Norway','Switzerland',
                  'Mean value'],loc='lower right')

ax.set_axisbelow(True)
plt.grid(color='lightgray', linewidth=1, axis='x')

plt.tight_layout()

fig.savefig(os.path.join(dir_output,'Media_coverage_titles')+'.png',dpi=600)
