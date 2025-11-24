# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:29:27 2023

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

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)


def cm2inch(cm):
    inc = cm / 2.54
    return inc


def median_calculation(e):
    return np.median(e)


# %% Directories and files

# Social media analysis
dir_input_socmedia = os.path.join(os.path.dirname(
    os.getcwd()), 'Data', 'Input', 'SocialMediaTopics')
file_socmedia_coverage = os.path.join(
    dir_input_socmedia, "social_media_coverage.xlsx")

# Outputs
dir_output = os.path.join(os.path.dirname(
    os.getcwd()), 'Data', 'Output', 'SocialMedia')


# %% Read files

# Social media analysis

# Read the excel file
df_socmediacov = pd.read_excel(file_socmedia_coverage, "Percentage", nrows=18)


# %% Settings for the plots

labels_size = 8
ticklabels_size = 8
markers_size = 4.5
line_width = 2
matplotlib.rcParams['axes.labelsize'] = labels_size
matplotlib.rcParams['axes.titlesize'] = labels_size
matplotlib.rcParams['xtick.labelsize'] = ticklabels_size
matplotlib.rcParams['ytick.labelsize'] = ticklabels_size
matplotlib.rcParams['legend.fontsize'] = labels_size
matplotlib.rcParams['lines.markersize'] = markers_size
matplotlib.rcParams['lines.linewidth'] = line_width


# %% Plotting chart 2

fig, ax = plt.subplots(1, 1, dpi=600, sharey=True,
                       figsize=(cm2inch(12.0), cm2inch(16.0)))

# Media analysis

ax.barh(np.arange(1, len(df_socmediacov)+1, 1), width=df_socmediacov['Average'], height=0.7,
           color='lightsteelblue')

aus = ax.plot(df_socmediacov['Austria'], np.arange(1, len(df_socmediacov)+1, 1), linestyle='', color='green',
                 marker='D')
ax.plot(df_socmediacov['Denmark'], np.arange(1, len(df_socmediacov)+1, 1), linestyle='', color='red',
           marker='D')
ax.plot(df_socmediacov['France'], np.arange(1, len(df_socmediacov)+1, 1), linestyle='', color='blue',
           marker='D')
ax.plot(df_socmediacov['Italy'], np.arange(1, len(df_socmediacov)+1, 1), linestyle='', color='green',
           marker='o')
ax.plot(df_socmediacov['Ireland'], np.arange(1, len(df_socmediacov)+1, 1), linestyle='', color='red',
           marker='o')
ax.plot(df_socmediacov['Germany'], np.arange(1, len(df_socmediacov)+1, 1), linestyle='', color='blue',
           marker='o')
ax.plot(df_socmediacov['Norway'], np.arange(1, len(df_socmediacov)+1, 1), linestyle='', color='green',
           marker='x')
ax.plot(df_socmediacov['Switzerland'], np.arange(1, len(df_socmediacov)+1, 1), linestyle='',
           color='red', marker='x')

ax.set_xlim([0, 0.6])

vals = ax.get_xticks()
ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])

ax.set_xlabel('Incidence of topics in tweets')

ax.set_yticks(np.arange(1,len(df_socmediacov)+1))

labels = df_socmediacov['Topic'].tolist()
labels = ['\n'.join(wrap(l, 21)) for l in labels]
ax.set_yticklabels(labels)


ax.legend(labels=['Austria', 'Denmark', 'France', 'Italy', 'Ireland',
                     'Germany', 'Norway', 'Switzerland',
                     'Average'], loc='lower right')

ax.set_axisbelow(True)
ax.grid(color='lightgray', linewidth=1, axis='x')


# All figure

plt.tight_layout()

fig.savefig(os.path.join(
    dir_output, 'Social_media_analysis')+'.png', dpi=600)



