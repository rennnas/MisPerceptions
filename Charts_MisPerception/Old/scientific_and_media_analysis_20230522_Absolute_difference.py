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

# Scientific analysis
dir_input_sci = os.path.join(os.path.dirname(
    os.getcwd()), 'Data', 'Input', 'Scopus')
file_topics_list = os.path.join(dir_input_sci, "List_topics.xlsx")
file_sci_incidence = os.path.join(
    dir_input_sci, "Incidence_scientific_literature_wo_transmission.xlsx")

# Media coverage analysis
dir_input_media = os.path.join(os.path.dirname(
    os.getcwd()), 'Data', 'Input', 'MediaCoverageTitles')
file_media_coverage = os.path.join(dir_input_media, "media_coverage.xlsx")

# Outputs
dir_output = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Output')


# %% Read files

# Scientific analysis

# Read the list of topics
df_topiclist = pd.read_excel(file_topics_list, 'Sheet1')

# Read the results
df_scicov = pd.read_excel(file_sci_incidence, 'Sheet1')


# Media coverage

# Read the excel file
df_mediacov = pd.read_excel(file_media_coverage, "Percentage", nrows=18)


# %% Normalise number of articles in scientific analysis

# Calculate median from general database
df_scicov['Sci_incidence'] = df_scicov['Number of articles'] / \
    df_scicov['Number of articles'].iloc[-1]

# Drop the last row
df_scicov = df_scicov.drop(df_scicov.tail(1).index)


# %% Handle rankings

# Merge the two rankings
df_allcov = pd.concat([df_scicov, df_mediacov], axis=1)

# Calculate difference and absolute difference between incidences
df_allcov['Value_difference'] = df_allcov['Average'] - \
    df_allcov['Sci_incidence']
df_allcov['Abs_value_difference'] = abs(df_allcov['Value_difference'])

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


# %% Plotting chart 1

fig, ax = plt.subplots(1, 2, dpi=600, sharey=True,
                       figsize=(cm2inch(16.0), cm2inch(16.0)))


# Scientific analysis

ax[0].barh(np.arange(1, len(df_allcov)+1, 1), width=df_allcov['Sci_incidence'], height=0.7,
           color='lightsteelblue')

ax[0].invert_yaxis()

ax[0].set_yticks(np.arange(1, len(df_scicov['Topic_sci'].tolist())+1))

labels = df_allcov['Topic_sci'].tolist()
labels = ['\n'.join(wrap(l, 21)) for l in labels]
ax[0].set_yticklabels(labels)

ax[0].set_xlim([0,0.6])

vals = ax[0].get_xticks()
ax[0].set_xticklabels(['{:.0%}'.format(x) for x in vals])

ax[0].set_xlabel('Incidence of topics in scientific literature')

ax[0].set_axisbelow(True)
ax[0].grid(color='lightgray', linewidth=1, axis='x')


# Media analysis

ax[1].barh(np.arange(1, len(df_allcov)+1, 1), width=df_allcov['Average'], height=0.7,
           color='lightsteelblue')

aus = ax[1].plot(df_allcov['Austria'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='green',
                 marker='D')
ax[1].plot(df_allcov['Denmark'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='red',
           marker='D')
ax[1].plot(df_allcov['France'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='blue',
           marker='D')
ax[1].plot(df_allcov['Italy'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='green',
           marker='o')
ax[1].plot(df_allcov['Ireland'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='red',
           marker='o')
ax[1].plot(df_allcov['Germany'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='blue',
           marker='o')
ax[1].plot(df_allcov['Norway'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='green',
           marker='x')
ax[1].plot(df_allcov['Switzerland'], np.arange(1, len(df_allcov)+1, 1), linestyle='',
           color='red', marker='x')

ax[1].set_xlim([0,0.6])

vals = ax[1].get_xticks()
ax[1].set_xticklabels(['{:,.0%}'.format(x) for x in vals])

ax[1].set_xlabel('Incidence of topics in newspaper headlines')

ax[1].legend(labels=['Austria', 'Denmark', 'France', 'Italy', 'Ireland',
                     'Germany', 'Norway', 'Switzerland',
                     'Average'], loc='lower right')

ax[1].set_axisbelow(True)
ax[1].grid(color='lightgray', linewidth=1, axis='x')


# All figure

plt.tight_layout()

fig.savefig(os.path.join(
    dir_output, 'Scientific_media_analysis')+'.png', dpi=600)



# %% Plotting chart 2

# Sorting df_allcov based on the absolute difference on incidence
df_allcov = df_allcov.sort_values(by='Abs_value_difference')


## Plotting

fig, ax = plt.subplots(1, 1, dpi=600, sharey=True,
                       figsize=(cm2inch(12.0), cm2inch(16.0)))

plt.barh(df_allcov['Topic'], abs(df_allcov['Value_difference']), color=(
    df_allcov['Value_difference'] > 0).map({True: 'orange', False: 'navy'}))

ax.set_axisbelow(True)
ax.grid(color='lightgray', linewidth=1, axis='both')

ax.set_xlim([0, 0.25])

labels = df_allcov['Topic_sci'].tolist()
labels = ['\n'.join(wrap(l, 21)) for l in labels]
ax.set_yticklabels(labels)

vals = ax.get_xticks()
ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])

ax.set_xlabel(
    'Absolute difference between incidence of topics in media\nand scientific literature')

custom_lines = [Patch(facecolor='orange', edgecolor='orange',
                         label='Color Patch'),
                Patch(facecolor='navy', edgecolor='navy',
                                         label='Color Patch')]

ax.legend(custom_lines, [
          'Higher attention in\nmedia', 'Higher attention in\nscientific literature'], loc="lower right")

plt.tight_layout()

fig.savefig(os.path.join(
    dir_output, 'Difference_coverage_scientific_media_values')+'.png', dpi=600)
