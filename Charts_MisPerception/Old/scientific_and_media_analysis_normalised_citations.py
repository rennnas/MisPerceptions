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
    os.getcwd()), 'Data', 'Input', 'PublishPerish')
file_topics_list = os.path.join(dir_input_sci, "List_topics.xlsx")
file_scigeneral = os.path.join(dir_input_sci, "General.csv")

# Media analysis
dir_input_media = os.path.join(os.path.dirname(
    os.getcwd()), 'Data', 'Input', 'MediaCoverageTitles')
file_media_coverage = os.path.join(dir_input_media, "media_coverage.xlsx")

# Outputs
dir_output = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Output')


# %% Read files

# Scientific analysis

# Read the list of topics
df_topiclist = pd.read_excel(file_topics_list, 'Sheet1')

# Read the general results
df_scigeneral = pd.read_csv(file_scigeneral)

# Create a list for reading the results
list_scilit_cit = []
list_topic_median = []

# For cycle across topics
for index, row in df_topiclist.iterrows():
    file_name = os.path.join(dir_input_sci, 'Topic_' +
                             str(row['Topic number']) + '.csv')
    df_topic = pd.read_csv(file_name)
    list_scilit_cit.append(df_topic['Cites'].values)
    list_topic_median.append(df_topic['Cites'].median())

# Create a new dataframe



df_scicov = df_topiclist


# Media coverage

# Read the excel file
df_mediacov = pd.read_excel(file_media_coverage, "Percentage", nrows=18)


#%% Normalise values of citations in scientific analysis

# Calculate median from general database
median_scigeneral = df_scigeneral['Cites'].median()

# Update values with median
list_topic_median_norm = []
ii = 0

for aaa in list_topic_median:
    list_topic_median_norm.append(aaa/median_scigeneral)
    ii = ii+1
                
# %% Sort topics

# Scientific analysis

# Create a column with arrays
df_scicov['Sci_cit_dis'] = list_scilit_cit

# Sort values by median
df_scicov['Sci_median'] = list_topic_median
df_scicov['Sci_median_norm'] = list_topic_median_norm
df_scicov = df_scicov.sort_values(by=['Sci_median'], ascending=False)

# Create a column with points
df_scicov['Sci_points'] = range(len(df_scicov), 0, -1)


# Media coverage

# Sort topics based on their mean value and calculate the max
# df_mediacov['All'] = df_mediacov.iloc[:, 1:].mean(axis=1)
df_mediacov = df_mediacov.sort_values(by=['Weighted_average'], ascending=False)
max_val = df_mediacov.iloc[:, 1:].max().max()

# Create a column with points
df_mediacov['Media_points'] = range(len(df_mediacov), 0, -1)


# Overall rank

# Merge the two rankings and sort based on distance between rankings
df_allcov = pd.concat([df_scicov, df_mediacov], axis=1)
df_allcov['Rank_distance'] = abs(
    df_allcov['Sci_points']-df_allcov['Media_points'])
df_allcov = df_allcov.sort_values(by=['Rank_distance'], ascending=False)


# %% Settings for the plots

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


# %% Plotting chart 1

fig, ax = plt.subplots(1, 2, dpi=600, sharey=True,
                       figsize=(cm2inch(16.0), cm2inch(12.0)))


# Scientific analysis

ax[0].barh(np.arange(1, len(df_allcov)+1, 1), width=df_allcov['Sci_median_norm'], height=0.7,
            color='lightsteelblue')

ax[0].invert_yaxis()

ax[0].set_yticks(np.arange(1, len(df_scicov['Topic_sci'].tolist())+1))
# ax[0].yaxis.set_label_position("right")

labels = df_allcov['Topic_sci'].tolist()
labels = ['\n'.join(wrap(l, 21)) for l in labels]
ax[0].set_yticklabels(labels)

ax[0].set_xlim([0,0.7])

vals = ax[0].get_xticks()
ax[0].set_xticklabels(['{:.0%}'.format(x) for x in vals])

ax[0].set_xlabel('Median of citations per topic / medians of general citations', fontsize=6)

ax[0].set_axisbelow(True)
ax[0].grid(color='lightgray', linewidth=1, axis='x')


# Media analysis

ax[1].barh(np.arange(1, len(df_allcov)+1, 1), width=df_allcov['Weighted_average'], height=0.7,
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

ax[1].set_xlim([0,0.7])

vals = ax[1].get_xticks()
ax[1].set_xticklabels(['{:,.0%}'.format(x) for x in vals])

ax[1].set_xlabel('Incidence of topics in newspaper headlines', fontsize=6)

ax[1].legend(labels=['Austria', 'Denmark', 'France', 'Italy', 'Ireland',
                      'Germany', 'Norway', 'Switzerland',
                      'Weighted average'], loc='lower right')

ax[1].set_axisbelow(True)
ax[1].grid(color='lightgray', linewidth=1, axis='x')


# All figure

plt.tight_layout()

fig.savefig(os.path.join(dir_output,'Media_coverage_titles')+'.png',dpi=600)


# %% Plotting chart 2

fig, ax = plt.subplots(1, 2, dpi=600, sharey=True,
                       figsize=(cm2inch(16.0), cm2inch(12.0)))
