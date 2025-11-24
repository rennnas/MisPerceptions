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
import matplotlib.ticker as plticker

from openpyxl import load_workbook
from textwrap import wrap


params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)


def cm2inch(cm):
    inc = cm / 2.54
    return inc


# %% Directories and files

# Media time analysis
dir_input_time = os.path.join(os.path.dirname(
    os.getcwd()), 'Data', 'Input', 'MediaCoverageTime')

# Outputs
dir_output = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Output','TimeAnalysis')


# %% Read files

time_coverage = []
countries = []

for file in os.listdir(dir_input_time):
    filename = os.fsdecode(file)
    countries.append(filename[:-4])
    time_coverage.append(pd.read_csv(os.path.join(dir_input_time, filename)))


# %% Organise files by month

month_coverage = []

for ii in time_coverage:
    df_cov = ii.drop(['ratio'], axis=1)
    df_cov.index = pd.to_datetime(df_cov['date'], format='%Y-%m-%d %H:%M:%S')
    df_cov['Year'] = df_cov.index.year
    df_cov['Month'] = df_cov.index.month
    df_cov = df_cov.groupby(by=['Month', 'Year'], as_index=False).sum()
    df_cov = df_cov.sort_values(by=['Year', 'Month'])
    df_cov['Month-Year'] = df_cov['Month'].astype(
        str) + '/' + df_cov['Year'].astype(str)
    df_cov['Percentage'] = df_cov['count']/df_cov['total_count']

    # Drop years from 2010 to 2013
    df_cov = df_cov[df_cov['Year'] > 2014]

    month_coverage.append(df_cov)


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
matplotlib.rcParams['lines.linewidth'] = line_width


# %% Saving output

ii = 0

for country_name in countries:

    month_coverage[ii].to_excel(os.path.join(
        dir_output, 'Monthly_media_coverage_' + country_name + '_.xlsx'), index=False)

    ii = ii+1

# %% Plotting chart 1

fig, ax = plt.subplots(4, 2, dpi=600, sharex=True, sharey=True,
                       figsize=(cm2inch(16.0), cm2inch(16.0)))

cr = 0
cc = 0

ii = 0

# Time analysis
for country_name in countries:

    ax[cr, cc].bar(month_coverage[ii]['Month-Year'],
                   month_coverage[ii]['Percentage'])

    ax[cr, cc].set_ylim(0, 0.06)

    vals = ax[cr, cc].get_yticks()
    ax[cr, cc].set_yticklabels(['{:.1%}'.format(x) for x in vals])

    # this locator puts ticks at regular intervals
    loc = plticker.MultipleLocator(base=12.0)
    ax[cr, cc].xaxis.set_major_locator(loc)
    ax[cr, cc].tick_params(axis='x', rotation=90)

    ax[cr, cc].set_title(country_name)

    ax[cr,cc].set_axisbelow(True)
    ax[cr,cc].grid(color='lightgray', linewidth=1, axis='both')

    cr = cr + 1
    if cr == 4:
        cr = 0
        cc = 1

    ii = ii+1

fig.supxlabel("Months and years", fontsize=labels_size)
fig.supylabel('Incidence of wind energy in newspapers', fontsize=labels_size)


# All figure
plt.tight_layout()
fig.savefig(os.path.join(dir_output, 'Media_coverage_time')+'.png', dpi=600)
