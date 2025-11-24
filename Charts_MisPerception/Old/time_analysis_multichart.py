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
import matplotlib.patches as mpatches

from openpyxl import load_workbook
from textwrap import wrap


params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)


def cm2inch(cm):
    inc = cm / 2.54
    return inc


# %% Directories and files

# Media time analysis
dir_input_time_media = os.path.join(os.path.dirname(
    os.getcwd()), 'Data', 'Input', 'MediaCoverageTime')

# Social media time analysis
dir_input_time_socmedia = os.path.join(os.path.dirname(
    os.getcwd()), 'Data', 'Input', 'SocialMediaTime')

# Outputs
dir_output = os.path.join(os.path.dirname(
    os.getcwd()), 'Data', 'Output', 'TimeAnalysis')


# %% Read files

# Media and social media
time_coverage_media = []
time_coverage_socmedia = []
countries = []

for file in os.listdir(dir_input_time_media):
    filename = os.fsdecode(file)
    countries.append(filename[:-4])
    time_coverage_media.append(pd.read_csv(
        os.path.join(dir_input_time_media, filename)))
    time_coverage_socmedia.append(pd.read_csv(os.path.join(
        dir_input_time_socmedia, filename)))


# %% Organise files by month

# Media - organise counts
month_media_coverage = []

for ii in time_coverage_media:
    df_cov = ii.drop(['ratio'], axis=1)
    df_cov.index = pd.to_datetime(df_cov['date'], format='%Y-%m-%d %H:%M:%S')
    df_cov['Year'] = df_cov.index.year
    df_cov['Month'] = df_cov.index.month
    df_cov = df_cov.groupby(by=['Month', 'Year'],
                            as_index=False).sum(numeric_only=True)
    df_cov = df_cov.sort_values(by=['Year', 'Month'])
    df_cov['Month-Year'] = df_cov['Month'].astype(
        str) + '/' + df_cov['Year'].astype(str)

    # Drop years from 2010 to 2013
    df_cov = df_cov[df_cov['Year'] > 2014]

    month_media_coverage.append(df_cov)


# Social media - count and organise counts
month_socmedia_coverage = []

for ttt in time_coverage_socmedia:
    tweets_current = pd.DataFrame(columns=['Timestamp', 'Year', 'Month'])
    tweets_current['Timestamp'] = pd.to_datetime(
        ttt['created_at'])
    tweets_current['Year'] = pd.DatetimeIndex(tweets_current['Timestamp']).year
    tweets_current['Month'] = pd.DatetimeIndex(
        tweets_current['Timestamp']).month
    tweets_current['Count'] = 1
    df_soccov = tweets_current
    df_soccov = df_soccov.groupby(
        by=['Year', 'Month'], as_index=False).sum(numeric_only=True)
    df_soccov = df_soccov.sort_values(by=['Year', 'Month'])
    df_soccov['Month-Year'] = df_soccov['Month'].astype(
        str) + '/' + df_soccov['Year'].astype(str)

    # Drop years from 2010 to 2013
    df_soccov = df_soccov[df_soccov['Year'] > 2014]

    month_socmedia_coverage.append(df_soccov)


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

    month_media_coverage[ii].to_excel(os.path.join(
        dir_output, 'Monthly_media_coverage_' + country_name + '.xlsx'), index=False)
    month_socmedia_coverage[ii].to_excel(os.path.join(
        dir_output, 'Monthly_social_media_coverage_' + country_name + '.xlsx'), index=False)
    ii = ii+1


# %% Plotting chart 1

fig, ax = plt.subplots(5, 2, dpi=600, sharex=True,
                       figsize=(cm2inch(19.0), cm2inch(20.0)))

cr = 0
cc = 0

ii = 0

# # Time analysis
for country_name in countries:

    # ax[cr, cc].bar(month_media_coverage[ii]['Month-Year'],
    #                 month_media_coverage[ii]['count'])
    # ax[cr, cc].bar(month_socmedia_coverage[ii]['Month-Year'],
    #                 -month_socmedia_coverage[ii]['Count'])

    ax1 = ax[cr, cc]
    l1 = ax1.plot(month_media_coverage[ii]['Month-Year'],
                  month_media_coverage[ii]['count'], color='C0')
    ax2 = ax1.twinx()
    l2 = ax2.plot(month_socmedia_coverage[ii]['Month-Year'][:-1],
                  month_socmedia_coverage[ii]['Count'][:-1], color='C1')
    # ax[cr, cc].set_ylim(0, 0.06)

    ax1.set_ylabel('Number of articles', color='C0')
    ax2.set_ylabel('Number of tweets', color='C1')

    # this locator puts ticks at regular intervals
    loc = plticker.MultipleLocator(base=12.0)
    ax[cr, cc].xaxis.set_major_locator(loc)
    ax[cr, cc].tick_params(axis='x', rotation=45)

    ax[cr, cc].set_title(country_name)

    ax[cr, cc].set_axisbelow(True)
    ax[cr, cc].grid(color='lightgray', linewidth=1, axis='both')

    # Color ybars
    ax1.tick_params(axis='y', color='C0', labelcolor='C0')
    ax2.tick_params(axis='y', color='C1', labelcolor='C1')
    ax2.spines['right'].set_color('C1')
    ax2.spines['left'].set_color('C0')

    cr = cr + 1
    if cr == 5:
        cr = 0
        cc = 1

    ii = ii+1

# Legend

media_patch = mpatches.Patch(color='C0', label='Media')
socmedia_patch = mpatches.Patch(color='C1', label='Social media')
fig.legend(handles=[media_patch, socmedia_patch], loc='upper center', ncol=2)

# fig.supxlabel("Months and years", fontsize=labels_size)
# fig.supylabel('Wind energy in newspapers and social media', fontsize=labels_size)


# # All figure
plt.tight_layout()
fig.savefig(os.path.join(
    dir_output, 'Media_social_media_time_analysis')+'.png', dpi=600)
