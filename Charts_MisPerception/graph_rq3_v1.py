# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 11:56:58 2024

@author: magal
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 11:51:52 2024

@author: magal
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from openpyxl import load_workbook

# Set up plotting parameters
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

def cm2inch(cm):
    return cm / 2.54

# User inputs
countries = ['Austria', 'Denmark', 'Germany', 'Norway']

# Directories and files
dir_input_time_media = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Input', 'MediaCoverageTime')
dir_input_time_socmedia = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Input', 'SocialMediaTime')
dir_output = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Output', 'TimeAnalysis')

# Read files
time_coverage_media = []
time_coverage_socmedia = []

for country_name in countries:
    file = country_name + '.csv'
    filename = os.fsdecode(file)
    time_coverage_media.append(pd.read_csv(os.path.join(dir_input_time_media, filename), sep=',', lineterminator='\n'))
    time_coverage_socmedia.append(pd.read_csv(os.path.join(dir_input_time_socmedia, filename), low_memory=False))

# Newspapers per country per year
numb_news_year = pd.read_excel(os.path.join(dir_input_time_media, 'newspapers_year_country.xlsx'))

# Global Twitter users per year
numb_twitter_users = pd.read_excel(os.path.join(dir_input_time_socmedia, 'twitter_users.xlsx'), usecols='A:F')

# Organize data per month

# Media - organize counts
month_media_coverage = []
for idx, ii in enumerate(time_coverage_media):
    df_cov = ii
    df_cov['date'] = pd.to_datetime(df_cov['date'])
    df_cov.index = df_cov['date']
    df_cov['Year'] = df_cov.index.year
    df_cov['Month'] = df_cov.index.month
    df_cov = df_cov.groupby(by=['Month', 'Year'], as_index=False).sum(numeric_only=True)
    df_cov = df_cov.sort_values(by=['Year', 'Month'])
    df_cov['Month-Year'] = df_cov['Month'].astype(str) + '/' + df_cov['Year'].astype(str)
    df_cov = df_cov[df_cov['Year'] > 2014]
    
    # Compute the ratio between count and the number of newspapers per each year and country
    spec_val = []
    for index, row in df_cov.iterrows():
        aa = row['count'] / numb_news_year[numb_news_year['Year'].values == row['Year']][countries[idx]]
        spec_val.append(aa.values[0])
    df_cov['Ratio_count/#_newspapers'] = spec_val
    month_media_coverage.append(df_cov)

# Social media - count and organize counts
month_socmedia_coverage = []
for ccc, ttt in enumerate(time_coverage_socmedia):
    tweets_current = pd.DataFrame(columns=['Timestamp', 'Year', 'Month'])
    tweets_current['Timestamp'] = pd.to_datetime(ttt['created_at'])
    tweets_current['Year'] = pd.DatetimeIndex(tweets_current['Timestamp']).year
    tweets_current['Month'] = pd.DatetimeIndex(tweets_current['Timestamp']).month
    tweets_current['Count'] = 1
    df_soccov = tweets_current
    df_soccov = df_soccov.groupby(by=['Year', 'Month'], as_index=False).sum(numeric_only=True)
    df_soccov = df_soccov.sort_values(by=['Year', 'Month'])
    df_soccov['Month-Year'] = df_soccov['Month'].astype(str) + '/' + df_soccov['Year'].astype(str)
    df_soccov = df_soccov[df_soccov['Year'] > 2014]
    
    # Compute the ratio between the count and the number of users
    spec_val = []
    for index, row in df_soccov.iterrows():
        aa = row['Count'] / numb_twitter_users[numb_twitter_users['Year'].values == row['Year']][countries[ccc]]
        spec_val.append(aa.values[0])
    df_soccov['Ratio_count/#_twitterusers'] = spec_val
    month_socmedia_coverage.append(df_soccov)

# Settings for the plots
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
matplotlib.rcParams['legend.title_fontsize'] = labels_size

# Plotting chart
ii = 0
for country_name in countries:
    fig, ax1 = plt.subplots(dpi=600, figsize=(cm2inch(19.0), cm2inch(10.0)))
    
    # Convert 'Month-Year' to datetime for plotting
    month_media_coverage[ii]['Month-Year'] = pd.to_datetime(month_media_coverage[ii]['Month-Year'], format='%m/%Y')
    month_socmedia_coverage[ii]['Month-Year'] = pd.to_datetime(month_socmedia_coverage[ii]['Month-Year'], format='%m/%Y')

    # Plot data
    l1 = ax1.plot(month_media_coverage[ii]['Month-Year'], month_media_coverage[ii]['Ratio_count/#_newspapers'], color='C0', label='Print Media')
    ax2 = ax1.twinx()
    l2 = ax2.plot(month_socmedia_coverage[ii]['Month-Year'], month_socmedia_coverage[ii]['Ratio_count/#_twitterusers'], color='C1', label='Tweets')
    
    ax1.set_ylabel('Number of articles / number of newspapers', color='C0')
    ax2.set_ylabel('Number of tweets / number of Twitter users', color='C1')

    # Format x-axis to show yearly ticks
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Set the x-axis limits to cover only up to the end of December 2022
    ax1.set_xlim([pd.to_datetime('2018-01-01').to_pydatetime(), pd.to_datetime('2022-12-31').to_pydatetime()])

    # Set x-axis tick labels
    ticks = [pd.to_datetime(f'{year}-01-01').to_pydatetime() for year in range(2018, 2023)]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([f'{year}' for year in range(2018, 2023)])

    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', color='C0', labelcolor='C0')
    ax2.tick_params(axis='y', color='C1', labelcolor='C1')

    # Add gridlines and format them
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax1.grid(color='lightgray', linewidth=1, axis='x')
    ax1.grid(color='lightgray', linewidth=1, axis='y')

    # Add monthly vertical grid lines
    for month in pd.date_range(start='2018-01-01', end='2023-01-01', freq='MS'):
        ax1.axvline(x=month.to_pydatetime(), color='lightgray', linestyle='--', alpha=0.5, linewidth=0.5)
    
    # Add the highlighted rectangle
    highlight_start = pd.to_datetime('2019-12-01').to_pydatetime()
    highlight_end = pd.to_datetime('2020-02-28').to_pydatetime()
    ax1.axvspan(highlight_start, highlight_end, color='red', alpha=0.2, label='Aug - Oct, 2022')


    # Centralized title
    fig.suptitle(country_name, fontsize=14, ha='center')
    
    # Legends
    print_media_patch = Line2D([0], [0], color='C0', label='Print Media')
    tweets_patch = Line2D([0], [0], color='C1', label='Tweets')
    highlight_patch = Line2D([0], [0], color='red', alpha=0.2, label='Dec, 2019 - Feb, 2020', linestyle='--')
    handles = [print_media_patch, tweets_patch, highlight_patch]

    ax1.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, ncol=3)

    # Save the figure
    fig.savefig(os.path.join(dir_output, 'Time_analysis_' + country_name + '.png'), dpi=600, bbox_inches='tight')

    ii = ii + 1