
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as plticker
import matplotlib.patches as mpatches

from openpyxl import load_workbook
from textwrap import wrap
from matplotlib.lines import Line2D
from datetime import date

params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

def cm2inch(cm):
    inc = cm / 2.54
    return inc

# %% User inputs

# countries = ['Austria', 'Denmark', 'France', 'Germany',
#              'Ireland', 'Italy', 'Norway', 'Switzerland']

countries = ['Austria', 'Denmark', 'Germany', 'Norway']

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
events_socmedia = []

for country_name in countries:
    file = country_name + '.csv'
    filename = os.fsdecode(file)
    file_events = country_name + '_events.xlsx'
    filename_events = os.fsdecode(file_events)
    time_coverage_media.append(pd.read_csv(
        os.path.join(dir_input_time_media, filename), sep=',', lineterminator='\n'))
    time_coverage_socmedia.append(pd.read_csv(os.path.join(
        dir_input_time_socmedia, filename), low_memory=False))
    events_socmedia.append(pd.read_excel(os.path.join(
        dir_input_time_socmedia, filename_events)).dropna())

# Newspapers per country per year
numb_news_year = pd.read_excel(os.path.join(dir_input_time_media,'newspapers_year_country.xlsx'))

# Global Twitter users per year
numb_twitter_users = pd.read_excel(os.path.join(dir_input_time_socmedia,'twitter_users.xlsx'),usecols='A:F')

#%% Organize data per month

# Media - organise counts
month_media_coverage = []

for idx, ii in enumerate(time_coverage_media):
    df_cov = ii
    # df_cov = ii.drop(['ratio'], axis=1)
    df_cov['date'] = pd.to_datetime(df_cov['date'])  # Let pandas infer the date format
    df_cov.index = df_cov['date']  # Set the index to the 'date' column
    df_cov['Year'] = df_cov.index.year
    df_cov['Month'] = df_cov.index.month
    df_cov = df_cov.groupby(by=['Month', 'Year'],
                            as_index=False).sum(numeric_only=True)

    df_cov = df_cov.sort_values(by=['Year', 'Month'])
    df_cov['Month-Year'] = df_cov['Month'].astype(
        str) + '/' + df_cov['Year'].astype(str)

    # Drop years from 2010 to 2013
    df_cov = df_cov[df_cov['Year'] > 2014]
    
    # Compute the ratio between count and the number of newspapers per each year and country
    spec_val = []
    for index, row in df_cov.iterrows():
        aa = row['count']/numb_news_year[numb_news_year['Year'].values==row['Year']][countries[idx]]
        spec_val.append(aa.values[0])
    df_cov['Ratio_count/#_newspapers'] = spec_val
    
    month_media_coverage.append(df_cov)

# Social media - count and organise counts
month_socmedia_coverage = []

ccc = 0

for ttt in time_coverage_socmedia:
    tweets_current = pd.DataFrame(columns=['Timestamp', 'Year', 'Month'])
    tweets_current['Timestamp'] = pd.to_datetime(
        ttt['created_at'])  # Let pandas infer the date format
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

    # Compute the ratio between the count and the number of users
    spec_val = []
    for index, row in df_soccov.iterrows():
        aa = row['Count']/numb_twitter_users[numb_twitter_users['Year'].values==row['Year']][countries[ccc]]
        spec_val.append(aa.values[0])
    df_soccov['Ratio_count/#_twitterusers'] = spec_val

    month_socmedia_coverage.append(df_soccov)
    
    ccc = ccc + 1
    
# %% Saving output
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
matplotlib.rcParams['lines.linewidth'] = line_width
matplotlib.rcParams['legend.title_fontsize'] = labels_size

# %% Saving output

ii = 0

for country_name in countries:

    month_media_coverage[ii].to_excel(os.path.join(
        dir_output, 'Monthly_media_coverage_' + country_name + '.xlsx'), index=False)
    month_socmedia_coverage[ii].to_excel(os.path.join(
        dir_output, 'Monthly_social_media_coverage_' + country_name + '.xlsx'), index=False)

# %% Plotting chart 1

ii = 0

for country_name in countries:

    fig, (ax0, ax1) = plt.subplots(2, 1, dpi=600, gridspec_kw={'height_ratios': [1, 4]}, sharex=True,
                                   figsize=(cm2inch(19.0), cm2inch(10.0)))
    # Time analysis

    l1 = ax1.plot(month_media_coverage[ii]['Month-Year'],
                  month_media_coverage[ii]['Ratio_count/#_newspapers'], color='C0')
    ax2 = ax1.twinx()
    l2 = ax2.plot(month_socmedia_coverage[ii]['Month-Year'],
                  month_socmedia_coverage[ii]['Ratio_count/#_twitterusers'], color='C1')

    ax1.set_ylabel('Number of articles / number of newspapers', color='C0')
    ax2.set_ylabel('Number of tweets / number of Twitter users', color='C1')

    # this locator puts ticks at regular intervals
    loc_12 = plticker.MultipleLocator(12.0)
    ax1.xaxis.set_major_locator(loc_12)
    ax1.xaxis.set_minor_locator(plticker.MultipleLocator(1.0))

    labels = [item.get_text() for item in ax1.get_xticklabels()]
    labels[-2] = '1/2023'
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis='x', rotation=45)

    ax1.set_xlim([-3, 99])

    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax1.grid(color='lightgray', linewidth=1, axis='x')
    ax1.grid(color='lightgray', linewidth=1, axis='y')

    for xmin in ax1.xaxis.get_minorticklocs():
        ax1.axvline(x=xmin, lw=0.5, ls='-',
                    color='lightgray', alpha=0.5, zorder=0)

    # Color ybars
    ax1.tick_params(axis='y', color='C0', labelcolor='C0')
    ax2.tick_params(axis='y', color='C1', labelcolor='C1')
    ax2.spines['right'].set_color('C1')
    ax2.spines['left'].set_color('C0')

    # Axis of the events
    ax0.grid(color='lightgray', linewidth=1, axis='x')
    ax0.grid(color='lightgray', linewidth=1, axis='y')

    for xmin in ax1.xaxis.get_minorticklocs():
        ax0.axvline(x=xmin, lw=0.5, ls='-',
                    color='lightgray', alpha=0.5, zorder=0)

    ax0.yaxis.set_major_locator(plticker.MultipleLocator(1.0))

    ax0.set_axisbelow(True)

    # Add a table on top of the chart with the relevant events
    df_events = events_socmedia[ii]
    df_events['date'] = pd.to_datetime(df_events['date'])
    df_events = df_events[df_events['date'].dt.year > 2014]
    df_events['date'] = df_events['date'].dt.strftime('%m/%Y')
    df_events = df_events[df_events['date'].isin(
        month_socmedia_coverage[ii]['Month-Year'])]

    #ax0.table(cellText=[df_events['event']], colWidths=[0.1], cellLoc = 'center', loc='center')
    for i in range(len(df_events)):
        ax0.text(df_events['date'].values[i], 0.4, '\n'.join(
            wrap(df_events['event'].values[i], 15)), ha='center', va='center')

    ax0.get_yaxis().set_visible(False)
    ax0.set_ylim([0, 0.5])

    plt.subplots_adjust(left=0.11,
                        bottom=0.18,
                        right=0.89,
                        top=0.84,
                        wspace=0.05,
                        hspace=0.05)

    plt.savefig(os.path.join(dir_output, 'Monthly_coverage_' +
                country_name + '.png'), dpi=600, bbox_inches="tight")
    plt.show()
    ii = ii + 1
