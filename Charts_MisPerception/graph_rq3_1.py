# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:27:32 2024

@author: magal
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

def cm2inch(cm):
    return cm / 2.54

# User inputs
countries = ['Germany', 'Austria', 'Denmark', 'Norway']

# Directories and files
dir_input_time_media = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Input', 'MediaCoverageTime')
dir_input_time_socmedia = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Input', 'SocialMediaTime')
dir_output = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Output', 'TimeAnalysis')

# Read files
time_coverage_media = []
time_coverage_socmedia = []
events_socmedia = []

for country_name in countries:
    file = country_name + '.csv'
    file_events = country_name + '_events.xlsx'
    time_coverage_media.append(pd.read_csv(os.path.join(dir_input_time_media, file), sep=',', lineterminator='\n'))
    time_coverage_socmedia.append(pd.read_csv(os.path.join(dir_input_time_socmedia, file), low_memory=False))
    events_socmedia.append(pd.read_excel(os.path.join(dir_input_time_socmedia, file_events)).dropna())

# Newspapers per country per year
numb_news_year = pd.read_excel(os.path.join(dir_input_time_media, 'newspapers_year_country.xlsx'))

# Global Twitter users per year
numb_twitter_users = pd.read_excel(os.path.join(dir_input_time_socmedia, 'twitter_users.xlsx'), usecols='A:F')

# Organize data per month
month_media_coverage = []
month_socmedia_coverage = []

for idx, ii in enumerate(time_coverage_media):
    df_cov = ii
    df_cov['date'] = pd.to_datetime(df_cov['date'])
    df_cov.index = df_cov['date']
    df_cov['Year'] = df_cov.index.year
    df_cov['Month'] = df_cov.index.month
    df_cov = df_cov.groupby(by=['Month', 'Year'], as_index=False).sum(numeric_only=True)
    df_cov = df_cov.sort_values(by=['Year', 'Month'])
    df_cov['Month-Year'] = df_cov['Month'].astype(str) + '/' + df_cov['Year'].astype(str)
    df_cov = df_cov[df_cov['Year'] == 2022]
    
    # Filter for August, September, October
    df_cov = df_cov[df_cov['Month'].isin([8, 9, 10])]
    
    spec_val = []
    for index, row in df_cov.iterrows():
        aa = row['count'] / numb_news_year[numb_news_year['Year'].values == row['Year']][countries[idx]]
        spec_val.append(aa.values[0])
    df_cov['Ratio_count/#_newspapers'] = spec_val
    month_media_coverage.append(df_cov)

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
    df_soccov = df_soccov[df_soccov['Year'] == 2022]
    
    # Filter for August, September, October
    df_soccov = df_soccov[df_soccov['Month'].isin([8, 9, 10])]
    
    spec_val = []
    for index, row in df_soccov.iterrows():
        aa = row['Count'] / numb_twitter_users[numb_twitter_users['Year'].values == row['Year']][countries[ccc]]
        spec_val.append(aa.values[0])
    df_soccov['Ratio_count/#_twitterusers'] = spec_val
    month_socmedia_coverage.append(df_soccov)

# Plotting all charts in a single figure
fig, axs = plt.subplots(2, 2, dpi=1200, figsize=(cm2inch(32.0), cm2inch(20.0)), gridspec_kw={'height_ratios': [5, 5]})

for idx, country_name in enumerate(countries):
    ax1, ax2 = axs[idx // 2, idx % 2], axs[idx // 2, idx % 2].twinx()
    
    # Plot media coverage
    ax1.plot(month_media_coverage[idx]['Month-Year'], month_media_coverage[idx]['Ratio_count/#_newspapers'], color='C0', label='Number of articles / number of newspapers')
    ax2.plot(month_socmedia_coverage[idx]['Month-Year'], month_socmedia_coverage[idx]['Ratio_count/#_twitterusers'], color='C1', label='Number of tweets / number of Twitter users')
    
    ax1.set_ylabel('Number of articles / number of newspapers', color='C0')
    ax2.set_ylabel('Number of tweets / number of Twitter users', color='C1')
    
    # Set x-ticks
    ax1.set_xticks(['8/2022', '9/2022', '10/2022'])
    ax1.set_xticklabels(['08-2022', '09-2022', '10-2022'], rotation=45)

    # Set y-axis limits based on the country
    if country_name == 'Germany':
        ax1.set_ylim(0, 140)
        ax2.set_ylim(200, 1800)
    elif country_name == 'Austria':
        ax1.set_ylim(0, 7)
        ax2.set_ylim(0, 700)
    elif country_name == 'Norway':
        ax1.set_ylim(0, 4)
        ax2.set_ylim(0, 2500)
    elif country_name == 'Denmark':
        ax1.set_ylim(0, 45)
        ax2.set_ylim(0, 1750)
    
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax1.grid(color='lightgray', linewidth=1, axis='x')
    ax1.grid(color='lightgray', linewidth=1, axis='y')

    for xmin in ax1.xaxis.get_minorticklocs():
        ax1.axvline(x=xmin, lw=0.5, ls='-', color='lightgray', alpha=0.5, zorder=0)

    ax1.tick_params(axis='y', color='C0', labelcolor='C0')
    ax2.tick_params(axis='y', color='C1', labelcolor='C1')
    ax2.spines['right'].set_color('C1')
    ax2.spines['left'].set_color('C0')

    # Title with the country name, centered and larger font size
    axs[idx // 2, idx % 2].set_title(country_name, loc='center', fontsize=16, pad=20)

    # Remove the box around the legend
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, ncol=1, frameon=False)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fancybox=True, ncol=1, frameon=False)

# Adjust layout to fit all plots
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

# Save the figure
fig.savefig(os.path.join(dir_output, 'Time_analysis_all_countries.png'), dpi=600, bbox_inches='tight')

# Show the plot
plt.show()
