import os
print(os.getcwd())



import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as plticker
from datetime import datetime
from matplotlib.lines import Line2D

def cm2inch(cm):
    return cm / 2.54

# Directories and files for time analysis
dir_input_time_media = os.path.join(os.getcwd(), 'Data', 'Input', 'MediaCoverageTime')
dir_input_time_socmedia = os.path.join(os.getcwd(), 'Data', 'Input', 'SocialMediaTime')
dir_output = os.path.join(os.getcwd(), 'Data', 'Output', 'TimeAnalysis')

# Load data for media and social media time analysis
countries = ['Germany']
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

# Load newspapers per country per year
numb_news_year = pd.read_excel(os.path.join(dir_input_time_media,'newspapers_year_country.xlsx'))

# Load global Twitter users per year
numb_twitter_users = pd.read_excel(os.path.join(dir_input_time_socmedia,'twitter_users.xlsx'),usecols='A:F')

# Process media coverage data per month
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
    
    spec_val = []
    for index, row in df_cov.iterrows():
        aa = row['count']/numb_news_year[numb_news_year['Year'].values==row['Year']][countries[idx]]
        spec_val.append(aa.values[0])
    df_cov['Ratio_count/#_newspapers'] = spec_val
    month_media_coverage.append(df_cov)

# Process social media coverage data per month
month_socmedia_coverage = []
ccc = 0
for ttt in time_coverage_socmedia:
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

    spec_val = []
    for index, row in df_soccov.iterrows():
        aa = row['Count']/numb_twitter_users[numb_twitter_users['Year'].values==row['Year']][countries[ccc]]
        spec_val.append(aa.values[0])
    df_soccov['Ratio_count/#_twitterusers'] = spec_val

    month_socmedia_coverage.append(df_soccov)
    ccc += 1

# Load the topic incidence data for Germany
file_path_topics = r'C:\Users\magal\OneDrive\Desktop\Charts_MisPerception\Data\Input\TM\de_tm_list.csv'
df_topics = pd.read_csv(file_path_topics)
df_topics['year_month'] = pd.to_datetime(df_topics['year_month'], format='%Y-%m')
df_topics = df_topics[(df_topics['year_month'] >= '2015-01') & (df_topics['year_month'] <= '2022-12')]
df_topics = df_topics.dropna(subset=['tm'])
topic_counts = df_topics.groupby(['year_month', 'tm']).size().unstack(fill_value=0)

# Define custom colors for specific topics
custom_colors = {'bird': 'green', 'citizen': 'purple'}

# Create the combined plot for Germany
country_name = 'Germany'
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, dpi=600, gridspec_kw={'height_ratios': [1, 2, 2]}, sharex=True,
                                   figsize=(cm2inch(19.0), cm2inch(15.0)))

# Time analysis
l1 = ax1.plot(month_media_coverage[0]['Month-Year'],
              month_media_coverage[0]['Ratio_count/#_newspapers'], color='C0')
ax2_tw = ax1.twinx()
l2 = ax2_tw.plot(month_socmedia_coverage[0]['Month-Year'],
                 month_socmedia_coverage[0]['Ratio_count/#_twitterusers'], color='C1')

ax1.set_ylabel('Number of articles / number of newspapers', color='C0')
ax2_tw.set_ylabel('Number of tweets / number of Twitter users', color='C1')

loc_12 = plticker.MultipleLocator(12.0)
ax1.xaxis.set_major_locator(loc_12)
ax1.xaxis.set_minor_locator(plticker.MultipleLocator(1.0))

labels = [item.get_text() for item in ax1.get_xticklabels()]
labels[-2] = '1/2023'
ax1.set_xticklabels(labels)
ax1.tick_params(axis='x', rotation=45)

ax1.set_xlim([-3, 99])

ax1.set_axisbelow(True)
ax2_tw.set_axisbelow(True)
ax1.grid(color='lightgray', linewidth=1, axis='x')
ax1.grid(color='lightgray', linewidth=1, axis='y')

for xmin in ax1.xaxis.get_minorticklocs():
    ax1.axvline(x=xmin, lw=0.5, ls='-',
                color='lightgray', alpha=0.5, zorder=0)

ax1.tick_params(axis='y', color='C0', labelcolor='C0')
ax2_tw.tick_params(axis='y', color='C1', labelcolor='C1')
ax2_tw.spines['right'].set_color('C1')
ax2_tw.spines['left'].set_color('C0')

# Axis of the events
ax0.grid(color='lightgray', linewidth=1, axis='x')
ax0.grid(color='lightgray', linewidth=1, axis='y')

for xmin in ax1.xaxis.get_minorticklocs():
    ax0.axvline(x=xmin, lw=0.5, ls='-',
                color='lightgray', alpha=0.5, zorder=0)

ax0.set_yticks([])
ax0.set_yticklabels([])

marker_list = ['o', 'D', '*']
color_list = ['tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

handles_evs = []
mm = 0
cc = 0

for index, row in events_socmedia[0].iterrows():
    ev, = ax0.plot(row['# month'], 0.5, marker=marker_list[mm], color=color_list[cc],
                   linestyle='', label=row['Event'])
    ax0.axvline(x=row['# month'], lw=1, ls='-',
                color=color_list[cc], alpha=0.5)
    ax1.axvline(x=row['# month'], lw=1, ls='-',
                color=color_list[cc], alpha=0.5)

    handles_evs.append(ev)

    cc = cc+1
    if cc == len(color_list):
        cc = 0
        mm = mm+1

# Plot topic incidence data
ax2 = plt.subplot(313)
for topic in topic_counts.columns:
    color = custom_colors.get(topic, None)
    ax2.plot(topic_counts.index, topic_counts[topic], label=topic, linewidth=0.5, alpha=0.7, color=color)

# Formatting the plot
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_minor_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks
