import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def cm2inch(cm):
    inc = cm / 2.54
    return inc

# Load the data for topics over time
dir_input_tm = r'C:\Users\magal\OneDrive\Desktop\Charts_MisPerception\Data\Input\TM'
de_tm_list = pd.read_csv(os.path.join(dir_input_tm, 'de_tm_list.csv'))

# Format year_month as datetime
de_tm_list['year_month'] = pd.to_datetime(de_tm_list['year_month'])

# Select years 2015 to 2022
de_tm_list = de_tm_list[(de_tm_list['year_month'].dt.year >= 2015) & (de_tm_list['year_month'].dt.year <= 2022)]

# Plotting both charts together
fig, (ax0, ax1) = plt.subplots(2, 1, dpi=600, gridspec_kw={'height_ratios': [1, 4]}, sharex=True,
                               figsize=(cm2inch(19.0), cm2inch(20.0)))

# Plot 1: Time analysis from previous code
countries = ['Austria', 'Denmark', 'Germany', 'Norway']

# Directories and files
dir_input_time_media = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Input', 'MediaCoverageTime')
dir_input_time_socmedia = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Input', 'SocialMediaTime')
dir_output = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Output', 'TimeAnalysis')

# Load data
time_coverage_media = []
time_coverage_socmedia = []
events_socmedia = []
numb_news_year = pd.read_excel(os.path.join(dir_input_time_media, 'newspapers_year_country.xlsx'))
numb_twitter_users = pd.read_excel(os.path.join(dir_input_time_socmedia, 'twitter_users.xlsx'), usecols='A:F')

# Read files
for country_name in countries:
    file = country_name + '.csv'
    filename = os.fsdecode(file)
    file_events = country_name + '_events.xlsx'
    filename_events = os.fsdecode(file_events)
    time_coverage_media.append(pd.read_csv(os.path.join(dir_input_time_media, filename), sep=',', lineterminator='\n'))
    time_coverage_socmedia.append(pd.read_csv(os.path.join(dir_input_time_socmedia, filename), low_memory=False))
    events_socmedia.append(pd.read_excel(os.path.join(dir_input_time_socmedia, filename_events)).dropna())

# Organize data per month for media
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
        aa = row['count'] / numb_news_year[numb_news_year['Year'] == row['Year']][countries[idx]]
        spec_val.append(aa.values[0])
    df_cov['Ratio_count/#_newspapers'] = spec_val
    
    month_media_coverage.append(df_cov)

# Organize data per month for social media
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
        aa = row['Count'] / numb_twitter_users[numb_twitter_users['Year'] == row['Year']][countries[ccc]]
        spec_val.append(aa.values[0])
    df_soccov['Ratio_count/#_twitterusers'] = spec_val

    month_socmedia_coverage.append(df_soccov)
    
    ccc = ccc + 1

# Plot media and social media ratios
l1 = ax1.plot(month_media_coverage[0]['Month-Year'], month_media_coverage[0]['Ratio_count/#_newspapers'], color='C0')
ax2 = ax1.twinx()
l2 = ax2.plot(month_socmedia_coverage[0]['Month-Year'], month_socmedia_coverage[0]['Ratio_count/#_twitterusers'], color='C1')

ax1.set_ylabel('Number of articles / number of newspapers', color='C0')
ax2.set_ylabel('Number of tweets / number of Twitter users', color='C1')

loc_12 = plticker.MultipleLocator(12.0)
loc_1 = plticker.MultipleLocator(1.0)
ax1.xaxis.set_major_locator(loc_12)
ax1.xaxis.set_minor_locator(loc_1)

labels = [item.get_text() for item in ax1.get_xticklabels()]
labels[-2] = '1/2023'
ax1.set_xticklabels(labels)
ax1.tick_params(axis='x', rotation=45)

ax1.set_xlim([-3, 99])

ax1.set_axisbelow(True)
ax2.set_axisbelow(True)

ax1.grid(visible=True, axis='x', color='lightgray', linewidth=1, which='both')
for xmin in ax1.xaxis.get_minorticklocs():
    ax0.axvline(x=xmin, lw=0.5, ls='-', color='lightgray', alpha=0.5)
    ax1.axvline(x=xmin, lw=0.5, ls='-', color='lightgray', alpha=0.5)

ax1.grid(visible=True, axis='y', color='lightgray', linewidth=1)
ax2.grid(visible=False)
ax1.tick_params(axis='y', color='C0', labelcolor='C0')
ax2.tick_params(axis='y', color='C1', labelcolor='C1')
ax2.spines['right'].set_color('C1')
ax2.spines['left'].set_color('C0')

ax0.grid(visible=True, axis='x', color='lightgray', linewidth=1, which='both')
ax0.set_yticks([])
ax0.set_yticklabels([])

# Legends
media_patch = Line2D([0], [0], color='C0', label='Media')
socmedia_patch = Line2D([0], [0], color='C1', label='Social media')
handles = [media_patch, socmedia_patch]

ax1.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, ncol=5)
ax0.legend(handles=handles_evs, loc='upper center', bbox_to_anchor=(0.5, 2.9), fancybox=True, ncol=3)

ax0.get_legend().set_title(country_name)

# Plot 2: Topics over time
topics = de_tm_list['tm'].unique()
colors = plt.cm.tab10.colors  # Get 10 distinct colors from the 'tab10' colormap

for i, topic in enumerate(topics):
    if pd.notna(topic):
        topic_data = de_tm_list[de_tm_list['tm'] == topic]
        ax0.plot(topic_data['year_month'], topic_data['incidence'], label=topic, color=colors[i % len(colors)], alpha=0.7, lw=0.7)

ax0.legend(loc='upper left', fontsize='small')
ax0.set_ylabel('Incidence')
ax0.set_title('Topics Incidence over Time')
ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Improve resolution
plt.tight_layout()
plt.savefig(os.path.join(dir_output, 'Combined_Plots.png'), dpi=1200)
plt.show()
