import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def cm2inch(cm):
    inc = cm / 2.54
    return inc

# Load the data
file_path = r'C:\Users\magal\OneDrive\Desktop\Charts_MisPerception\Data\Input\TM\de_tm_list.csv'
df = pd.read_csv(file_path)

# Convert 'year_month' to datetime
df['year_month'] = pd.to_datetime(df['year_month'], format='%Y-%m')

# Filter the data for the years 2015 to 2022
df = df[(df['year_month'] >= '2015-01') & (df['year_month'] <= '2022-12')]

# Drop rows with NA in 'tm'
df = df.dropna(subset=['tm'])

# Count the occurrences of each topic per month
topic_counts = df.groupby(['year_month', 'tm']).size().unstack(fill_value=0)

# Define custom colors for specific topics
custom_colors = {'bird': 'green', 'citizen': 'purple'}

# Plotting
plt.figure(figsize=(cm2inch(19.0), cm2inch(10.0)), dpi=1200)  # Increase the DPI for higher resolution
ax = plt.gca()

for topic in topic_counts.columns:
    color = custom_colors.get(topic, None)  # Use custom color if defined, otherwise default
    ax.plot(topic_counts.index, topic_counts[topic], label=topic, linewidth=0.5, alpha=0.7, color=color)

# Formatting the plot
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.xlabel('Year-Month')
plt.ylabel('Incidence of Topics')
plt.title('Incidence of Topics Over Time (2015-2022)')
plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save the plot with higher resolution
output_dir = r'C:\Users\magal\OneDrive\Desktop\Charts_MisPerception\Data\Output\TM'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, 'Topics_Incidence_Over_Time.png'), dpi=1200)
plt.show()

