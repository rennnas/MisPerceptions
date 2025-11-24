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
    dir_input_sci, "Incidence_scientific_literature.xlsx")

# Media coverage analysis
dir_input_media = os.path.join(os.path.dirname(
    os.getcwd()), 'Data', 'Input', 'MediaCoverageTitles')
file_media_coverage = os.path.join(dir_input_media, "media_coverage.xlsx")

# Social media coverage analysis
dir_input_socmedia = os.path.join(os.path.dirname(
    os.getcwd()), 'Data', 'Input', 'SocialMediaTopics')
file_socmedia_coverage = os.path.join(
    dir_input_socmedia, "social_media_coverage.xlsx")

# Outputs
dir_output = os.path.join(os.path.dirname(
    os.getcwd()), 'Data', 'Output', 'Science-media_comparison')


# %% Read files


# Scientific analysis

# Read the list of topics
df_topiclist = pd.read_excel(file_topics_list, 'Sheet1')

# Read the results
df_scicov = pd.read_excel(file_sci_incidence, 'Sheet1')


# Media coverage

# Read the excel file and insert an appendix for media
df_mediacov = pd.read_excel(file_media_coverage, "Percentage", nrows=18)
df_mediacov.columns = df_mediacov.columns + '_media'


# Social media coverage

# Read the excel file and insert an appendix for socmedia
df_socmediacov = pd.read_excel(file_socmedia_coverage, "Percentage", nrows=18)
df_socmediacov.columns = df_socmediacov.columns + '_socmedia'


# %% Normalise number of articles in scientific analysis

# Calculate median from general database
df_scicov['Sci_incidence'] = df_scicov['Number of articles'] / \
    df_scicov['Number of articles'].iloc[-1]

# Drop the last row
df_scicov = df_scicov.drop(df_scicov.tail(1).index)


# %% Handle rankings

# Merge the two rankings
df_allcov = pd.concat([df_scicov, df_mediacov, df_socmediacov], axis=1)

# Calculate difference and absolute difference between incidences
df_allcov['Difference_media_science'] = df_allcov['Average_media'] - \
    df_allcov['Sci_incidence']
df_allcov['Difference_media_socmedia'] = df_allcov['Average_media'] - \
    df_allcov['Average_socmedia']
df_allcov['Difference_science_socmedia'] = df_allcov['Sci_incidence'] - \
    df_allcov['Average_socmedia']
    
    
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



# %% Plotting chart 2 - Scientific literature VS media

# Sorting df_allcov based on the absolute difference on incidence
df_allcov = df_allcov.sort_values(by='Difference_media_science')


# Plotting

fig, ax = plt.subplots(1, 1, dpi=600, sharey=True,
                        figsize=(cm2inch(12.0), cm2inch(16.0)))

plt.barh(df_allcov['Topic_sci'], df_allcov['Difference_media_science'], color='navy')

ax.set_axisbelow(True)
ax.grid(color='lightgray', linewidth=1, axis='both')

ax.set_xlim([-0.25, 0.25])
ax.set_ylim([-2.4, 17])


labels = df_allcov['Topic_sci'].tolist()
labels = ['\n'.join(wrap(l, 21)) for l in labels]
ax.set_yticklabels(labels)

vals = abs(ax.get_xticks())
ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])

ax.annotate("", xy=(0, -2.0), xytext=(0.2, -2.0),
            arrowprops=dict(arrowstyle="<-", color='green', lw=3))
ax.annotate("", xy=(-0.2, -2.0), xytext=(0, -2.0),
            arrowprops=dict(arrowstyle="->", color='purple', lw=3))


ax.text(0.055, -1.5, 'Higher incidence in\n print media',
        fontsize=labels_size, color='green',multialignment='center')
ax.text(-0.23, -1.5, 'Higher incidence in\nscientific literature',
        fontsize=labels_size, color='purple',multialignment='center')
# ax.arrow(-0.2,-1,0.4,0, shape='full',head_width=0.2,head_length=0.01)

ax.set_xlabel(
    'Difference between incidence of topics in scientific \nliterature and in print media')

custom_lines = [Patch(facecolor='orange', edgecolor='orange',
                      label='Color Patch'),
                Patch(facecolor='navy', edgecolor='navy',
                      label='Color Patch')]

ax.axvspan(0,1,color='green',alpha=0.1,zorder=0)
ax.axvspan(-1,0,color='purple',alpha=0.1,zorder=0)

plt.tight_layout()

fig.savefig(os.path.join(
    dir_output, 'Difference_coverage_scientific_media_values')+'.png', dpi=600)



# %% Plotting chart 3 - Scientific literature VS social media

# Sorting df_allcov based on the absolute difference in incidence (largest to smallest)
df_allcov = df_allcov.sort_values(by='Difference_science_socmedia', ascending=False)

# Plotting
fig, ax = plt.subplots(1, 1, dpi=600, figsize=(cm2inch(12.0), cm2inch(16.0)))

# Plot only scientific literature data (navy blue bars)
ax.barh(df_allcov['Topic_sci'], -df_allcov['Difference_science_socmedia'], color='navy', label='Scientific literature')

ax.set_axisbelow(True)
ax.grid(color='lightgray', linewidth=1, axis='both')

# Set x-axis limits
ax.set_xlim([-0.25, 0.25])

# Adjust ylim to ensure consistent spacing similar to Plot 2
ax.set_ylim([-2.4, 17])  # Adjust ylim for consistent spacing at bottom and top

# Adjust spacing for y-axis tick labels
labels = df_allcov['Topic_sci'].tolist()
labels = ['\n'.join(wrap(l, 21)) for l in labels]
ax.set_yticklabels(labels, fontsize=8)

# Set x-tick labels formatting
vals = abs(ax.get_xticks())
ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals], fontsize=8)

# Add annotations and arrows with correct colors
ax.annotate("", xy=(0, -2.0), xytext=(0.2, -2.0),
            arrowprops=dict(arrowstyle="<-", color='green', lw=3))
ax.annotate("", xy=(-0.2, -2.0), xytext=(0, -2.0),
            arrowprops=dict(arrowstyle="->", color='purple', lw=3))

# Add text labels with correct colors
ax.text(0.055, -1.5, 'Higher incidence in\nsocial media',
        fontsize=8, color='green', multialignment='center')
ax.text(-0.23, -1.5, 'Higher incidence in\nscientific literature',
        fontsize=8, color='purple', multialignment='center')

# Set x-axis label
ax.set_xlabel(
    'Difference between incidence of topics in scientific \nliterature and in social media',
    fontsize=8)

# Highlight the areas with correct colors
ax.axvspan(0, 1, color='green', alpha=0.1, zorder=0)
ax.axvspan(-1, 0, color='purple', alpha=0.1, zorder=0)

plt.tight_layout()

# Save the figure
fig.savefig(os.path.join(
    dir_output, 'Difference_coverage_science_socmedia_values')+'.png', dpi=600)