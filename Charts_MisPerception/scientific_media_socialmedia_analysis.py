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


# %% Plotting chart 1 - Incidence in scientific literature, newspaper headlines and tweets

fig, ax = plt.subplots(1, 3, dpi=600, sharey=True,
                       figsize=(cm2inch(19.0), cm2inch(16.0)))


# Scientific analysis

ax[0].barh(np.arange(1, len(df_allcov)+1, 1), width=df_allcov['Sci_incidence'], height=0.7,
           color='thistle', label='Scientific community')

ax[0].invert_yaxis()

ax[0].set_yticks(np.arange(1, len(df_scicov['Topic_sci'].tolist())+1))

labels = df_allcov['Topic_sci'].tolist()
labels = ['\n'.join(wrap(l, 21)) for l in labels]
ax[0].set_yticklabels(labels)

ax[0].set_xlim([0, 0.5])

vals = ax[0].get_xticks()
ax[0].set_xticklabels(['{:.0%}'.format(x) for x in vals])

ax[0].set_xlabel('Incidence of topics in\nscientific literature')

ax[0].set_axisbelow(True)
ax[0].grid(color='lightgray', linewidth=1, axis='both')


# Media analysis

ax[1].barh(np.arange(1, len(df_allcov)+1, 1), width=df_allcov['Average_media'], height=0.7,
           color='lightsteelblue', label='Average on countries')

ax[1].plot(df_allcov['Austria_media'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='green',
           marker='D', label='Austria')
ax[1].plot(df_allcov['Denmark_media'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='red',
           marker='D', label='Denmark')
ax[1].plot(df_allcov['France_media'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='blue',
           marker='D', label='France')
ax[1].plot(df_allcov['Italy_media'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='green',
           marker='o', label='Italy')
ax[1].plot(df_allcov['Ireland_media'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='red',
           marker='o', label='Ireland')
ax[1].plot(df_allcov['Germany_media'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='blue',
           marker='o', label='Germany')
ax[1].plot(df_allcov['Norway_media'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='green',
           marker='x', label='Norway')
ax[1].plot(df_allcov['Switzerland_media'], np.arange(1, len(df_allcov)+1, 1), linestyle='',
           color='red', marker='x', label='Switzerland')

ax[1].set_xlim([0, 0.5])

vals = ax[1].get_xticks()
ax[1].set_xticklabels(['{:,.0%}'.format(x) for x in vals])

ax[1].set_xlabel('Incidence of topics in\nnewspaper headlines')


ax[1].set_axisbelow(True)
ax[1].grid(color='lightgray', linewidth=1, axis='both')


# Socmedia analysis

ax[2].barh(np.arange(1, len(df_allcov)+1, 1), width=df_allcov['Average_socmedia'], height=0.7,
           color='lightsteelblue')

aus = ax[2].plot(df_allcov['Austria_socmedia'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='green',
                 marker='D')
ax[2].plot(df_allcov['Denmark_socmedia'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='red',
           marker='D')
ax[2].plot(df_allcov['France_socmedia'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='blue',
           marker='D')
ax[2].plot(df_allcov['Italy_socmedia'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='green',
           marker='o')
ax[2].plot(df_allcov['Ireland_socmedia'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='red',
           marker='o')
ax[2].plot(df_allcov['Germany_socmedia'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='blue',
           marker='o')
ax[2].plot(df_allcov['Norway_socmedia'], np.arange(1, len(df_allcov)+1, 1), linestyle='', color='green',
           marker='x')
ax[2].plot(df_allcov['Switzerland_socmedia'], np.arange(1, len(df_allcov)+1, 1), linestyle='',
           color='red', marker='x')

ax[2].set_xlim([0, 0.5])

vals = ax[2].get_xticks()
ax[2].set_xticklabels(['{:,.0%}'.format(x) for x in vals])

ax[2].set_xlabel('Incidence of topics in\ntweets')


ax[2].set_axisbelow(True)
ax[2].grid(color='lightgray', linewidth=1, axis='both')


# All figure

plt.tight_layout()

handles, labels = [
    (a + b) for a, b in zip(ax[0].get_legend_handles_labels(), ax[1].get_legend_handles_labels())]

fig.legend(handles, labels, loc='upper center',bbox_to_anchor=(0.5, 1.08),ncol=5)

fig.savefig(os.path.join(
    dir_output, 'Scientific_media_socmedia_analysis')+'.png', dpi=600,bbox_inches='tight')


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
    'Difference between incidence of topics in scientific literature \nand in print media')

custom_lines = [Patch(facecolor='orange', edgecolor='orange',
                      label='Color Patch'),
                Patch(facecolor='navy', edgecolor='navy',
                      label='Color Patch')]

ax.axvspan(0,1,color='green',alpha=0.1,zorder=0)
ax.axvspan(-1,0,color='purple',alpha=0.1,zorder=0)

plt.tight_layout()

fig.savefig(os.path.join(
    dir_output, 'Difference_coverage_scientific_media_values')+'.png', dpi=600)


# %% Plotting chart 3 - Media VS social media

# Sorting df_allcov based on the absolute difference on incidence
df_allcov = df_allcov.sort_values(by='Difference_media_socmedia')


# Plotting

fig, ax = plt.subplots(1, 1, dpi=600, sharey=True,
                        figsize=(cm2inch(12.0), cm2inch(16.0)))

plt.barh(df_allcov['Topic_sci'], df_allcov['Difference_media_socmedia'], color='navy')

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


ax.text(0.055, -1.5, 'Higher incidence in\nprint media',
        fontsize=labels_size, color='green',multialignment='center')
ax.text(-0.23, -1.5, 'Higher incidence in\nsocial media',
        fontsize=labels_size, color='purple',multialignment='center')
# ax.arrow(-0.2,-1,0.4,0, shape='full',head_width=0.2,head_length=0.01)

ax.set_xlabel(
    'Difference between incidence of topics in social media\nand incidence of topics in print media')

custom_lines = [Patch(facecolor='orange', edgecolor='orange',
                      label='Color Patch'),
                Patch(facecolor='navy', edgecolor='navy',
                      label='Color Patch')]

ax.axvspan(0,1,color='green',alpha=0.1,zorder=0)
ax.axvspan(-1,0,color='purple',alpha=0.1,zorder=0)

plt.tight_layout()

fig.savefig(os.path.join(
    dir_output, 'Difference_coverage_media_socmedia_values')+'.png', dpi=600)


# %% Plotting chart 4 - Scientific literature VS social media

# Sorting df_allcov based on the absolute difference on incidence
df_allcov = df_allcov.sort_values(by='Difference_science_socmedia')


# Plotting

fig, ax = plt.subplots(1, 1, dpi=600, sharey=True,
                        figsize=(cm2inch(12.0), cm2inch(16.0)))

plt.barh(df_allcov['Topic_sci'], df_allcov['Difference_science_socmedia'], color='navy')

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


ax.text(0.055, -1.5, 'Higher incidence in\nscientific literature',
        fontsize=labels_size, color='green',multialignment='center')
ax.text(-0.23, -1.5, 'Higher incidence in\nsocial media',
        fontsize=labels_size, color='purple',multialignment='center')
# ax.arrow(-0.2,-1,0.4,0, shape='full',head_width=0.2,head_length=0.01)

ax.set_xlabel(
    'Difference between incidence of topics in social media \nand incidence of topics in scientific literature')

custom_lines = [Patch(facecolor='orange', edgecolor='orange',
                      label='Color Patch'),
                Patch(facecolor='navy', edgecolor='navy',
                      label='Color Patch')]

ax.axvspan(0,1,color='green',alpha=0.1,zorder=0)
ax.axvspan(-1,0,color='purple',alpha=0.1,zorder=0)

plt.tight_layout()

fig.savefig(os.path.join(
    dir_output, 'Difference_coverage_science_socmedia_values')+'.png', dpi=600)