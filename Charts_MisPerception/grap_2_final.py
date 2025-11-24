# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:55:40 2024

@author: magal
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from textwrap import wrap

def cm2inch(cm):
    return cm / 2.54

def get_flag_image(country):
    img_path = f'flags/{country}.png'
    img = Image.open(img_path).convert('RGBA')
    if img.getbands() == ('R', 'G', 'B', 'A'):
        img = img.convert('RGB')
    return OffsetImage(img, zoom=0.05, resample=Image.BOX)

def add_flag(ax, country, x, y, dx=0, dy=0):
    imagebox = get_flag_image(country)
    ab = AnnotationBbox(imagebox, (x + dx, y + dy),
                        frameon=False, pad=0, xycoords='data', boxcoords="offset points",
                        bboxprops=dict(edgecolor='none'), zorder=10)
    ax.add_artist(ab)

# Data setup
df_sci_data = {
    'Topic': ['Birds and bats strike', 'Visual impact', 'Public participation', 'Long permitting procedures'],
    'Sci_incidence': [0.032, 0.071, 0.107, 0.008]
}
df_sci = pd.DataFrame(df_sci_data).set_index('Topic')

media_data = {
    'Topic': ['Birds and bats strike', 'Visual impact', 'Public participation', 'Long permitting procedures'],
    'Austria': [0.08, 0.0267, 0.3067, 0.12],
    'Denmark': [0.0879, 0.1081, 0.4414, 0.0018],
    'France': [0.0592, 0.1895, 0.1946, 0.00],
    'Germany': [0.0940, 0.0302, 0.6142, 0.1089],
    'Ireland': [0.2536, 0.0797, 0.0072, 0.0072],
    'Italy': [0.0263, 0.2193, 0.0614, 0.0614],
    'Norway': [0.3088, 0.0735, 0.0000, 0.0588],
    'Switzerland': [0.1720, 0.2043, 0.1398, 0.0538],
    'Average': [0.14, 0.12, 0.22, 0.05]
}
df_media = pd.DataFrame(media_data).set_index('Topic')

social_media_data = {
    'Topic': ['Birds and bats strike', 'Visual impact', 'Public participation', 'Long permitting procedures'],
    'Austria': [0.1723, 0.1799, 0.3098, 0.1125],
    'Denmark': [0.1127, 0.1062, 0.2756, 0.0046],
    'France': [0.0812, 0.4331, 0.1390, 0.0009],
    'Germany': [0.1475, 0.1162, 0.1341, 0.1590],
    'Ireland': [0.2573, 0.0720, 0.0125, 0.0104],
    'Italy': [0.0464, 0.2359, 0.0573, 0.0724],
    'Norway': [0.4951, 0.1152, 0.0011, 0.0251],
    'Switzerland': [0.1061, 0.2681, 0.1143, 0.0362],
    'Average': [0.177, 0.191, 0.130, 0.053]
}
df_socmedia = pd.DataFrame(social_media_data).set_index('Topic')

# Define colors and markers
lightsteelblue = '#B0C4DE'
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown']
countries = ['Austria', 'Denmark', 'France', 'Germany', 'Ireland', 'Italy', 'Norway', 'Switzerland']
markersize = 30

# Define font sizes and figure size
legend_fontsize = 30
xlabel_fontsize = 32
tick_label_fontsize = 28
fig_width_cm = 70.0
fig_height_cm = 60.0

# Create figure and axes
fig, ax = plt.subplots(1, 3, dpi=300, figsize=(cm2inch(fig_width_cm), cm2inch(fig_height_cm)))

# Plot 1: Scientific Analysis
ax[0].barh(np.arange(len(df_sci)), width=df_sci['Sci_incidence'], height=0.7, color='thistle', label='Scientific literature')
ax[0].invert_yaxis()
ax[0].set_yticks(np.arange(len(df_sci)))
labels = df_sci.index.tolist()
labels = ['\n'.join(wrap(l, 30)) for l in labels]
labels = [label if label != 'Long permitting procedures' else 'Long permitting\nprocedures' for label in labels]
ax[0].set_yticklabels(labels, fontsize=40)
ax[0].set_xlim([0, 0.65])
vals = ax[0].get_xticks()
ax[0].set_xticklabels(['{:.0%}'.format(x) for x in vals], fontsize=tick_label_fontsize)
ax[0].set_xlabel('Incidence of topics in\nscientific literature', fontsize=xlabel_fontsize)
ax[0].set_axisbelow(True)
ax[0].grid(color='lightgray', linewidth=1, axis='both')

# Plot 2: Media Coverage
ax[1].barh(np.arange(len(df_media)), width=df_media['Average'], height=0.7, color=lightsteelblue, label='Average on countries')
for i, country in enumerate(countries):
    for j in range(len(df_media)):
        # Adjust flag positions for specific countries/topics
        dx, dy = 0, 0  # Default offsets
        if country == 'France' and df_media.index[j] == 'Birds and bats strike':
            dy = -0.15
        if country == 'Austria' and df_media.index[j] == 'Birds and bats strike':
                dy = -0.30
        if country == 'Denmark' and df_media.index[j] == 'Birds and bats strike':
                dy = 0.15
        if country == 'France' and df_media.index[j] == 'Visual impact':
             dy = -0.15
        if country == 'Austria' and df_media.index[j] == 'Visual impact':
                  dy = -0.15
        if country == 'Ireland' and df_media.index[j] == 'Visual impact':
                dy = -0.15
        if country == 'Norway' and df_media.index[j] == 'Visual impact':
                dy = 0.15
        if country == 'Switzerland' and df_media.index[j] == 'Visual impact':
                dy = 0.15
        if country == 'Ireland' and df_media.index[j] == 'Public participation':
                dy = -0.15
        if country == 'Ireland' and df_media.index[j] == 'Long permitting procedures':
                dy = -0.15
        if country == 'Austria' and df_media.index[j] == 'Long permitting procedures':
                dy = -0.15
        if country == 'Norway' and df_media.index[j] == 'Long permitting procedures':
                dy = 0.15
        if country == 'Denmark' and df_media.index[j] == 'Long permitting procedures':
                dy = 0.15
        if country == 'Switzerland' and df_media.index[j] == 'Long permitting procedures':
                dy = -0.15
        add_flag(ax[1], country, df_media[country][j], j, dx, dy)

ax[1].invert_yaxis()
ax[1].set_yticks([])
ax[1].set_xlim([0, 0.65])
vals = ax[1].get_xticks()
ax[1].set_xticklabels(['{:.0%}'.format(x) for x in vals], fontsize=tick_label_fontsize)
ax[1].set_xlabel('Incidence of topics in\nnewspaper headlines', fontsize=xlabel_fontsize)
ax[1].set_axisbelow(True)
ax[1].grid(color='lightgray', linewidth=1, axis='both')

# Plot 3: Social Media Coverage
ax[2].barh(np.arange(len(df_socmedia)), width=df_socmedia['Average'], height=0.7, color=lightsteelblue, label='Average on countries')
for i, country in enumerate(countries):
    for j in range(len(df_socmedia)):
        # Adjust flag positions for specific countries/topics
        dx, dy = 0, 0  # Default offsets
        if country == 'Switzerland' and df_socmedia.index[j] == 'Birds and bats strike':
                dy = 0.15  # Move the flag up
        if country == 'Austria' and df_socmedia.index[j] == 'Birds and bats strike':
                dy = 0.15  # Move the flag up        
        if country == 'Italy' and df_socmedia.index[j] == 'Birds and bats strike':
                dy = 0.15  # Move the flag up
        if country == 'Denmark' and df_socmedia.index[j] == 'Birds and bats strike':
                dy = -0.15  # Move the flag up
        if country == 'Austria' and df_socmedia.index[j] == 'Visual impact':
            dy = 0.15  # Move the flag up
        if country == 'Norway' and df_socmedia.index[j] == 'Visual impact':
                dy = -0.15  # Move the flag up
        if country == 'Switzerland' and df_socmedia.index[j] == 'Visual impact':
                dy = -0.15  # Move the flag up
        if country == 'Denmark' and df_socmedia.index[j] == 'Visual impact':
                dy = 0.15  # Move the flag up
        if country == 'Switzerland' and df_socmedia.index[j] == 'Public participation':
            dy = 0.15  # Move the flag down
        if country == 'Denmark' and df_socmedia.index[j] == 'Public participation':
            dy = 0.15  # Move the flag down
        if country == 'France' and df_socmedia.index[j] == 'Public participation':
            dy = -0.15  # Move the flag down
        if country == 'Ireland' and df_socmedia.index[j] == 'Public participation':
                dy = -0.15  # Move the flag down        
        if country == 'Ireland' and df_socmedia.index[j] == 'Long permitting procedures':
                        dy = -0.30  # Move the flag down
        if country == 'Norway' and df_socmedia.index[j] == 'Long permitting procedures':
                        dy = 0.30  # Move the flag down  
        if country == 'Switzerland' and df_socmedia.index[j] == 'Long permitting procedures':
                dy = -0.15  # Move the flag down  
        if country == 'France' and df_socmedia.index[j] == 'Long permitting procedures':
                dy = 0.15  # Move the flag down
        if country == 'Austria' and df_socmedia.index[j] == 'Long permitting procedures':
                        dy = 0.15  # Move the flag down  
        add_flag(ax[2], country, df_socmedia[country][j], j, dx, dy)

ax[2].invert_yaxis()
ax[2].set_yticks([])
ax[2].set_xlim([0, 0.65])
vals = ax[2].get_xticks()
ax[2].set_xticklabels(['{:.0%}'.format(x) for x in vals], fontsize=tick_label_fontsize)
ax[2].set_xlabel('Incidence of topics in\ntweets', fontsize=xlabel_fontsize)
ax[2].set_axisbelow(True)
ax[2].grid(color='lightgray', linewidth=1, axis='both')

# Legend for the three plots
custom_lines = [Line2D([0], [0], color='thistle', lw=4, label='Scientific literature'),
                Line2D([0], [0], color=lightsteelblue, lw=4, label='Average on countries')] + \
               [Line2D([0], [0], color='black', marker='o', markersize=markersize, linestyle='', label=countries[i]) for i in range(len(countries))]

fig.legend(custom_lines, ['Scientific literature', 'Average on countries'] + countries, loc='upper center', ncol=5, fontsize=legend_fontsize, frameon=False)

# Adjust layout
plt.subplots_adjust(left=0.03, bottom=0.05, right=0.97, top=0.85, wspace=0.1, hspace=0.3)

# Save and show
plt.savefig('Scientific_Media_and_Social_Media_coverage_plots_updated_large.png', bbox_inches="tight")
plt.show()
