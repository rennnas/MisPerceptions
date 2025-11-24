# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:50:41 2024

@author: magal
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from textwrap import wrap

# Configurações
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)

def cm2inch(cm):
    return cm / 2.54

# DataFrame com os novos valores para análise científica
data = {
    'Topic': ['Birds and bats strike', 'Visual impact', 'Public participation', 'Long permitting procedures'],
    'Sci_incidence': [0.032, 0.071, 0.107, 0.008]
}
df_allcov = pd.DataFrame(data)
df_allcov = df_allcov.set_index('Topic')

# Configurações para os gráficos
labels_size = 8
ticklabels_size = 8
markers_size = 4.5
line_width = 2
plt.rcParams['axes.labelsize'] = labels_size
plt.rcParams['axes.titlesize'] = labels_size
plt.rcParams['xtick.labelsize'] = ticklabels_size
plt.rcParams['ytick.labelsize'] = ticklabels_size
plt.rcParams['legend.fontsize'] = labels_size
plt.rcParams['lines.markersize'] = markers_size
plt.rcParams['lines.linewidth'] = line_width

# Plotar gráficos
fig, ax = plt.subplots(1, 1, dpi=600, figsize=(cm2inch(19.0), cm2inch(16.0)))

# Análise Científica
ax.barh(np.arange(len(df_allcov)), width=df_allcov['Sci_incidence'], height=0.7, color='thistle', label='Scientific community')
ax.invert_yaxis()
ax.set_yticks(np.arange(len(df_allcov)))
labels = df_allcov.index.tolist()
labels = ['\n'.join(wrap(l, 21)) for l in labels]
ax.set_yticklabels(labels)
ax.set_xlim([0, 0.5])
vals = ax.get_xticks()
ax.set_xticklabels(['{:.0%}'.format(x) for x in vals])
ax.set_xlabel('Incidence of topics in\nscientific literature')
ax.set_axisbelow(True)
ax.grid(color='lightgray', linewidth=1, axis='both')

# Salvando o gráfico
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.savefig('Scientific_analysis_plot.png', bbox_inches="tight")
plt.show()
