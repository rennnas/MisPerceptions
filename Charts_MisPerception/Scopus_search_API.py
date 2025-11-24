# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:29:27 2023

@author: ricca
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as plticker

from openpyxl import load_workbook
from textwrap import wrap

from pybliometrics.scopus.utils import config
from pybliometrics.scopus import ScopusSearch

params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)


def cm2inch(cm):
    inc = cm / 2.54
    return inc


# %% User inputs

research_type = 'TITLE-ABS-KEY'
common_query = '"wind farm" OR "wind energy" OR "wind power" OR "wind turbines" OR "wind turbine" OR "turbine blades"'
first_year = 2010
last_year = 2022


# %% Directories and files

# Scopus input folder
dir_input_time = os.path.join(os.path.dirname(
    os.getcwd()), 'Data', 'Input', 'Scopus')

# Scopus output folder
dir_output = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'Output', 'Scopus')

# Scopus dictionary file
scopus_dict_file = os.path.join(dir_input_time, 'Dictionary_Scopus.xlsx')

# Output file
output_excel_file = os.path.join(dir_output, 'Scopus_queries.xlsx')

# %% Read files

df_dict = pd.read_excel(scopus_dict_file, 'Keywords')


# %% Create strings for developing the queries

# We will develop a four steps analysis:
# - the overall query with all keywords
# - the query for each topic
# - the sensitivity analysis on each keyword
# - the sensitivity analysis on the marginal contribution of each keyword
# Therefore, we need four groups of queries


# QUERY 1: Overall query with all keywords

string_var = ''

for index, row in df_dict.iterrows():
    if index != 0:
        string_var = string_var + " OR "
    string_var = string_var + '"' + row['Keyword'] + '"'

string_Q1 = research_type + '((' + common_query + ') AND (' + string_var + ')) ' + \
    'AND PUBYEAR > ' + str(first_year-1) + ' AND PUBYEAR < ' + str(last_year+1)


# QUERY 2: Query for each topic

list_Q2 = []

for tt in df_dict['Topic'].unique():

    df_tt = df_dict[df_dict['Topic'] == tt].reset_index()

    string_tt = ''

    for index, row in df_tt.iterrows():
        if index != 0:
            string_tt = string_tt + " OR "
        string_tt = string_tt + '"' + row['Keyword'] + '"'

    string_tt_Q2 = research_type + '((' + common_query + ') AND (' + string_tt + ')) ' + \
        'AND PUBYEAR > ' + str(first_year-1) + \
        ' AND PUBYEAR < ' + str(last_year+1)

    list_Q2.append(string_tt_Q2)


# QUERY 3: Sensitivity analysis on each keyword

list_Q3 = []

for index, row in df_dict.iterrows():

    string_kk = '"' + row['Keyword'] + '"'

    string_tt_Q3 = research_type + '((' + common_query + ') AND (' + string_kk + ')) ' + \
        'AND PUBYEAR > ' + str(first_year-1) + \
        ' AND PUBYEAR < ' + str(last_year+1)

    list_Q3.append(string_tt_Q3)


# QUERY 4: Sensitivity analysis on the marginal contribution of each keyword

list_Q4 = []

for index, row in df_dict.iterrows():

    string_marg = ''
    ccc = 0

    for index_in, row_in in df_dict.iterrows():
        if index_in != index:
            if ccc != 0:
                string_marg = string_marg + " OR "
            string_marg = string_marg + '"' + row_in['Keyword'] + '"'
            ccc = ccc+1

    string_tt_Q4 = research_type + '((' + common_query + ') AND (' + string_marg + ')) ' + \
        'AND PUBYEAR > ' + str(first_year-1) + \
        ' AND PUBYEAR < ' + str(last_year+1)

    list_Q4.append(string_tt_Q4)


# %% Research through Scopus API

time.sleep(10)

# # Configuration
# config['Authentication']['APIKey'] = '608b807df041b44d76684e1b54d36bbf'
# config['Authentication']['InstToken'] = '3912cae38962c7a917223c7f9741de19'

# QUERY 1

df_Q1 = pd.DataFrame(columns=['Query', 'Count'])

s = ScopusSearch(string_Q1,download=False)
val = s.get_results_size()
# val = 20000

dict_Q1 = {'Query': ['Q1'], 'Count': [val]}

df_Q1 = pd.concat([df_Q1, pd.DataFrame.from_dict(dict_Q1)])


# QUERY 2

df_Q2 = pd.DataFrame(columns=['Query', 'Topic', 'Count'])

for idx, x in enumerate(df_dict['Topic'].unique()):

    s = ScopusSearch(list_Q2[idx],download=False)
    val = s.get_results_size()
    # val = 50000

    dict_Q2 = {'Query': ['Q2'], 'Topic': x, 'Count': [val]}

    df_Q2 = pd.concat([df_Q2, pd.DataFrame.from_dict(dict_Q2)])


# QUERY 3

df_Q3 = pd.DataFrame(columns=['Query', 'Topic', 'Keyword', 'Count'])

for index, row in df_dict.iterrows():

    s = ScopusSearch(list_Q3[index], download=False)
    val = s.get_results_size()
    # val = 50000

    dict_Q3 = {'Query': ['Q3'], 'Topic': row['Topic'],
               'Keyword': row['Keyword'], 'Count': [val]}

    df_Q3 = pd.concat([df_Q3, pd.DataFrame.from_dict(dict_Q3)])


# QUERY 4

df_Q4 = pd.DataFrame(columns=['Query', 'Topic', 'Keyword', 'Count'])

for index, row in df_dict.iterrows():

    s = ScopusSearch(list_Q4[index], download=False)
    val = s.get_results_size()
    # val = 50000

    dict_Q4 = {'Query': ['Q4'], 'Topic': row['Topic'],
               'Keyword': row['Keyword'], 'Count': [val]}

    df_Q4 = pd.concat([df_Q4, pd.DataFrame.from_dict(dict_Q4)])


# Writing output file
with pd.ExcelWriter(output_excel_file) as writer:
    df_Q1.to_excel(writer, sheet_name='Q1',index=False)
    df_Q2.to_excel(writer, sheet_name='Q2',index=False)
    df_Q3.to_excel(writer, sheet_name='Q3',index=False)
    df_Q4.to_excel(writer, sheet_name='Q4',index=False)
