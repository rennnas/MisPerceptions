# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:58:09 2024

@author: magal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Data you provided
data = {
    'Topic': list(range(1, 64)),  # Topic numbers from 1 to 63
    'Count': [
        6595, 1327, 800, 611, 480, 478, 238, 231, 229, 225, 216, 200, 173, 162, 161, 
        144, 142, 141, 129, 124, 117, 115, 113, 112, 107, 97, 95, 89, 87, 81, 77, 67, 
        65, 62, 62, 62, 58, 57, 57, 52, 50, 49, 47, 45, 45, 43, 43, 42, 42, 41, 41, 
        40, 39, 38, 35, 34, 33, 30, 30, 29, 29, 26, 25
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Total occurrences for normalization
total_occurrences = df['Count'].sum()

# Normalize the frequencies
df['frequency_normalized'] = df['Count'] / total_occurrences

# Display the first few rows of the DataFrame
print(df.head())


# Only apply PCA to the 'frequency' column (or normalized frequency)
X = df[['frequency_normalized']]


# Sort topics by frequency
df_sorted = df.sort_values(by='frequency_normalized', ascending=False)

# Plot the cumulative sum of normalized frequencies
plt.plot(np.cumsum(df_sorted['frequency_normalized']))
plt.xlabel('Number of Topics')
plt.ylabel('Cumulative Frequency')
plt.title('Diversity of Conversation by Number of Topics')
plt.show()