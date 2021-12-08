# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Plotting the incidences of BW counties

import pandas as pd    # Pandas is a library to analyse various data structures
import numpy as np
import matplotlib.pyplot as plt

# Here we read the csv-file holding the data.
# Note: the separator is a ; instead of a comma, all NA-values gave to be skipped
df_csv = pd.read_csv('/Users/greiner/Programming/Python/FreiburgSeminar/Infizierte.csv', sep=';',na_filter=False)
keys=df_csv.keys()

# Let us pack the counties names into a dictionary and number them
# e.g. {'Biberach': 1}
nkreise=np.arange(len(df_csv.to_numpy()[:,0]))
dict_kreise = dict(zip(df_csv.to_numpy()[:,0],nkreise))

data=df_csv.to_numpy()[:,1:]

irange=np.arange(data.shape[0])
jrange=np.arange(data.shape[1])
datafloat=np.zeros(data.shape)
for i in irange: 
    for j in jrange:
        if data[i,j] =='': 
            datafloat[i,j]=0.
        else:
            datafloat[i,j]=float(data[i,j])

# %matplotlib notebook
fig, ax = plt.subplots() # let us plot the data
y = np.flip(datafloat[0])
delta = y - np.roll(y,shift=7)
deltabool = delta > 0
ax.plot(delta[deltabool])
ax.set_title('Die 7-Tage Inzidenz')
ax.set(xlabel='Tag')
ax.set(ylabel='Anzahl')

datafloat[0]


