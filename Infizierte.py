# -*- coding: utf-8 -*-
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
#
df_csv = pd.read_csv('/Users/greiner/Programming/Python/FreiburgSeminar/Infizierte220111.csv',header=None, sep=';',na_filter=False)
data=df_csv.to_numpy()
#
dfEWZ_csv = pd.read_csv('/Users/greiner/Programming/Python/FreiburgSeminar/EWZLandkreise.csv', sep=';',na_filter=False)
dict_EWZkreise = dict(dfEWZ_csv.to_numpy())

# transform data to float
irange=np.arange(data.shape[0])
jrange=np.arange(1,data.shape[1])
datafloat=np.zeros(data.shape)
dataint=np.zeros(data.shape)
for i in irange: 
    for j in jrange:
        if data[i,j] =='': 
            datafloat[i,j]=0.
            dataint[i,j]=0
        else:
            datafloat[i,j]=float(data[i,j])
            dataint[i,j]=int(data[i,j])
#
kreisliste = ['Schwäbisch Hall', 'Freiburg im Breisgau (Stadtkreis)']
EWZahl = np.ones(44)
EWZahl[36] = 230940
EWZahl[6] = 166862
EWZahl[20] = 432580
EWZahl[0] = 198204
EWZahl[40] = 309721
EWZahl[35] = 55449
EWZahl[14] = 308436
EWZahl[28] = 197860

# %matplotlib notebook
fig, ax = plt.subplots() # let us plot the data
for kreis in kreisliste:
#for i in np.arange(44):
    factor=1.e5/EWZahl[dict_kreise[kreis]]
    #factor = 1.
    #y = np.flip(datafloat[dict_kreise[kreis]])
    y = np.flip(datafloat[dict_kreise[kreis],1:])
    y14 = np.roll(y,7)
    y14[:13] = 0
    delta = y  - y14
    ax.plot(factor*delta[0:], label=kreis)
    #ax.plot(y, label=kreis)
#
ax.legend()
ax.set_title('Die 7-Tage Inzidenz, absolut')
ax.set(xlabel='Tag')
ax.set(ylabel='Anzahl')

dict_kreise

EWZahl = np.ones(44)
EWZahl[36] = 230940
EWZahl[6] = 166862
EWZahl[20] = 432580

df_dict_kreise = pd.DataFrame(dict_kreise.items())
df_dict_kreise.to_csv("Landkreise.dat",sep='\t')



dict_kreise

dict_EWZkreise['Konstanz, Landkreis']

a=np.arange(8)

a.reshape((2,4)).T

dict_kreise

dict_EWZkreise

for key in kreisliste:
    print(dict_kreise[key])


