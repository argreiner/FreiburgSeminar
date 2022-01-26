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
#
df_csv = pd.read_csv('Infizierte220111.csv',header=None, sep=';',na_filter=False)
data=df_csv.to_numpy()
nkreise=np.arange(len(data[:,0]))
dict_kreise = dict(zip(data[:,0],nkreise))
#
dfEWZ_csv = pd.read_csv('EWZLandkreise.csv',header=None, sep=';',na_filter=False)
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

# %matplotlib notebook
kreisliste = ['Freiburg im Breisgau (Stadtkreis)', 'Schwäbisch Hall']
fig, ax = plt.subplots() # let us plot the data
for kreis in kreisliste:
    factor=1.e5/dict_EWZkreise[kreis]
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

yHall = datafloat[28,1:]
yFrei =datafloat[36,1:]
DeltaHall = (yHall-np.roll(yHall,-7))*1.e5/EWZahl[28]
DeltaFrei = (yFrei-np.roll(yFrei,-7))*1.e5/EWZahl[36]
#plt.plot(np.flip(DeltaHall[:-7]))
#plt.plot(np.flip(DeltaFrei[:-7]))

# # Correlation functions
#
# The correlation function for two signals $A(t)$ and $B(t)$ is defined as 
# $$
# C(t)=\lim_{\tau\rightarrow\infty}\frac{1}{\tau}\int\limits_0^\tau A(t')B(t'-t)dt'$$
# Diskrete version
# $$ C(k\cdot\Delta\,t)=\lim_{\Delta\,t\rightarrow 0\atop n\rightarrow\infty}\sum_{i=1}^{n}A(i\cdot\Delta\,t)\cdot
# B\left((i-k)\cdot\Delta\,t\right)$$
# Let's have some examples.
#
# Note: Please subtract mean of every series A and B. See below

# %matplotlib notebook
# A = B = sin(a t)
n = 20000
a = 1.
t = np.linspace(0,20*np.pi,n)
#A = np.sin(t)
#B = np.cos(t)
A = np.random.rand(n)
B = np.random.rand(n)
A = A - np.mean(A) # Subtract mean
B = B - np.mean(B) # dto.
plt.plot(t, A)
plt.plot(t, B)

# Calculate Korrelation function between A and B. Watch out for the correct n
A = datafloat[36,1:]
B = datafloat[28,1:]
A = A - np.mean(A)
B = B - np.mean(B)
lmax = B.shape[0]//2
C = np.zeros(lmax)
S_A = np.sqrt(np.sum(A[:lmax]*A[:lmax])/lmax)
for k in np.arange(lmax):
    D = np.roll(B, shift = -k)[:lmax]
    S_B = np.sqrt(np.sum(D*D)/lmax)
    C[k] = np.sum(A[:lmax] * D)/lmax/S_A/S_B


# %matplotlib notebook
plt.plot(C[1:])

np.mean(np.flip(datafloat[dict_kreise[kreisliste[0]],1:]))

# ## Correlation between incidences of 2 counties in BW
#

# %matplotlib notebook
fig, ax = plt.subplots() # let us plot the data
# Plot Freiburg incidences - mean
for kreis in kreisliste:
    ax.plot(datafloat[dict_kreise[kreis],1:]-np.mean(datafloat[dict_kreise[kreis],1:]), label = kreis)
ax.legend()
ax.set_title('Infizierte')
ax.set(xlabel='Tag')
ax.set(ylabel='Anzahl')


dict_kreise['Schwäbisch Hall']

A = np.arange(10)
B = np.arange(10,20)

A

np.roll(B, shift = -1)

np.sum((A*np.roll(B, shift = -1))[:-1])

datafloat[36,1:].shape[0]

B

dict_EWZkreise.keys()

df_csv = pd.read_csv('Infizierte220111.csv',header=None, sep=';',na_filter=False)
nkreise=np.arange(len(df_csv.to_numpy()[:,0]))
dict_kreise = dict(zip(df_csv.to_numpy()[:,0],nkreise))

dict_kreise.keys()


