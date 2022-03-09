# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Grafische Darstellung der Inzidenzen in BW
#
# Hier käme noch Text zur Erklärung...

import pandas as pd               # Pandas ist eine Bibliothek zur Analys verschiedener Datenstrukturen
import numpy as np                # numpy ist eine Bilbiothek zur array Manipulation so wie MatLab
import matplotlib.pyplot as plt   # matplotlib ist für Grafik
from datetime import datetime     # Für das Datum: to be used in a string for filenames

# Diese URL die Daten, die vom "Ministerium für Soziales, 
# Gesundheit und Integration Baden-Württemberg" veröffentlicht werden
url = 'https://sozialministerium.baden-wuerttemberg.de/fileadmin/redaktion/m-sm/intern/downloads/Downloads_Gesundheitsschutz/Tabelle_Coronavirus-Faelle-BW.xlsx'
# Einlesen der excel Datei in einen pandas dataframe
#df_excel = pd.read_excel('Tabelle_Coronavirus-Faelle-BW-5.xlsx',na_filter=False)
df_excel = pd.read_excel(url,na_filter=False)
#
now = datetime.now()   # heutiges Datum und Urhzeit
datum = now.strftime("%Y%m%d")  # Extrahiere Jahr, Monat, Tag
df_excel.to_excel('CoronafaelleBW'+datum+'.xlsx',sheet_name='Cases')  # Sichere das von der URL gelesene File
# 
data = df_excel.to_numpy()[6:-3]  # Konvertiere Pandas Dataframe zu einem numpy array
data[data == ''] = '0'  # Setze die leeren Einträge auf den String '0'
#
ni,nj = data[:,1:].shape  # Shape des array (am 5.3.2022 z.B. (44,721))
data_int = np.array([[int(z) for z in data[i,1:]] for i in np.arange(ni)])  # Wandle die Stringeinträge in int um
data_dict = dict(zip(data[:,0],data_int))  # Mache einen Dictionary aus Daten und Kreisnamen
#
dfEWZ_csv = pd.read_csv('EWZLandkreise.csv',header=None, sep=';',na_filter=False)  # Einwohnerzahl der Kreise
dict_EWZkreise = dict(dfEWZ_csv.to_numpy())   # Dictionary der Einwohnerzahlen und Kreise


# +

def keyString(partKey,dictionary):
    '''
    Diese Funktion gibt uns den korrekten key zurück, um in den dictionaries 
    nach values zu suchen. Die Funktion verlangt als Eingabe einen String und 
    einen dict.
    Returnvalue ist der korrekte Keyname.
    Bsp: 
    > keyString('Ulm',data_dict)
    > 'Ulm (Stadtkreis)'
    '''
    for key in dictionary.keys(): 
        if key.startswith(partKey): searchkey = key
    return searchkey


# -

# %matplotlib notebook
fig, ax = plt.subplots() # initialisiere die Grafik
#
factor = 1.e5/sum(dict_EWZkreise.values()) # Gesamteinwohnerzahl BW
totalData = np.flip(sum(data_dict.values()))     # Gesamtzahl Infizierter nach Datum
ytotal = (totalData - np.roll(totalData,7))
ax.plot(factor*ytotal[7:], label='GesamtBW')
#
kreisliste = [ 'Schwäbisch Hall', 'Lörrach', 'Freiburg']   # Wähle ein paar Kreise die zu plotten willst
for kreis in kreisliste:
    keystring = keyString(kreis,data_dict) 
    factor=1.e5/dict_EWZkreise[keystring]  # Faktor 7 Tage Inzidenz
    y = np.flip(data_dict[keystring])   # Daten umdrehen
    y7 = np.roll(y,7)     # 7 Tage vorher
    y7[:8] = 0            # Erste 7 Einträge auf 0 setzen. Eigentlich nicht notwendig wenn man von 7 an plottet
    delta = y  - y7       # Differenz zwischen Eintrag und 7 Tage zuvor
    ax.plot(factor*delta[7:], label=kreis)    # Zeichnen
    #ax.plot(y, label=kreis)
#
ax.legend()
ax.set_title('Die 7-Tage Inzidenz, absolut')
ax.set(xlabel='Tag')
ax.set(ylabel='Anzahl')

# # Korrelationsfunktionen
#
# The correlation function for two signals $A(t)$ and $B(t)$ is defined as 
# $$
# C(t)=\lim_{\tau\rightarrow\infty}\frac{1}{\tau}\int\limits_0^\tau A(t')B(t'-t)dt'$$
# Diskrete version in terms of python numpy arrays, the algorithms. Unfortunately the limit $n\rightarrow\infty$ we cannot do. So we keep a finite $n$ as the length of an array. This gives us
# $$C[k]=\frac{1}{(n-k)}\sum_{i=0}^{n-k}A[i]\cdot B[i+k]$$
# In order to normalize the correlation function we divide by the standard deviations of $A(t)$ and $B(t)$ to read
# $$C(k\cdot\Delta\,t)=\frac{1}{(n-k)}\frac{\sum\limits_{i=0}^{n-k}A[i]\cdot B[i+k]}{S_AS_B}$$
# where the standard deviations are 
# $$S_A=\sqrt{\frac{1}{n-k}\sum_{i=0}^{n-k}A(i\cdot\Delta\,t)^2}$$
# and
# $$S_B=\sqrt{\frac{1}{n-k}\sum_{i=0}^{n-k}B(i\cdot\Delta\,t+k\cdot\Delta\,t)^2}$$
#
# Note: Please subtract mean of every series $A$ and $B$. See below 

# ### We calculate the correlation function between A and B. Watch out for the correct n

# %matplotlib notebook
fig, ax = plt.subplots() # initialisiere die Grafik
#
kreisA = keyString('Stuttgart',data_dict)
kreisB = keyString('Freiburg',data_dict)
A = data_dict[kreisA] # Choose dataset A
B = data_dict[kreisB] # Choose dataset B
A = A - np.mean(A) # subtract mean
B = B - np.mean(B) # subtract mean
lmax = B.shape[0]//2 # Choose the maximum value of k (see formulae above)
C = np.zeros(lmax)
S_A = np.sqrt(np.sum(A[:lmax]*A[:lmax])/lmax) # Standard deviation of A
for k in np.arange(lmax):
    D = np.roll(B, shift = -k)[:lmax] # shift and take the first lmax entries
    S_B = np.sqrt(np.sum(D*D)/lmax) # Standard deviation of B
    C[k] = np.sum(A[:lmax] * D)/lmax/S_A/S_B # The correlation function
# Note that we could just skip the division by lmax because it cancels !!!
ax.plot(C[1:], label=kreisA+', '+kreisB)
#
kreisA = keyString('Lörrach',data_dict)
kreisB = keyString('Schwäbisch Hall',data_dict)
A = data_dict[kreisA] # Choose dataset A
B = data_dict[kreisB] # Choose dataset B
A = A - np.mean(A) # subtract mean
B = B - np.mean(B) # subtract mean
lmax = B.shape[0]//2 # Choose the maximum value of k (see formulae above)
C = np.zeros(lmax)
S_A = np.sqrt(np.sum(A[:lmax]*A[:lmax])/lmax) # Standard deviation of A
for k in np.arange(lmax):
    D = np.roll(B, shift = -k)[:lmax] # shift and take the first lmax entries
    S_B = np.sqrt(np.sum(D*D)/lmax) # Standard deviation of B
    C[k] = np.sum(A[:lmax] * D)/lmax/S_A/S_B # The correlation function
# Note that we could just skip the division by lmax because it cancels !!!
ax.plot(C[1:], label=kreisA+', '+kreisB)
ax.legend()
ax.set_title('Korrelation zwischen 2 Landkreisen')
ax.set(xlabel='Zeit t')
ax.set(ylabel='Korrelation C(t)')

# # Autokorrelation
# Das bedeutet wir berechnen
# $$
# C(t)=\lim_{\tau\rightarrow\infty}\frac{1}{\tau}\int\limits_0^\tau A(t')A(t'-t)dt'$$

# %matplotlib notebook
fig, ax = plt.subplots() # initialisiere die Grafik
#
gesamtBW = sum(data_dict.values()) 
A = gesamtBW - np.mean(gesamtBW)
lmax = A.shape[0]//2 # Choose the maximum value of k (see formulae above)
C = np.zeros(lmax)
S_A = np.sqrt(np.sum(A[:lmax]*A[:lmax])/lmax) # Varianz
for k in np.arange(lmax):
    temp = np.roll(A, shift = -k)[:lmax]
    B = temp - np.mean(temp) # shift and take the first lmax entries
    S_B = np.sqrt(np.sum(B*B)/lmax) # Standard deviation of B
    #S_B = 1.0
    #S_A = 1.0
    C[k] = np.sum(A[:lmax] * B)/lmax/S_A/S_B # The correlation function
# Note that we could just skip the division by lmax because it cancels !!!
ax.plot(C[1:], label='gesamtBW')
ax.legend()
ax.set_title('Autokorrelation gesamt BW')
ax.set(xlabel='Zeit t')
ax.set(ylabel='Korrelation C(t)')

S_A

k=0
temp = np.roll(A, shift = -k)[:lmax]
B = temp - np.mean(temp) # shift and take the first lmax entries
S_B = np.sqrt(np.sum(B*B)/lmax) # Standard deviation of B
print(S_B)

A


