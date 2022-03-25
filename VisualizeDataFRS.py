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

# import the WFDB package: python3 -m pip install wfdb
# https://pypi.org/project/wfdb/
import wfdb
# import plotting library matplotlib
import matplotlib.pyplot as plt
import numpy as np

# +
# load a record using the 'rdrecord' function
# Note: plese change path according to your implementation
# The data are to be found in https://archive.physionet.org/physiobank/database/ptbdb/
folder  = 'Datasets'
dataset = 's0287lre'
filename = folder+'/'+dataset
record = wfdb.rdrecord(filename)

# plot the record to screen
#wfdb.plot_wfdb(record=record, title='Example signals')

# +
# %matplotlib inline
'''
record.p_signals contains NxM rows, where N is the number of data points
and M is the number of channels revealed. Therefore we transpose this array.
M = 15 we chose a new array shape (3,15,N)
'''
signals = record.p_signal.T.reshape((3,5,record.p_signal.T.shape[1]))
'''
Plot the 15 channels in 5 rows and three colums. All the channels do have a slope.
It seems that the ground is drifting. This is compensated by removing a least square
fitted straight line.
'''
fig, ax = plt.subplots(5,3)

for col in np.arange(3):
    for row in np.arange(5):
        x = np.arange(signals[col,row].shape[0]) # enumerate the data points from 0 to signals[col,row].shape[0]
        A = np.vstack([x, np.ones(len(x))]).T    # stack the enumerated points with the same number of ones [x,ones]
        m,c = np.linalg.lstsq(A, signals[col,row], rcond=None)[0] # find slope and y(0) = c 
        y = signals[col,row]-(m*x+c) # subtract the straight line from the signaly
        ax[row,col].plot(y[:1100]) # plot it in the respective subplot
# -

y = signals[0,0]
x = np.arange(signals[0,0].shape[0])
A = np.vstack([x, np.ones(len(x))]).T
np.linalg.lstsq(A, y, rcond=None)[3]

y = signals[0,0]
x = np.arange(signals[0,0].shape[0])
A = np.vstack([x, np.ones(len(x))]).T
np.linalg.lstsq(A, y, rcond=None)

record.sig_name

y.shape


