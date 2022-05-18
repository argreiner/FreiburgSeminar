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

import numpy as np
import matplotlib.pyplot as plt

# Harmonischer Oszillator
x = np.zeros(100000)
x[0] = 1./np.sqrt(2.)
v = np.zeros(100000)
v[0] = 1./np.sqrt(2.)
dt = .1
#
for i in np.arange(1,100000):
    v[i] = v[i-1] - x[i-1]*dt
    x[i] = x[i-1] + v[i]*dt


# %matplotlib notebook
plt.plot(x,v)

for i in np.arange(100000,1):
    x[i-1] = x[i] - v[i]*dt
    v[i-1] = v[i] + x[i]*dt

# %matplotlib notebook
plt.plot(x,v)

x[0]

from scipy.integrate import odeint
#
def func(r,t):
    x,v=r
    # here you enter the differential equation system
    dx=v
    dv=-x
    return dx,dv


# here you integrate the DE and make a side by side plot
r0=(1./np.sqrt(2.),1./np.sqrt(2.))
t=np.linspace(0,100,10000)  # We do 10000 steps in the iterval [0,100]
pos=odeint(func,r0,t) # Integrate the system of differential equations

x = pos.T[0]
v = pos.T[1]

# %matplotlib notebook
plt.plot(x,v)


