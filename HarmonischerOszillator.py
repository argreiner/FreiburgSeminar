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
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# # Harmonic oscillator revisited
#
# Die Oszillatorgleichung lautet $m\cdot \ddot{x} + k\cdot x = 0$
#
# \begin{align}
# \dot{x} &= v\\
# \dot{v} &= -\omega^2 x\\
# \end{align}

# Harmonischer Oszillator
x = np.zeros(100000)
x[0] = 1./np.sqrt(2.)
v = np.zeros(100000)
v[0] = 1./np.sqrt(2.)
dt = .01
#
for i in np.arange(1,100000):
    x[i] = x[i-1] + v[i-1]*dt
    v[i] = v[i-1] - x[i]*dt


# %matplotlib notebook
plt.plot(x,v)

for i in np.arange(100000,1):
    x[i-1] = x[i] - v[i]*dt
    v[i-1] = v[i] + x[i]*dt

# %matplotlib notebook
plt.plot(x,v)



from scipy.integrate import odeint, solve_ivp
#
def func4odeint(r,t):
    x,v=r
    # here you enter the differential equation system
    dx=v
    dv=-x
    return dx,dv
#
def func4solve_ivp(t,r):
    x,v=r
    # here you enter the differential equation system
    dx=v
    dv=-x
    return dx,dv


# here you integrate the DE and make a side by side plot
ti = 0.
tf = 1000.
r0=(1./np.sqrt(2.),1./np.sqrt(2.))
#
t=np.linspace(0,1000,10000)  # We do 10000 steps in the iterval [0,100]
t_eval = np.linspace(0,1000,10000)
pos = odeint(func4odeint,r0,t) # Integrate the system of differential equations
sol = solve_ivp(func4solve_ivp,[ti, tf],r0, method = 'DOP853', t_eval=t_eval, rtol = 1.e-10, atol = 1.e-12)

x = pos.T[0]
v = pos.T[1]

# %matplotlib notebook
plt.plot(x,v)

x,v = sol.y

# %matplotlib notebook
plt.plot(x,v)


# # Planetenbewegung
# Die Newton'sche Bewegungsgleichung für einen Massenpunkt im Zentralfeld lautet
# $$m\ddot{\mathbf{r}}=-\frac{\Gamma Mm}{r^2}\frac{\mathbf{r}}{r}$$
# Man kann durch $m$ dividieren und erhält
# \begin{align}
# \ddot{x} &= -\frac{\Gamma M}{r^2}\frac{x}{r}\\
# \ddot{y} &= -\frac{\Gamma M}{r^2}\frac{y}{r}
# \end{align}
# Wir schreiben das ganze als System von 4 gekoppelten gewöhnlichen Differentialgleichungen 1. Ordnung
# \begin{align}
# \dot{x}   &= v_x\\
# \dot{v}_x &= -\frac{\Gamma M}{r^3}x\\
# \dot{y}   &= v_y\\
# \dot{v}_y &= -\frac{\Gamma M}{r^3}y
# \end{align}
#
# Der Abstand ist $r = \sqrt{x^2+y^2}$
#
# ### Periheldrehung des Merkur
#
# Das Gravitationspotential nahe einer Gravitationsquelle ergibt sich zu 
#
# $$V_{eff} = -\frac{M}{r}+\frac{l^2}{2r^2}-\frac{Ml^2}{r^3}$$

# Let's define the function
def func(t, z):
    m1 = 1.
    m2 = 2.
    M  = 1.
    Gamma = 1.
#
    x1,vx1,y1,vy1,x2,vx2,y2,vy2 = z
    r1 = np.sqrt(x1**2 + y1**2)
    r2 = np.sqrt(x2**2 + y2**2)
    r12 = np.sqrt((x1-x2)**2+(y1-y2)**2)
    dx1 = vx1
    dvx1 = -Gamma*M/r1**3 * x1 + Gamma*m2/r12**3*(x2-x1)
    dy1 = vy1
    dvy1 = -Gamma*M/r1**3 * y1 + Gamma*m2/r12**3*(y2-y1)
    dx2 = vx2
    dvx2 = -Gamma*M/r2**3 * x2 + Gamma*m1/r12**3*(x1-x2)
    dy2 = vy2
    dvy2 = -Gamma*M/r2**3 * y2 + Gamma*m1/r12**3*(y1-y2)
    return [dx1,dvx1,dy1,dvy1,dx2,dvx2,dy2,dvy2]


# Let's define the function
def func(t, z):
    lsquare = (7.e10*47400.)**2
    GammaMm = 3.3e23*6.672e-11*2.*1.e30*lsquare
    x1,vx1,y1,vy1 = z
    r1 = np.sqrt(x1**2 + y1**2)
    dx1 = vx1
    dvx1 = -GammaMm/r1**3 * x1 - lsquare/2./r1**4*x1-GammaMm/r1**5 *x1#+ Gammam/r12**3*(x2-x1)
    dy1 = vy1
    dvy1 = -GammaMm/r1**3 * y1 - lsquare/2./r1**4*x1- GammaMm/r1**5 *y1#+ Gammam/r12**3*(y2-y1)
    return [dx1,dvx1,dy1,dvy1]


# Let's define the function
def func(t, z):
    x1,vx1,y1,vy1 = z
    GammaM = 1.
    r1 = np.sqrt(x1**2 + y1**2)
    dx1 = vx1
    dvx1 = -GammaM/r1**3 * x1 - 0.01*GammaM/r1**5 *x1#+ Gammam/r12**3*(x2-x1)
    dy1 = vy1
    dvy1 = -GammaM/r1**3 * y1 - 0.01*GammaM/r1**5 *y1#+ Gammam/r12**3*(y2-y1)
    return [dx1,dvx1,dy1,dvy1]


# here you integrate the DE and make a side by side plot
ti = 0.
tf = 1000.
r0=(1.,0.,0.,1.,2.,0.,0.,.9)
#r0=(7.e10,0.,0.,47400.)
t_eval = np.linspace(ti,tf,10000)
GammaM = 1.
#
sol = solve_ivp(func, [ti, tf],r0, method = 'DOP853', t_eval=t_eval, rtol = 1.e-10, atol = 1.e-12)

x1 = sol.y[0]
y1 = sol.y[2]
x2 = sol.y[4]
y2 = sol.y[6]
#xcom = (x1+x2)/2.
#ycom = (y1+y2)/2.

# %matplotlib notebook
plt.plot(x1,y1)
plt.plot(x2,y2)
#plt.plot(xcom,ycom)


