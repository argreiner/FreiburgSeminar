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

import numpy as np
import matplotlib.pyplot as plt

a = np.zeros((10,10))
a[5,5] = 1
plt.imshow(a)

b = np.roll(a,shift=(2,-2), axis=(0,1))
plt.imshow(b)


