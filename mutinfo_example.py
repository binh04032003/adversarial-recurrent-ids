#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"

resolution = 10000
x = np.arange(0,1,1/resolution)

y = (np.mod(np.arange(resolution), np.ceil(resolution/3.5)) < np.ceil(resolution/3.5)/2).astype(int)
y2 = y.astype(float)
y2[np.argmin(y):] = .5

plt.figure(figsize=(5,1.3))
plt.subplot(1,2,1)
plt.plot(x,y2, c='k')
plt.xlim((0,1))
plt.yticks([0,.5,1])
plt.ylim((-0.1,1.1))
plt.xlabel('$x$')
plt.ylabel('$f(A,x)$')

plt.subplot(1,2,2)
plt.plot(x,y, c='k')
plt.xlim((0,1))
plt.yticks([0,.5,1])
plt.ylim((-0.1,1.1))
plt.xlabel('$x$')
plt.ylabel('$f(A,x)$')

plt.tight_layout()
plt.savefig('plots/mutinfo_example.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()
