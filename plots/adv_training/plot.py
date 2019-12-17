#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas

data_l1 = pandas.read_csv('run-Aug10_18-59-17_gpu-tag-adv_avdistance.csv')
data_l2 = pandas.read_csv('run-Aug10_18-59-18_gpu-tag-adv_avdistance.csv')

plt.plot(data_l1['Value'])
plt.plot(data_l2['Value'])

plt.legend(['Training with $\ell_1$ distance', 'Training with $\ell_2$ distance'])
plt.ylabel('Average $\ell_1$ distance')
plt.xlabel('Adversarial training step')
plt.savefig('avdistance.pdf')
#plt.show()
