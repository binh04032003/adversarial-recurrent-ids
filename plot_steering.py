#!/usr/bin/env python3

import numpy as np
import json
import sys
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams['pdf.fonttype'] = 42

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

path = sys.argv[1]
f = open(path, "r")
chosen_packets_per_epoch_sampling_rl, all_packets_per_epoch_sampling_rl, tradeoffs_sampling_rl = json.loads(f.read())
f.close()

chosen_packets_per_epoch_sampling_rl = np.array(chosen_packets_per_epoch_sampling_rl[:-1])
all_packets_per_epoch_sampling_rl = np.array(all_packets_per_epoch_sampling_rl[:-1])
tradeoffs_sampling_rl = np.array(tradeoffs_sampling_rl)

assert len(chosen_packets_per_epoch_sampling_rl) == len(all_packets_per_epoch_sampling_rl) == len(tradeoffs_sampling_rl)

# print("Epochs", len(chosen_packets_per_epoch_sampling_rl))

x = list(range(1,len(chosen_packets_per_epoch_sampling_rl)+1))

# print("x", x, "chosen_packets_per_epoch_sampling_rl", chosen_packets_per_epoch_sampling_rl, "all_packets_per_epoch_sampling_rl", all_packets_per_epoch_sampling_rl, "tradeoffs_sampling_rl", tradeoffs_sampling_rl)

plt.figure(figsize=(5,3))
lines = []
lines += plt.plot(x, 100*chosen_packets_per_epoch_sampling_rl/all_packets_per_epoch_sampling_rl, color=colors[0])
plt.xlabel("Time steps")
plt.ylabel('Sparsity (%)')

plt.twinx()
lines += plt.plot(x, tradeoffs_sampling_rl, color=colors[1])
plt.legend(lines, ["Sparsity", "Tradeoff"], loc='upper right')
plt.ylabel('Tradeoff')
plt.tight_layout()
plt.savefig("paper_rl/img/" + (path.split("/")[-1])[:-5] + '.pdf', bbox_inches = 'tight', pad_inches = 0)
# plt.show()

plt.close()