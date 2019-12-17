#!/usr/bin/env python3

# An ad-hoc script for plotting tables copy-pasted from LibreOffice Calc

import fileinput
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D

group_names = []
values = []

plt.rcParams['font.family'] = 'serif'
plt.rcParams['savefig.format'] = 'pdf'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for line in fileinput.input('-'):
	field = line.rstrip().split('\t')
	group_names.append(field[0])
	values.append(field[1:])

feature_names = values[0]
group_names = group_names[1:]
values = values[1:]

print("feature_names", feature_names)
print("group_names", group_names)
print("values", values)
values = np.array(values, dtype=float)

def importance():
	global values
	plt.figure(figsize=(5,4))
	values[values < 0] = 0
	values /= np.sum(values,axis=0)

	FACTOR = 0.85
	# width = 1/(1+values.shape[1])
	width = FACTOR / values.shape[1]

	order = np.argsort(-np.mean(values, axis=1))
	x = np.arange(len(group_names))

	for i in range(values.shape[1]):
		plt.bar(x + width*i, values[order,i], width, alpha=0.9, label=feature_names[i])
	for i in range(values.shape[1]):
		plt.bar(x + width*i, np.mean(values[order,:], axis=1), width, color='gray', alpha=0.5, **({'label': 'Mean'} if i==0 else {}), zorder=0)

	plt.legend()
	plt.xticks(x + width*values.shape[1]/2-width*0.5, [group_names[i] for i in order], rotation=45, horizontalalignment='right')
	for tick in plt.gca().xaxis.get_major_ticks():
		label = tick.label1
		# don't know how to specify translation in axis coordinates, so pdf output
		# looks slightly different from plot
		label.set_transform(label.get_transform() + Affine2D().translate(5,0))
	plt.ylabel('Normalized metric')
	plt.tight_layout()

def adv_results():
	global values
	plt.figure(figsize=(5,3))
	values = np.reshape(values, (-1,values.shape[1]//2,2))

	order = (2+np.argsort(np.mean(values[2:,:,1], axis=1))).tolist()
	order = [0, 1] + order
	width = 0.95 / values.shape[1]

	x = np.arange(len(group_names), dtype=float)
	x[:2] -= .5
	for i in range(values.shape[1]):
		plt.bar(x +width*i, values[order,i,1], width, color='gray', alpha=.5, **({'label': 'Original'} if i==0 else {}))
	for i in range(values.shape[1]):
		plt.bar(x +width*i, values[order,i,0], width, color=colors[i], label=feature_names[i*2].rstrip(' adv'))
	plt.legend(loc='lower center', bbox_to_anchor=(0.5,1), ncol=4)
	plt.xticks(x + width*values.shape[1]/2-width*0.5, [group_names[i] for i in order], rotation=45, horizontalalignment='right')
	for i,label in  enumerate(plt.gca().get_xticklabels()):
		if i < 2:
			label.set_fontweight('bold')
	for tick in plt.gca().xaxis.get_major_ticks():
		label = tick.label1
		label.set_transform(label.get_transform() + Affine2D().translate(8,0))
	plt.ylabel('Recall')
	plt.tight_layout()

def ars_original():
	global group_names
	global values

	y_labels = [None, "Recall", "Recall", "ARS", "ARS"]
	value_indices_to_plot = (1,3)

	values = [[float(item) for item in sublist] for sublist in values]

	index_by_which_to_sort = 1
	old_len = len(group_names)
	orig_group_names, orig_values = group_names, values
	group_names, values = list(zip(*[(group_name, value) for group_name, value in zip(group_names, values) if value[0] <= 0.5]))
	# print(f"Dropped {old_len-len(group_names)} elements.")
	robust_groups = set(orig_group_names)-set(group_names)
	robust_indices = [orig_group_names.index(item) for item in robust_groups]
	print("robust attack types:", "; ".join([f"{g}: orig_flow_recall={v[1]}, adv_flow_recall={v[0]}" for g, v in zip([orig_group_names[index] for index in robust_indices], [orig_values[index] for index in robust_indices])]))
	order = list(zip(*sorted(enumerate(values), key=lambda item: item[1][index_by_which_to_sort], reverse=True)))[0]
	# print("order", order)
	width = 0.75 / len(value_indices_to_plot)

	values = np.array(values)

	x = np.arange(len(group_names), dtype=float)

	fig, ax1 = plt.subplots(figsize=(5,3))
	plt.xticks(x + width/2, [group_names[i] for i in order], rotation=45, horizontalalignment='right')
	for tick in plt.gca().xaxis.get_major_ticks():
		label = tick.label1
		label.set_transform(label.get_transform() + Affine2D().translate(8,0))
	ax2 = ax1.twinx()
	axes = [ax1, ax2]

	all_labels = []
	for index, i in enumerate(value_indices_to_plot):
		label = axes[index].bar(x + width*index, values[order,i], width, color=colors[index], label=feature_names[i])
		all_labels.append(label)
		axes[index].set_ylabel(y_labels[i])

	all_legends = [item.get_label() for item in all_labels]

	# ax1.legend(all_legends, all_labels, loc='upper right', bbox_to_anchor=(1,0.95))
	#  plt.ylim((min(first_quartiles), max(third_quartiles)))

	ax2.set_ylabel_legend(Rectangle((0,0), 1, 1, fc=colors[1]), handlelength=1)
	ax1.set_ylabel_legend(Rectangle((0,0), 1, 1, fc=colors[0]), handlelength=1)
	# ax1.set_ylabel_legend(Rectangle((0,0), 1, 1, fc='gray', alpha=0.2), handlelength=1)
	# plt.legend(all_labels, all_legends, loc='lower center', bbox_to_anchor=(0.5,1), ncol=4)

	plt.tight_layout()

def ars():
	plt.figure(figsize=(5,2))
	x_values = [ float(value) for value in group_names ]
	plt.plot(x_values, values)
	plt.legend(feature_names, ncol=2)
	plt.xlabel('Training duration in epochs')
	plt.ylabel('ARS')
	ylim1,ylim2 = plt.ylim()
	plt.ylim((ylim1,ylim2+20)) # move plots away from legend
	plt.tight_layout()

def adv():
	plt.figure(figsize=(5,2))
	x_values = [ float(value) for value in group_names ]
	lines = plt.plot(x_values, values[:,:2])
	plt.xlabel('Tradeoff $c$')
	plt.ylabel('Success ratio')
	plt.gca().set_ylabel_legend(Line2D([0],[0], color='gray'), handlelength=1.4)
	plt.twinx()
	plt.plot(x_values, values[:,2:], linestyle='--')
	plt.ylabel('$L_1$ distance')
	plt.gca().set_ylabel_legend(Line2D([0],[0], color='gray', linestyle='--'), handlelength=1.4)
	plt.legend(lines, ['CIC-IDS-2017', 'UNSW-NB15'])
	ylim1,ylim2 = plt.ylim()
	plt.ylim((ylim1,11))
	#  plt.ylim((ylim1,ylim2+20)) # move plots away from legend
	plt.tight_layout()

globals()[sys.argv[1]]()
if len(sys.argv) > 2:
	plt.savefig(sys.argv[2], bbox_inches = 'tight', pad_inches = 0)
plt.show()
