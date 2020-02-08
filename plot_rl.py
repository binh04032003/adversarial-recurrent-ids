#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import sys
import os
import json
import pickle

def numpy_sigmoid(x):
	return 1/(1+np.exp(-x))

DIR_NAME = "plots/plot_rl"
ORDERING = ['', 'Length min', 'IAT min', 'Length max', 'IAT max']

MAX_X = 20
SHOW_TITLE = False

plt.rcParams["font.family"] = "serif"


dataroot_basename = sys.argv[1].split('_')[0]

with open(dataroot_basename + "_categories_mapping.json", "r") as f:
	categories_mapping_content = json.load(f)
categories_mapping, mapping = categories_mapping_content["categories_mapping"], categories_mapping_content["mapping"]
reverse_mapping = {v: k for k, v in mapping.items()}
print("reverse_mapping", reverse_mapping)

file_name = sys.argv[1]
with open(file_name, "rb") as f:
	loaded = pickle.load(f)
results_by_attack_number, sample_indices_by_attack_number, orig_seq_lens = loaded["results_by_attack_number"], loaded["sample_indices_by_attack_number"], loaded["orig_lens"]

# print("results", results_by_attack_number)
# print("sample_indices", sample_indices_by_attack_number)
lens_results = [len(item) for item in results_by_attack_number]
lens_indices = [len(item) for item in sample_indices_by_attack_number]
lens_lens = [len(item) for item in orig_seq_lens]

print("lens_indices", "\n".join(["{}: {}".format(reverse_mapping[attack], length) for attack, length in zip(range(len(lens_indices)), lens_indices)]))
assert lens_results == lens_indices == lens_lens

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

all_seqs = []
all_attacks = []
all_seqs_lens = []
all_attacks_lens = []
for attack_type, (seqs, lens) in enumerate(zip(results_by_attack_number, orig_seq_lens)):
	for seq in seqs:
		seq[:,-1] = numpy_sigmoid(seq[:,-1])

	if reverse_mapping[attack_type] == 'Normal':
		for seq in seqs:
			seq[:,-1] = 1-seq[:,-1]
		all_seqs.extend(seqs)
		all_seqs_lens.extend(lens)
	else:
		all_seqs.extend(seqs)
		all_attacks.extend(seqs)
		all_seqs_lens.extend(lens)
		all_attacks_lens.extend(lens)

reverse_mapping[len(results_by_attack_number)] = 'All samples'
results_by_attack_number.append(all_seqs)
reverse_mapping[len(results_by_attack_number)] = 'All attacks'
results_by_attack_number.append(all_attacks)

orig_seq_lens.append(all_seqs_lens)
orig_seq_lens.append(all_attacks_lens)

for attack_type, (seqs, lens) in enumerate(zip(results_by_attack_number, orig_seq_lens)):
	print (attack_type, reverse_mapping[attack_type], len(seqs))
	if len(seqs) <= 0:
		continue

	# lens = sorted(lens, reverse=True)
	max_length = max(lens)

	# values_by_length = []
	lens_to_plot = np.zeros((max_length))
	for l in lens:
		lens_to_plot[:l] += 1

	filled_ins = []
	filled_in_confidences = []
	for seq, l in zip(seqs, lens):
		skip_pattern = seq[:,-2]
		assert (np.round(skip_pattern) == skip_pattern).all(), skip_pattern
		confidence = seq[:,-1]
		# print("confidence", confidence)
		assert (confidence <= 1).all(), confidence
		filled_in = np.zeros((l))
		filled_in_confidence = np.zeros((l))

		cursor = -1
		prev_confidence = None
		for index, (item, conf) in enumerate(zip(skip_pattern, confidence)):
			# print("filled_in_confidence", filled_in_confidence)
			old_cursor = cursor
			assert item >= 0, f"{item}"
			cursor += int(item)+1
			filled_in[cursor] = 1
			assert cursor - old_cursor >= 1, f"cursor: {cursor}, old_cursor: {old_cursor}"
			if prev_confidence is not None:
				filled_in_confidence[old_cursor:cursor] = prev_confidence
			prev_confidence = conf
		filled_in_confidence[cursor:] = prev_confidence
		assert (filled_in <= 1).all(), filled_in
		filled_ins.append(filled_in)
		assert (filled_in_confidence <= 1).all(), filled_in_confidence
		filled_in_confidences.append(filled_in_confidence)

	assert len(lens) == len(filled_ins) == len(filled_in_confidences)

	lens = lens_to_plot
	# lens /= len(seqs)
	print("lens", lens)
	# assert lens[0] == 1, lens[0]

	filled_ins_equal_lengths = [np.concatenate((item, np.zeros(max_length-len(item)))) for item in filled_ins]
	filled_ins_equal_lengths_stacked = np.stack(filled_ins_equal_lengths)
	# print("filled_ins_equal_lengths_stacked.shape", filled_ins_equal_lengths_stacked.shape)
	summed_chosen_packets = np.sum(filled_ins_equal_lengths_stacked, axis=0)
	fraction_of_chosen_packets_by_flows = summed_chosen_packets/len(seqs)
	print("fraction_of_chosen_packets_by_flows", fraction_of_chosen_packets_by_flows)
	# assert fraction_of_chosen_packets_by_flows[0] == 1, fraction_of_chosen_packets_by_flows[0]
	assert len(fraction_of_chosen_packets_by_flows) == max_length

	values_by_length = [list() for _ in range(max_length)]
	for filled_in_confidence in filled_in_confidences:
		for index, item in enumerate(filled_in_confidence):
			values_by_length[index].append(item)

	values_by_length = [np.array(item) for item in values_by_length]

	medians = np.array([np.median(item, axis=0) for item in values_by_length])
	first_quartiles = np.array([np.quantile(item, 0.25, axis=0) for item in values_by_length])
	third_quartiles = np.array([np.quantile(item, 0.75, axis=0) for item in values_by_length])
	tens_percentiles = np.array([np.quantile(item, 0.1, axis=0) for item in values_by_length])
	ninetieth_percentiles = np.array([np.quantile(item, 0.9, axis=0) for item in values_by_length])

	all_legends = []
	# lens = [item.shape[0] for item in values_by_length]

	fig, ax1 = plt.subplots(figsize=(5,3))
	x_values = list(range(min(len(lens), MAX_X)))
	ret = ax1.bar(x_values, lens[:MAX_X], width=1, color="gray", alpha=0.2, label="fraction of samples")
	# ret4 = ax1.bar(x_values, fraction_of_chosen_packets_by_flows[:MAX_X], width=1, color="red", alpha=0.2, label="fraction of chosen samples")
	ret4 = ax1.bar(x_values, summed_chosen_packets[:MAX_X], width=1, color="red", alpha=0.2, label="fraction of chosen samples")


	ax2 = ax1.twinx()

	ax2.set_ylabel('Confidence')
	ax1.set_ylabel("Flows with given length (gray)\nChosen packets (red)")

	ax1.yaxis.tick_right()
	ax1.yaxis.set_label_position("right")
	ax2.yaxis.tick_left()
	ax2.yaxis.set_label_position("left")

	ret = ax2.plot(medians[:MAX_X], color=colors[0], label="Median")

	ret2 = ax2.fill_between(x_values, first_quartiles[:MAX_X], third_quartiles[:MAX_X], alpha=0.5, edgecolor=colors[0], facecolor=colors[0], label="1st and 3rd quartile")
	plt.autoscale(False)
	ret3 = ax2.fill_between(x_values, tens_percentiles[:MAX_X], ninetieth_percentiles[:MAX_X], alpha=0.2, edgecolor=colors[0], facecolor=colors[0], label="10th and 90th percentile")

	all_legends += ret
	all_legends.append(ret2)
	all_legends.append(ret3)

	all_labels = [item.get_label() for item in all_legends]
	plt.legend(all_legends, all_labels, loc='upper right', bbox_to_anchor=(1,0.95))

	ax2.set_ylabel_legend(all_legends[0])
	# ax1.set_ylabel_legend(Rectangle((0,0), 1, 1, fc='gray', alpha=0.2), handlelength=0.7, handletextpad=0.4)

	ticks = plt.xticks()
	plt.xticks([ tick for tick in ticks[0][1:-1] if tick.is_integer() ])

	if SHOW_TITLE:
		plt.title(reverse_mapping[attack_type])
	ax1.set_xlabel('Packet $n$')
	plt.xlim((-0.5,max(x_values)+0.5))
	plt.tight_layout()

	os.makedirs(DIR_NAME, exist_ok=True)
	plt.savefig(DIR_NAME+'/{}_{}_{}.pdf'.format(file_name.split("/")[-1], attack_type, reverse_mapping[attack_type].replace("/", "-").replace(":", "-")), bbox_inches = 'tight', pad_inches = 0)
	plt.clf()


