#!/usr/bin/env python3

import pandas as pd
import numpy as np
import itertools

import pickle
import json

df = pd.read_csv('packet.csv')
mapping = dict([(b, a) for a, b in enumerate(sorted(set(list(df["Attack"]))))])

categories = ['Botnet', 'Brute Force', 'DoS', 'Infiltration', 'Normal', 'PortScan', 'Web Attack']

categories_mapping = {}
for key in mapping.keys():
	for category in categories:
		if category in key:
			if not category in categories_mapping:
				categories_mapping[category] = []
			categories_mapping[category].append(key)
			break
	else:
		if not key in categories_mapping:
			categories_mapping[key] = []
		categories_mapping[key].append(key)

with open("categories_mapping.json", "w") as f:
	json.dump({"categories_mapping": categories_mapping, "mapping": mapping}, f)

minimal = False
quic = False
only_unchangeable = False

all_flags = "SFRPAUECN"
all_flags_name = ["SYN", "FIN", "RST", "PSH", "ACK", "URG", "ECE", "CWR", "NS"]
assert len(all_flags) == len(all_flags_name)
assert [item_long[0]==item for item_long, item in zip(all_flags_name, all_flags)]

def read_list(l):
	if isinstance(l, float) and np.isnan(l): return []
	assert((l[0],l[-1]) == ('[', ']'))
	return l[1:-1].split(' ')

def read_numlist(l):
	return ( [float(item)] for item in read_list(l) )

def read_directionlist(l):
	return ( [int(item == 'true')] for item in read_list(l) )

def read_flaglist(l):
	return ( [ int(f in flags) for f in all_flags] for flags in read_list(l) )

const_feature_names = ["sourceTransportPort","destinationTransportPort","protocolIdentifier"]
ip_total = 'accumulate(ipTotalLength)'
inter_time = 'accumulate(_interPacketTimeNanoseconds)'
direction = 'accumulate(flowDirection)'
flags = "accumulate(_tcpFlags)"
wrote_json = False

def read_flow(row):
	global wrote_json
	features = []
	const_features = [ row[item] for item in const_feature_names ]

	generators = []
	if not minimal or quic:
		generators.append( (const_features for _ in itertools.count() ) )
		features += const_feature_names
	if not only_unchangeable:
		generators.append( read_numlist(row[ip_total]) )
		features.append(ip_total)
	# if not minimal:
	# 	generators.append( read_numlist(row['accumulate(ipTTL)']) )
	# if not minimal:
	# 	generators.append( read_numlist(row['accumulate(ipClassOfService)']) )
	if not only_unchangeable:
		generators.append( ([item[0]/1000000] for item in itertools.chain([[0]], read_numlist(row[inter_time])) ))
		features.append(inter_time)
	generators.append( read_directionlist(row[direction]) )
	features.append(direction)
	if not minimal:
		generators.append( read_flaglist(row[flags]) )
		features += [item for item in all_flags_name]
	generators.append(( [mapping[row["Attack"]], row["Label"]] for _ in itertools.count() ))
	if not wrote_json:
		with open("features.json", "w") as f:
			json.dump(features, f)
		wrote_json = True

	return [ list(itertools.chain(*feats)) for feats in zip(*generators) ]

if __name__=="__main__":
	flows = [None] * df.shape[0]

	for i, row in df.iterrows():
		if i%100000 == 0:
			print(i)
		flows[i] = np.array(read_flow(row), dtype=np.float32)

	pickle.dump(flows, open('flows.pickle', 'wb'))
