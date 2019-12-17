#!/usr/bin/env python3

import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
import json
from learn import pretty_print

hidden_size = 512
n_layers = 3
absolute_values = True

assert len(sys.argv) > 1, 'No model specified'

parameters = torch.load(sys.argv[1])

with open('features_meaningful_names.json', 'r') as f:
	feature_array = json.load(f)

#	parameters = lstm_module.state_dict()
weights_hh = [ parameters['lstm.weight_hh_l%d' % layer].cpu().numpy().reshape(4,hidden_size,hidden_size).sum(axis=0) for layer in range(n_layers) ]
weights_ih = [ parameters['lstm.weight_ih_l%d' % layer].cpu().numpy().reshape(4,hidden_size,-1).sum(axis=0) for layer in range(n_layers) ]
h2o = parameters['h2o.weight'].cpu().numpy()

if absolute_values:
	weights_hh = [ np.abs(weights) for weights in weights_hh ]
	weights_ih = [ np.abs(weights) for weights in weights_ih ]
	h2o = np.abs(h2o)
	

input_featimp = [ weights_ih[0] ]
for i in range(1,n_layers):
	input_featimp.append(np.matmul(weights_ih[i], input_featimp[-1]))
input_featimp.append(np.matmul(h2o, input_featimp[-1]))

output_featimp = [ h2o ]
for i in range(n_layers-1,-1,-1):
	output_featimp.append(np.matmul(output_featimp[-1], weights_ih[i]))
output_featimp.reverse()
#  assert (input_featimp[-1] == output_featimp[0]).all()

fi_over_time = [ np.abs(input_featimp[-1][0,:]) ]
hh_prod = weights_hh
for _ in range(10):
	hh_prod = [ np.matmul(hh_prod[i], weights_hh[i]) for i in range(n_layers) ]
	fi_over_time.append(np.abs(np.sum([ np.matmul(output_featimp[i+1], np.matmul(hh_prod[i], input_featimp[i])) for i in range(n_layers)], axis=0)[0,:]))

print ('Packet feature importance:')

for feature_index, feature_importance in sorted(enumerate(fi_over_time[0]/np.sum(fi_over_time[0])), key=lambda item: item[1]):
	pretty_print("weight feat. imp. for", feature_index, feature_array[feature_index], feature_importance)
	
# Highly experimental stuff
decomposition = [ np.linalg.eig(weights_hh[i]) for i in range(n_layers) ]
valid_eigenvalues = [ [ j for j in range(hidden_size) if np.imag(decomposition[i][0][j]) == 0 ] for i in range(n_layers) ]

max_eigenvalues = [ max(valid_eigenvalues[i], key=lambda j: np.abs(decomposition[i][0][j])) for i in range(n_layers) ]
layer_with_max_eigenvalue = max(range(n_layers), key=lambda i: decomposition[i][0][max_eigenvalues[i]])
#print ('Max eigen value:', decomposition[layer_with_max_eigenvalue][0][max_eigenvalues[layer_with_max_eigenvalue]])
eigenvector = np.real(decomposition[layer_with_max_eigenvalue][1][max_eigenvalues[layer_with_max_eigenvalue]])
flow_feat_imp = np.abs(np.matmul(output_featimp[layer_with_max_eigenvalue+1], np.matmul(np.matmul(eigenvector[:,None], eigenvector[None,:]), input_featimp[layer_with_max_eigenvalue]))).flatten()
flow_feat_imp /= np.sum(flow_feat_imp)
print('Flow feature importance:')
for feature_index, feature_importance in sorted(enumerate(flow_feat_imp.tolist()), key=lambda item: item[1]):
	pretty_print("weight feat. imp. for", feature_index, feature_array[feature_index], feature_importance)

to_plot = np.stack(fi_over_time)
plt.semilogy(to_plot[:,:to_plot.shape[1]//2])
plt.semilogy(to_plot[:,(to_plot.shape[1]//2):], linestyle='--')
plt.legend(feature_array)
plt.show()
