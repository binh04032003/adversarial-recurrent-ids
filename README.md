# reinforcement-learning-for-packet-sampling
Contact: Maximilian Bachl, Fares Meghdouri

This repository contains the code, the data and the machine learning models for our upcoming paper called *SparseIDS: Learning Packet Sampling with Reinforcement Learning*.

# Dataset
We use the CIC-IDS-2017 dataset. The repository contains a preprocessed version of the dataset (refer to the paper for more information). 

```flows.pickle``` is the file containing preprocessed flows of CIC-IDS-2017. Due to size constraints on GitHub they had to be split. Restore ```flows.pickle``` as follows:
* Concatenate the parts: ```cat flows.pickle.gz.part-* > flows.pickle.gz```
* Unzip them: ```gzip -d flows.pickle.gz```

# Requirements
Our code needs Python >= 3.6 as well as the following pip packages:
```
matplotlib==3.0.2
numpy==1.16.0
scikit-learn==0.21.2
scipy==1.3.0
tensorboard==1.12.2
tensorboardX==1.6
tensorflow==1.12.0
torch==1.1.0
tqdm==4.30.0
```

# Usage examples


# Trained models
Models in the [runs](runs) folder have been trained with the following configuration:
* Jan18_13-57-36_gpu: tradeoff 0.1, continuous actions
* Jan18_13-57-46_gpu: tradeoff 0.2, continuous actions
* Jan18_14-06-48_gpu: tradeoff 0.5, continuous actions
* Jan20_16-40-14_gpu: random with sparsity of 76.3%
* Jan20_16-40-27_gpu: uniform with sparsity of 76.3%
* Jan20_16-40-51_gpu: first_n with sparsity of 76.3%
* Jan20_16-41-01_gpu: first_n_equal with sparsity of 76.3%
* Jan21_14-11-40_gpu: train normally
* Jan22_13-55-26_gpu: tradeoff 0.1, discrete actions 20 steps
* Feb06_23-17-09_gpu: tradeoff 1.0, continuous actions
* Feb07_17-08-56_gpu: variable tradeoff, continuous actions
