# reinforcement-learning-for-packet-sampling
Contact: Maximilian Bachl, Fares Meghdouri

This repository contains the code, the data and the machine learning models for our upcoming paper called *SparseIDS: Learning Packet Sampling with Reinforcement Learning*.

# Dataset
We use the CIC-IDS-2017 dataset. The repository contains a preprocessed version of the dataset (refer to the paper for more information). 

```flows.pickle``` is the file containing preprocessed flows of CIC-IDS-2017. Due to size constraints on GitHub they had to be split. Restore ```flows.pickle``` as follows:
* Concatenate the parts: ```cat flows.pickle.gz.part-* > flows.pickle.gz```
* Unzip them: ```gzip -d flows.pickle.gz```

<!--
# Trained models
Models in the [runs](runs) folder have been trained with the following configuration:
* Oct26_00-03-50_gpu: CIC-IDS-2017
* Oct28_15-41-46_gpu: UNSW-NB-15
* Nov19_18-25-03_gpu: CIC-IDS-2017 with feature dropout
* Nov19_18-25-53_gpu: UNSW-NB-15 with feature dropout
* Nov20_18-27-31_gpu: CIC-IDS-2017 with adversarial training using L1 distance and CW with kappa=1
* Nov20_18-28-17_gpu: UNSW-NB-15 with adversarial training using L1 distance and CW with kappa=1
-->
