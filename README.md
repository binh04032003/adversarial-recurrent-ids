# adversarial-recurrent-ids
Contact: Alexander Hartl, Maximilian Bachl. 

This repository contains the code and the figures for the [upcoming paper](https://arxiv.org/abs/1912.09855) dealing with Explainability and Adversarial Robustness for RNNs. We perform our study in the context of Intrusion Detection Systems.

# Datasets
We use two datasets in the paper: CIC-IDS-2017 and UNSW-NB-15. The repository contains a preprocessed version of the datasets (refer to the paper for more information). 

```flows.pickle``` is for CIC-IDS-2017 while ```flows15.pickle``` is for UNSW-NB-15. Due to size constraints on GitHub they had to be split. Restore ```flows.pickle``` as follows:
* Concatenate the parts: ```cat flows.pickle.gz.part-* > flows.pickle.gz```
* Unzip them: ```gzip -d flows.pickle.gz```

Proceed analogously for ```flows15.pickle```. Information on how to reproduce the preprocessed datasets can be found in the [Datasets-preprocessing](https://github.com/CN-TU/Datasets-preprocessing) repository.

# Trained models
Models in the [runs](runs) folder have been trained with the following configurations:
* Oct26_00-03-50_gpu: CIC-IDS-2017
* Oct28_15-41-46_gpu: UNSW-NB-15
* Nov19_18-25-03_gpu: CIC-IDS-2017 with feature dropout
* Nov19_18-25-53_gpu: UNSW-NB-15 with feature dropout
* Nov20_18-27-31_gpu: CIC-IDS-2017 with adversarial training using L1 distance and CW with kappa=1
* Nov20_18-28-17_gpu: UNSW-NB-15 with adversarial training using L1 distance and CW with kappa=1
