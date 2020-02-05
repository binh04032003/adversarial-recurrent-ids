# adversarial-recurrent-ids
Contact: Alexander Hartl, Maximilian Bachl. 

This repository contains the code and the figures for the [upcoming paper](https://arxiv.org/abs/1912.09855) dealing with Explainability and Adversarial Robustness for RNNs. We perform our study in the context of Intrusion Detection Systems.

Unfortunately, due to size constraints, the datasets are missing. We'll look into ways of adding them or providing instructions for retrieving the datasets. 

Trained models
--------------
Models in the [runs](runs) folder have been trained with the following configuration:
* Oct26_00-03-50_gpu: CIC-IDS-2017
* Oct28_15-41-46_gpu: UNSW-NB-15
* Nov19_18-25-03_gpu: CIC-IDS-2017 with feature dropout
* Nov19_18-25-53_gpu: UNSW-NB-15 with feature dropout
* Nov20_18-27-31_gpu: CIC-IDS-2017 with adversarial training using L1 distance and CW with kappa=1
* Nov20_18-28-17_gpu: UNSW-NB-15 with adversarial training using L1 distance and CW with kappa=1
