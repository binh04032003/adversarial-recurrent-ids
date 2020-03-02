# SparseIDS: Learning Packet Sampling with Reinforcement Learning
Contact: Maximilian Bachl, Fares Meghdouri

This repository contains the code, the data and the machine learning models for our upcoming paper called *SparseIDS: Learning Packet Sampling with Reinforcement Learning*.

# Dataset
We use the CIC-IDS-2017 dataset. The repository contains a preprocessed version of the dataset (refer to the paper for more information). 

```flows.pickle``` is the file containing preprocessed flows of CIC-IDS-2017. Due to size constraints on GitHub it had to be split. Restore ```flows.pickle``` as follows:
* Concatenate the parts: ```cat flows.pickle.gz.part-* > flows.pickle.gz```
* Unzip them: ```gzip -d flows.pickle.gz```

If you want to produce the preprocessed files yourself: 
* Follow the information on how to reproduce the preprocessed datasets in the [Datasets-preprocessing](https://github.com/CN-TU/Datasets-preprocessing) repository.
* Run the ```parse.py``` script on the resulting ```.csv``` file to get the final ```.pickle``` file. 

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

## RL
To train a model with a tradeoff of 0.1 and continuous actions run the following:

    ./learn.py --dataroot flows.pickle --function train_rl --device cpu --batchSize 32 --accuracySparsityTradeoff 0.1 --continuous
 
To instead train with 20 discrete actions at each time step remove ```--continuous``` and add for example ```--lookaheadSteps 20```

To test the resulting model run: 

    ./learn.py --dataroot flows.pickle --function test --rl --sampling rl --device cpu --net runs/<path_to_neural_network_weights>/lstm_module_<epoch>.pth --net_actor runs/<path_to_neural_network_weights>/lstm_module_rl_actor_<epoch>.pth --continuous
    
Use the ```--maxSize <number>``` flag to lower the size of the test dataset to speed up the process. For example, 10000 will limit the test dataset to 10000 samples. 

## Other sampling techniques

Testing outputs several accuracy metrics as well as the resulting sparsity. To train a corresponding model with a classic sampling approach run:

    ./learn.py --dataroot flows.pickle --function train --rl --sampling <sampling_discipline> --samplingProbability <number_between_0_and_1>

Implemented sampling disciplines are ```uniform``` (every *i*th in the paper), ```random```, ```first_n``` (first *m* relative in the paper) and ```first_n_equal``` (first *m* in the paper). 

For testing them run 

    ./learn.py --dataroot flows.pickle --function test --rl --sampling <sampling_discipline> --device cpu --net runs/<path_to_neural_network_weights>/lstm_module_<epoch>.pth --samplingProbability <number_between_0_and_1>

## Steering

To use the steering functionalty you first have to train with a variable tradeoff:

    ./learn.py --dataroot flows.pickle --function train_rl --device cpu --batchSize 32 --variableTradeoff --continuous
    
Then you have to evaluate the model:

    ./learn.py --dataroot flows.pickle --function test --rl --sampling rl --device cpu --net runs/<path_to_neural_network_weights>/lstm_module_<epoch>.pth --net_actor runs/<path_to_neural_network_weights>/lstm_module_rl_actor_<epoch>.pth --continuous --variableTradeoff --steering_step_size 0.1 --global_tradeoff 1.0 --batches_to_consider_for_steering 1000

# Plotting 

## Chosen packets

To plot Figure 5 from the paper you need to use the ```prediction_outcomes``` file produced by the ```test``` function as follows:

    ./plot_rl.py <name_of_file>_prediction_outcomes_0_3.pickle
    
## Steering

If you test a model with the variable tradeoff option (see above), it will output a ```<name>_sampling_rl.json``` file. To create the plot of Figure 6 run:

    ./plot_steering.py <name>_sampling_rl.json
    
## Training 

To plot training (Figure 4) run

    ./plot_training.py runs/<path_to_neural_network_weights>/runs_extracted/events.out.tfevents.<timestamp>.gpu
    
It is, however, easier to simply use ```tensorboard``` to visualize training. 

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
