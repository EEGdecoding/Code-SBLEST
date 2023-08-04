# Code-SBLEST

This repo contains Matlab and Python code for the SBLEST (Sparse Bayesian Learning for End-to-End Spatio-Temporal-Filtering-Based Single-Trial EEG Classification) algorithm, as well as implementations of Convolutional Neural Networks (CNNs) used in the paper. Detailed information about the algorithms and CNN implementations can be found in [W. Wang, F. Qi, D. Wipf, C. Can, T. Yu, Z. Gu, Y. Li, Z. Yu, W. Wu. Sparse Bayesian Learning for End-to-End EEG Decoding. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023, 10.1109/tpami.2023.3299568](https://doi.org/10.1109/tpami.2023.3299568))". 

## Data
The data used in this repository is from Subject L1 (foot vs. tongue) in Dataset II, as mentioned in the referenced paper.

### File Descriptions

* [Dataset2_L1_FootTongue_train.mat](https://github.com/EEGdecoding/Code-SBLEST/blob/main/Dataset2_L1_FootTongue_train.mat) — This file contains the training data used in this repository.
* [Dataset2_L1_FootTongue_test.mat](https://github.com/EEGdecoding/Code-SBLEST/blob/main/Dataset2_L1_FootTongue_test.mat) —  This file contains the test data used in this repository.

## Matlab code for SBLEST

The MATLAB scripts provided in this section implement the SBLEST algorithm and have been tested with MATLAB R2018b.

### File Descriptions

* [SBLEST.m](https://github.com/EEGdecoding/Code-SBLEST/blob/main/SBLEST.m)                                           —Matlab code for the SBLEST algorithm.

* [SBLEST_main.m](https://github.com/EEGdecoding/Code-SBLEST/blob/main/SBLEST_main.m)  — An example code for classifying single-trial EEG data using SBLEST in Matlab.

### Usage

1. To run the code, download and extract them into a folder of your choice, and navigate to this folder within MATLAB. 

2. At the MATLAB command line, type 
 ```
 SBLEST_main
 ```


## Python code for SBLEST

The Python scripts for SBLEST are implemented in PyTorch and have been fully tested.

### File Descriptions

* [SBLEST_model.py](https://github.com/EEGdecoding/Code-SBLEST/blob/main/SBLEST_model.py)                             —Python code for the SBLEST algorithm.

* [SBLEST_main.py](https://github.com/EEGdecoding/Code-SBLEST/blob/main/SBLEST_main.py)  — An example code for classifying single-trial EEG data using SBLEST in Python.



 
 ## Python Implementations of sCNN, dCNN, EEGNet, EEG-Inception, and EEGSym
 
sCNN and dCNN are implemented in PyTorch using the braindecode package, which is provided at https://github.com/robintibor/braindecode.

EEGNet is implemented in TensorFlow using the Keras API, with the model provided at https://github.com/vlawhern/arl-eegmodels.

EEG-inception and EEGSym are also implemented in TensorFlow, with the models provided at https://github.com/esantamariavazquez/EEGInception and https://github.com/Serpeve/EEGSym, respectively.

### File Descriptions

* [sCNN_main.py](https://github.com/EEGdecoding/Code-SBLEST/blob/main/sCNN_main.py)   — An example code for classifying single-trial EEG data using sCNN.

* [dCNN_main.py](https://github.com/EEGdecoding/Code-SBLEST/blob/main/dCNN_main.py)   — An example code for classifying single-trial EEG data using dCNN.

* [EEGNet_main.py](https://github.com/EEGdecoding/Code-SBLEST/blob/main/EEGNet_main.py)   — An example code for classifying single-trial EEG data using EEGNet.
* [EEGModels.py](https://github.com/EEGdecoding/Code-SBLEST/blob/main/EEGModels.py)   — A model file used in the EEGNet implementation.

* [EEGInception_main.py](https://github.com/EEGdecoding/Code-SBLEST/blob/main/EEGInception_main.py)   — An example code for classifying single-trial EEG data using EEG-inception.
* [EEGInception.py](https://github.com/EEGdecoding/Code-SBLEST/blob/main/EEGInception.py)   — A model file used in the EEG-inception implementation.

* [EEGSym_main.py](https://github.com/EEGdecoding/Code-SBLEST/blob/main/EEGSym.py)   — An example code for classifying single-trial EEG data using EEGSym.
* [EEGSym_architecture.py](https://github.com/EEGdecoding/Code-SBLEST/blob/main/EEGSym_architecture.py)   — A model file used in the EEGSym implementation.
* [EEGSym_DataAugmentation.py](https://github.com/EEGdecoding/Code-SBLEST/blob/main/EEGSym_DataAugmentation.py)   — A python file for data augmentation used in the EEGSym implementation.

* [signal_target.py](https://github.com/EEGdecoding/Code-SBLEST/blob/main/signal_target.py)   — A code for preprocessing the signal and target used in all the cNNs implementations.



