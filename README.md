# Code-SBLEST

This repo contains MATLAB code for the SBLEST (Sparse Bayesian Learning for End-to-End Spatio-Temporal-Filtering-Based Single-Trial EEG Classification) algorithm, as well as implementations of Convolutional Neural Networks (CNNs) used in the paper. Detailed information about the algorithms and CNN implementations can be found in the "Sparse Bayesian Learning for End-to-End EEG Decoding' by W. Wang, F. Qi, D. Wipf, C. Can, T. Yu, Z. Gu, Y. Li, Z. Yu, and W. Wu(under review)". 

## Matlab code for SBLEST

All scripts SBLEST are written in MATLAB and have been tested with MATLAB R2018b.

### File Descriptions

* [SBLEST.m](https://github.com/EEGdecoding/Code-SBLEST/blob/main/SBLEST.m)                                                — Code for the SBLEST algorithm.

* [SBLEST_main.m](https://github.com/EEGdecoding/Code-SBLEST/blob/main/SBLEST_main.m)  — An example code for classifying single-trial EEG data using SBLEST.

* [Dataset2_L1_FootTongue_train.mat](https://github.com/EEGdecoding/Code-SBLEST/blob/main/Dataset2_L1_FootTongue_train.mat) — Training data used in SBLEST_main.m. The data is from Subject L1 (foot vs. tongue) in Dataset II used in the paper.

* [Dataset2_L1_FootTongue_test.mat](https://github.com/EEGdecoding/Code-SBLEST/blob/main/Dataset2_L1_FootTongue_test.mat) —  Test data used in SBLEST_main.m.

### Usage

1. To run the code, download and extract them into a folder of your choice, and navigate to this folder within MATLAB. 

2. At the MATLAB command line, type 
 ```
 SBLEST_main
 ```
 
 ## Python Implementation of sCNN, dCNN, EEGNet, EEG-Inception, and EEGSym
 
sCNN and dCNN are implemented in PyTorch using the braindecode package, which can be found at https://github.com/robintibor/braindecode.

EEGNet is implemented in TensorFlow using the Keras API, with the model provided in https://github.com/vlawhern/arl-eegmodels.

EEG-inception and EEGSym are also implemented in TensorFlow, with the models provided in https://github.com/esantamariavazquez/EEGInception and https://github.com/Serpeve/EEGSym respectively.

### File Descriptions


