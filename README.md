# Code-SBLEST

This repo provides the MATLAB code for the Sparse Bayesian Learning for End-to-end Spatio-Temporal-filtering-based single-trial EEG
classification (SBLEST) algorithm, which is presented in “W. Wang, F. Qi, D. Wipf, C. Can, T. Yu, Z. Gu, E. N. Brown, Z. Yu, W. Wu. Sparse Bayesian Learning for End-to-End EEG Decoding (under review)”. 

All scripts are written in MATLAB and have been tested with MATLAB R2018b.

## File Descriptions

* [SBLEST.m](https://github.com/EEGdecoding/Code-SBLEST/blob/main/SBLEST.m)                                                — Code for the SBLEST algorithm.

* [SBLEST_main.m](https://github.com/EEGdecoding/Code-SBLEST/blob/main/SBLEST_main.m)  — An example code for classifying single-trial EEG data using SBLEST.

* [Dataset2_L1_FootTongue_train.mat](https://github.com/EEGdecoding/Code-SBLEST/blob/main/Dataset2_L1_FootTongue_train.mat) — Training data used in SBLEST_main.m. The data is from Subject L1 (foot vs. tongue) in Dataset II used in the paper.

* [Dataset2_L1_FootTongue_test.mat](https://github.com/EEGdecoding/Code-SBLEST/blob/main/Dataset2_L1_FootTongue_test.mat) —  Test data used in SBLEST_main.m.

## Usage

1. To run the code, download and extract them into a folder of your choice, and navigate to this folder within MATLAB. 

2. At the MATLAB command line, type 
 ```
 SBLEST_main
 ```
