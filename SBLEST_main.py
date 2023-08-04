# An example code for classifying single-trial EEG data using SBLEST
from SBLEST_model import SBLEST, computer_acc, Enhanced_cov
import torch
from scipy.io import loadmat
from torch import DoubleTensor

# Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tau = 1
K = 2
Epoch = 5000

if __name__ == '__main__':
    # Load data: subject L1 from Dataset II ( "foot" vs "tongue")

    file_train = './Dataset2_L1_FootTongue_train.mat'
    file_test = './Dataset2_L1_FootTongue_test.mat'

    data_train = loadmat(file_train, mat_dtype=True)
    data_test = loadmat(file_test, mat_dtype=True)

    X_train = DoubleTensor(data_train['X_train']).to(device)
    Y_train = DoubleTensor(data_train['Y_train']).to(device)
    X_test = DoubleTensor(data_test['X_test']).to(device)
    Y_test = DoubleTensor(data_test['Y_test']).to(device)

    # Training stage: run SBLEST on the training set
    print('\n', 'FIR filter order: ', str(K), '      Time delay: ', str(tau))
    W, alpha, V, Wh = SBLEST(X_train, Y_train, K, tau, Epoch)

    # Test stage : predict labels in the test set
    R_test, _ = Enhanced_cov(X_test, K, tau, Wh, train=0)
    vec_W = W.T.flatten()   # vec operation (Torch)
    predict_Y = R_test @ vec_W
    accuracy = computer_acc(predict_Y, Y_test)
    print('Test   Accuracy: ', str(accuracy))



