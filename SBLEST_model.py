import torch
import warnings
import numpy as np
from torch import reshape, norm, zeros, eye, float64, mm, inverse, log, det
import numpy as np
from torch import linalg, diag, log
from torch import zeros, float64, mm, DoubleTensor

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def SBLEST(X, Y, K, tau, Epoch=5000, epoch_print=100):
    """
    SBLEST    : Sparse Bayesina Learning for End-to-end Spatio-Temporal-filtering-based single-trial EEG classification

    --- Parameters ---
    Y         : True label vector. [M, 1].
    X         : M trials of C (channel) x T (time) EEG signals. [C, T, M].
    K         : Order of FIR filter.
    tau       : Time delay parameter.

    --- Returns ---
    W         : Estimated low-rank weight matrix. [K*C, K*C].
    alpha     : Classifier weights. [L, 1].
    V         : Spatio-temporal filter matrix. [K*C, L].
                Each column of V represents a spatio-temporal filter.
    Wh        : Whitening matrix for enhancing covariance matrices (required for prediction on test set). [(K*C)^2, (K*C)^2].

    Reference:
    "W. Wang, F. Qi, D. Wipf, C. Can, T. Yu, Z. Gu, Y. Li, Z. Yu, W. Wu. Sparse Bayesian Learning for End-to-End EEG Decoding
    (accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence)."

    Wenlong Wang, Feifei Qi, Wei Wu, 2023.
    Email: 201710102248@mail.scut.edu.cn
    """

    # Compute enhanced covariance matrices and whitening matrix
    R_train, Wh = Enhanced_cov(X, K, tau, train=1)
    # print('\n')

    # Check properties of R
    M, D_R = R_train.shape  # M: number of samples; D_R: dimension of vec(R_m)
    KC = round(np.sqrt(D_R))
    Loss_old = 1e12
    threshold = 0.05
    r2_list = []

    assert D_R == KC ** 2, "ERROR: Columns of A do not align with square matrix"

    # Check if R is symmetric
    for j in range(M):
        row_cov = reshape(R_train[j, :], (KC, KC))
        row_cov = (row_cov + row_cov.T) / 2
        assert norm(row_cov - row_cov.T) < 1e-4, "ERROR: Measurement row does not form symmetric matrix"

    # Initializations
    U = zeros(KC, KC, dtype=float64).to(device)  # estimated low-rank matrix W initialized to be Zeros
    Psi = eye(KC, dtype=float64).to(device)  # covariance matrix of Gaussian prior distribution is initialized to be unit diagonal matrix
    lambda_noise = 1.  # variance of the additive noise set to 1

    # Optimization loop
    for i in range(Epoch+1):

        # update B,Sigma_y,u
        RPR = zeros(M, M, dtype=float64).to(device)
        B = zeros(KC ** 2, M, dtype=float64).to(device)
        for j in range(KC):
            start = j * KC
            stop = start + KC
            Temp = mm(Psi, R_train[:, start:stop].T)
            B[start:stop, :] = Temp
            RPR = RPR + mm(R_train[:, start:stop], Temp)
        Sigma_y = RPR + lambda_noise * eye(M, dtype=float64).to(device)
        uc = mm(mm(B, inverse(Sigma_y)), Y)  # maximum a posterior estimation of uc
        Uc = reshape(uc, (KC, KC))
        U = (Uc + Uc.T) / 2
        u = U.T.flatten()  # vec operation (Torch)

        # Update Phi (dual variable of Psi)
        Phi = []
        SR = mm(inverse(Sigma_y), R_train)
        for j in range(KC):
            start = j * KC
            stop = start + KC
            Phi_temp = Psi - Psi @ R_train[:, start:stop].T @ SR[:, start:stop] @ Psi
            Phi.append(Phi_temp)

        # Update Psi
        PHI = 0
        UU = 0
        for j in range(KC):
            PHI = PHI + Phi[j]
            UU = UU + U[:, j].reshape(-1, 1) @ U[:, j].reshape(-1, 1).T
        # UU = U @ U.T
        Psi = ((UU + UU.T) / 2 + (PHI + PHI.T) / 2) / KC    # make sure Psi is symmetric

        # Update theta (dual variable of lambda)
        theta = 0
        for j in range(KC):
            start = j * KC
            stop = start + KC
            theta = theta + (Phi[j] @ R_train[:, start:stop].T @ R_train[:, start:stop]).trace()

        # Update lambda
        lambda_noise = ((norm(Y - (R_train @ u).reshape(-1, 1), p=2) ** 2).sum() + theta) / M

        # Convergence check
        Loss = Y.T @ inverse(Sigma_y) @ Y + log(det(Sigma_y))
        delta_loss = abs(Loss_old - Loss.cpu().numpy()) / abs(Loss_old)
        if delta_loss < 2e-4:
            print('EXIT: Change in loss below threshold')
            break
        Loss_old = Loss.cpu().numpy()
        if i % epoch_print == 99:
            print('Iterations: ', str(i+1), '  lambda: ', str(lambda_noise.cpu().numpy()), '  Loss: ', float(Loss.cpu().numpy()),
                  '  Delta_Loss: ', float(delta_loss))

    # Eigen-decomposition of W
    W = U
    D, V_all = torch.linalg.eig(W)
    D, V_all = D.double().cpu().numpy(), V_all.double().cpu().numpy()
    idx = D.argsort()
    D = D[idx]
    V_all = V_all[:, idx]       # each column of V represents a spatio-temporal filter
    alpha_all = D

    # Determine spatio-temporal filters V and classifier weights alpha
    d = np.abs(alpha_all)
    d_max = np.max(d)
    w_norm = d / d_max      # normalize eigenvalues of W by the maximum eigenvalue
    index = np.where(w_norm > threshold)[0]    # indices of selected V according to a pre-defined threshold,.e.g., 0.05
    V = V_all[:, index]
    alpha = alpha_all[index]

    return W, alpha, V, Wh


def matrix_operations(A):
    """Calculate the -1/2 power of matrix A"""

    V, Q = linalg.eig(A)
    V_inverse = diag(V ** (-0.5))
    A_inverse = mm(mm(Q, V_inverse), linalg.inv(Q))

    return A_inverse.double()


def logm(A):
    """Calculate the matrix logarithm of matrix A"""

    V, Q = linalg.eig(A)  # V为特征值,Q为特征向量
    V_log = diag(log(V))
    A_logm = mm(mm(Q, V_log), linalg.inv(Q))

    return A_logm.double()


def computer_acc(predict_Y, Y_test):
    """Compute classification accuracy for test set"""

    predict_Y = predict_Y.cpu().numpy()
    Y_test = torch.squeeze(Y_test).cpu().numpy()
    total_num = len(predict_Y)
    error_num = 0

    # Compute classification accuracy for test set
    Y_predict = np.zeros(total_num)
    for i in range(total_num):
        if predict_Y[i] > 0:
            Y_predict[i] = 1
        else:
            Y_predict[i] = -1

    # Compute classification accuracy
    for i in range(total_num):
        if Y_predict[i] != Y_test[i]:
            error_num = error_num + 1

    accuracy = (total_num - error_num) / total_num
    return accuracy


def Enhanced_cov(X, K, tau, Wh=None, train=1):
    """
    Compute enhanced covariance matrices

    --- Parameters ---
    X         : M trials of C (channel) x T (time) EEG signals. [C, T, M].
    K         : Order of FIR filter
    tau       : Time delay parameter
    Wh        : Whitening matrix for enhancing covariance matrices.
                In training mode(train=1), Wh will be initialized as following python_code.
                In testing mode(train=0), Wh will receive the concrete value.
    train     : train = 1 denote training mode, train = 0 denote testing mode.

    --- Returns ---
    R         : Enhanced covariance matrices. [M,(K*C)^2*(K*C)^2 ]
    Wh : Whitening matrix. [(K*C)^2, (K*C)^2].
    """

    # Initialization, [KC, KC]: dimension of augmented covariance matrix
    X_order_k = None
    C, T, M = X.shape
    Cov = []
    Sig_Cov = zeros(K * C, K * C).to(device)

    for m in range(M):
        X_m = X[:, :, m]
        X_m_hat = DoubleTensor().to(device)

        # Generate augmented EEG data
        for k in range(K):
            n_delay = k * tau
            if n_delay == 0:
                X_order_k = X_m.clone()
            else:
                X_order_k[:, 0:n_delay] = 0
                X_order_k[:, n_delay:T] = X_m[:, 0:T - n_delay].clone()
            X_m_hat = torch.cat((X_m_hat, X_order_k), 0)

        # Compute covariance matrices
        R_m = mm(X_m_hat, X_m_hat.T)

        # Trace normalization
        R_m = R_m / R_m.trace()
        Cov.append(R_m)

        Sig_Cov = Sig_Cov + R_m

    # Compute Whitening matrix (Rp).
    if train == 1:
        Wh = Sig_Cov / M

    # Whitening, logarithm transform, and Vectorization
    Cov_whiten = zeros(M, K * C, K * C, dtype=float64).to(device)
    R_train = zeros(M, K * C * K * C, dtype=float64).to(device)

    for m in range(M):
        # progress_bar(m, M)

        # whitening
        Wh_inverse = matrix_operations(Wh)  # Rp^(-1/2)
        temp_cov = Wh_inverse @ Cov[m] @ Wh_inverse
        Cov_whiten[m, :, :] = (temp_cov + temp_cov.T) / 2
        R_m = logm(Cov_whiten[m, :, :])
        R_m = R_m.reshape(R_m.numel())  # column-wise vectorization
        R_train[m, :] = R_m

    return R_train, Wh