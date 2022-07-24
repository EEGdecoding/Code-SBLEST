%%% An example code for classifying single-trial EEG data using SBLEST
clc; clear; close all;

%% Load data: subject L1 from Dataset II ( "foot" vs "tongue")
load('Dataset2_L1_FootTongue_train.mat');
load('Dataset2_L1_FootTongue_test.mat');
tau_selected = 1; % this was determined based on 10-fold cross-validation on the training set

%% Initialization
tau = tau_selected;
if tau == 0
    K = 1;
else
    K = 2;
end
%% Training stage: run SBLEST on the training set
disp(['FIR filter order: ', num2str(K),  '      Time delay: ', num2str(tau)]);
disp('Running SBLEST : update W, Psi and lambda');
[W, alpha, V, Wh] = SBLEST(X_train, Y_train, K, tau);

%% Test stage : predicte labels in the test set
R_test = Enhanced_cov_test(X_test, K, tau, Wh);
predict_Y = R_test*vec(W);
accuracy = compute_acc (predict_Y, Y_test);
disp(['Test   Accuracy: ', num2str(accuracy)]);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function R_test = Enhanced_cov_test(X, K, tau, Wh)
% Compute enhanced covariace matrices of test set
%
% Inputs :
% X         : EEG signals ofm test set
% K         : Order of FIR filter
% tau       : Time delay parameter
% Wh        : Whitening matrix for enhancing covariance matrices

% Output    :
% R_test    : Enhanced covariace matrices of test set
% ************************************************************************
[C, T, M] = size(X);
KC = K*C; % [KC, KC]: dimension of augmented covariance matrix
Cov = cell(1, M);
Sig_Cov = zeros(KC, KC);
for m = 1:M
    X_m = X(:,:,m);
    X_m_hat = [];
    % Generate augumented EEG data
    for k = 1:K
        n_delay = (k-1)*tau;
        if n_delay ==0
            X_order_k = X_m;
        else
            X_order_k(:,1:n_delay) = 0;
            X_order_k(:,n_delay+1:T) = X_m(:,1:T-n_delay);
        end
        X_m_hat = cat(1,X_m_hat,X_order_k);
    end
    % Compute covariance and trace normalization
    Cov{1,m} = X_m_hat*X_m_hat';
    Cov{1,m} = Cov{1,m}/trace(Cov{1,m});
    Sig_Cov = Sig_Cov + Cov{1,m};
end

% Whitenning, logarithm transform, and vectorization
Cov_whiten = zeros(M, KC, KC);
for m = 1:M
    temp_cov = Wh^(-1/2)*Cov{1,m}*Wh^(-1/2);
    Cov_whiten(m,:,:) = (temp_cov + temp_cov')/2;
    R_m =logm(squeeze(Cov_whiten(m,:,:))); % logarithm transform
    R_m = R_m(:); % column-wise vectorization
    R_test(m,:) = R_m';
end
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function accuracy = compute_acc (predict_Y, Y_test)
% Compute classification accuracy for test set
Y_predict = zeros(length(predict_Y),1);
for i = 1:length(predict_Y)
    if (predict_Y(i) > 0)
        Y_predict(i) = 1;
    else
        Y_predict(i) = -1;
    end
end
% Compute classification accuracy
error_num = 0;
total_num = length(predict_Y);
for i = 1:total_num
    if (Y_predict(i) ~= Y_test(i))
        error_num = error_num + 1;
    end
end
accuracy = (total_num-error_num)/total_num;
end

