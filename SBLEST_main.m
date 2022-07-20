clc; clear; close all;

%% load data : subject L1 from Dataset II ( "foot" vs "tongue")
load('Dataset2_L1_FootTongue_train.mat');
load('Dataset2_L1_FootTongue_test.mat');
tau_selected = 1; % "tau_selected =1" for this subset is selected from CV
%% Initialization
Maxiters = 5000; % Pre-defined maximum iterations (5000 in this paper)
e = 2e-4;% Pre-defined threshold for break condition (2e-4 in this papter)
tau = tau_selected;
if tau==0
    K=1;
else
    K=2;
end
%% Trainning stage: run SBLEST 
disp(['Order: ', num2str(K),  '      Time delay: ', num2str(tau)]);
disp('Running SBLEST : update W, bias, psi and lambda');
[W, alpha,V,Cov_mean_train] = SBLEST(X_train, Y_train, Maxiters,K,tau,e);
%% Test stage : predicte test label
[R_test] = Enhanced_cov_test(X_test,K,tau,Cov_mean_train);
predict_Y = R_test*vec(W);
[accuracy] = compute_acc (predict_Y, Y_test);
disp(['Test   Accuracy: ', num2str(accuracy)]);

 
function [R_test] = Enhanced_cov_test(X,K,tau,Cov_mean_train)
% ************************************************************************
% Compute Enhanced covariace matrix of test set
% Input :
% K         : Order of FIR filter
% tau       : time delay parameter
% X         : M EEG signals from test set

% Output    :
% R_test         :Enhanced covariace matrix of test set
% ************************************************************************
%  Initializaiton 
M = length(X); 
[C,T] = size(X{1,1});
KC = K*C;
Cov = cell(1,M);
Sig_Cov = zeros(KC,KC);

for m = 1:M
    X_m = X{1,m};
    X_m_hat = [];  
    % Generate augument EEG data
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
     % Compute covariance and trace normalizaiton
    Cov{1,m} = X_m_hat*X_m_hat';
    Cov{1,m}= Cov{1,m}/trace(Cov{1,m}); 
    Sig_Cov = Sig_Cov + Cov{1,m};
end

% Whitenning, logarithm transform and reshape
Cov_whiten = zeros(M,KC,KC);
for m = 1:M
    Cov_whiten(m,:,:) = Cov_mean_train^(-1/2)*Cov{1,m}*Cov_mean_train^(-1/2);
    R_test(m,:) =  vec(logm(squeeze(Cov_whiten(m,:,:))));
end 
end


function [accuracy] = compute_acc (predict_Y, Y_test)
% Calculate class label 
Y_predict = zeros(length(predict_Y),1);
for i = 1:length(predict_Y)
    if (predict_Y(i) > 0)
        Y_predict(i) = 1;
    else
        Y_predict(i) = -1;
    end
end      
% Compute accuracy
error_num = 0; 
total_num = length(predict_Y);
for i = 1:total_num
    if (Y_predict(i) ~= Y_test(i))
        error_num = error_num + 1;
    end
end
accuracy = (total_num-error_num)/total_num;
end
