function [W, alpha, V, Cov_mean_train] = SBLEST(X, Y, Maxiters, K, tau, e)
% ************************************************************************
% SBLEST    : Spatio-Temporal-filtering-based single-trial EEG classification
%
% --- Inputs ---
% Y         : Observed label vector
% X         : M EEG signals from train set
%             M cells: Each X{1,i} represents a trial with size of [C*T]
% Maxiters  : Maximum number of iterations (5000 is suggested in this paper)
% K         : Order of FIR filter
% tau       : time delay parameter
% e         : Convergence threshold 
%
% --- Output ---
% W         : The estimated low-rank matrix
% alpha     : The classifier parameter (eigenvalues of W)
% V         : Each column of V represents a spatio-temporal filter (eigenvectors of W)
% Cov_mean_train : Averaged covariance matrix of train set (2 C^2 * 2 C^2)
%                  (using for whittening and logarithm transform in test stage)

% Reference:

% Copyright:

% ************************************************************************

[R_train, Cov_mean_train] = Enhanced_cov_train(X, K, tau);

%% Check properties of R
[M, D_R] = size(R_train); % M: # of samples; D_R: dimention of vec(R_m)
KC = round(sqrt(D_R));
epsilon = e;
Loss_old = 1e12;
threshold = 0.05; % 
if (D_R ~= KC^2)
    disp('ERROR: Columns of A do not align with square matrix');
    return;
end

% Check properties of R: symmetric ?
for c = 1:M
    row_cov = reshape(R_train(c,:), KC, KC);
    if ( norm(row_cov - row_cov','fro') > 1e-4 )
        disp('ERROR: Measurement row does not form symmetric matrix');
        return
    end
end

%% Initializations
U = zeros(KC,KC); % The estimated low-rank matrix W set to be Zeros
Psi = eye(KC); % the covariance matrix of Gaussian prior distribution is initialized to be Unit diagonal matrix
lambda = 1;% the variance of the additive noise set to 1 by default
 
%% Optimization loop
for i = 1:Maxiters
   %% Compute estimate of X 
    RPR = zeros(M, M); %  Predefined temporal variables RT*PSI*R
    B = zeros(KC^2, M); %  Predefined temporal variables
    for c = 1:KC
        start = (c-1)*KC + 1; stop = start + KC - 1;
        Temp = Psi*R_train(:,start:stop)'; 
        B(start:stop,:)= Temp; 
        RPR =  RPR + R_train(:,start:stop)*Temp;  
    end

    Sigma_y = RPR + lambda*eye(M); 
    u = B*( Sigma_y\Y ); % Maximum a posterior estimation of u
    U = reshape(u, KC, KC);
    U = (U + U')/2; % make sure U is symmetric
       
   %% Update the dual variables of PSI : PHi_i
    Phi = cell(1,KC);
    SR = Sigma_y\R_train;
    for c = 1:KC
        start = (c-1)*KC + 1; stop = start + KC - 1;
        Phi{1,c} = Psi - Psi * ( R_train(:,start:stop)' * SR(:,start:stop) ) * Psi;
    end
    
    %% Update covariance parameters Psi: Gx
    PHI = 0;    
    UU = 0;
    for c = 1:KC
        PHI = PHI +  Phi{1,c};
        UU = UU + U(:,c) * U(:,c)';
    end
    Psi = ( (UU + UU')/2 + (PHI + PHI')/2 )/KC; % make sure Psi is symmetric

   %% Update lambda
   theta = 0;
    for c = 1:KC
        start = (c-1)*KC + 1; stop = start + KC - 1;
        theta = theta +trace(Phi{1,c}* R_train(:,start:stop)'*R_train(:,start:stop)) ;
    end
    lambda = (sum((Y-R_train*u).^2) + theta)/M;  

   %% Output display and  convergence judgement
        Loss = Y'*Sigma_y^(-1)*Y + log(det(Sigma_y));         
        delta_loss = abs(Loss_old-Loss)/abs( Loss_old);
        if (delta_loss < epsilon)
            disp('EXIT: Change in Loss below threshold');
            break;
        end
        Loss_old = Loss;
         if (~rem(i,100))
            disp(['Iterations: ', num2str(i),  '  lambda: ', num2str(lambda),'  Loss: ', num2str(Loss), '  Delta_Loss: ', num2str(delta_loss)]);
         end   
end
    %% Eigendecomposition of W
     W = U;
     [~, D, V_all] = eig(W); % each column of V represents a spatio-temporal filter
     alpha_all = diag(D); % classifier parameters
     %% Select L pairs of spatio-temporal filters V and classifier parameters alpha
    d = abs(diag(D)); d_max = max(d); 
    w_norm = d/d_max; % normalize eigenvalues of W by the maximum eigenvalue
    index = find(w_norm > threshold); % find index of selected V by pre-defined threshold,.e.g., 0.05
    V = V_all(index); alpha = alpha_all(index);% select V and alpha by index   
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [R_train, Cov_mean_train] = Enhanced_cov_train(X, K, tau)
% ************************************************************************
% Compute Enhanced covariace matrix of train set
% Input :
% K         : Order of FIR filter
% tau       : time delay parameter
% X         : EEG signals from train set, # trials is M

% Output    :
% R         :Enhanced covariace matrix of train set (M * 4 C^2)
% Cov_mean_train : Averaged covariance matrix of train set (2 C^2 * 2 C^2)
%                 (using for whittening and logarithm transform in test stage)
% ************************************************************************

%  Initializaiton 
M = length(X); % M: # trials
[C,T] = size(X{1,1}); % C: # recorded channels, T: # sampled time points
KC = K*C; % KC: # augmented covariance matrices
Cov = cell(1, M);
Sig_Cov = zeros(KC, KC);
for m = 1:M
    X_m = X{1,m};
    X_m_hat = [];
     
    % Generate augument EEG data
    for k = 1 : K
        n_delay = (k-1)*tau;
        if n_delay == 0
            X_order_k = X_m;
        else
            X_order_k(:,1:n_delay) = 0;
            X_order_k(:,n_delay+1:T) = X_m(:,1:T-n_delay);
        end
        X_m_hat = cat(1, X_m_hat, X_order_k);
    end
    
    % Compute covariance matrices with trace normalizaiton
    Cov{1,m} = X_m_hat*X_m_hat';
    Cov{1,m}= Cov{1,m}/trace(Cov{1,m}); 
    Sig_Cov = Sig_Cov + Cov{1,m};
end

% compute averaged covariance matrix of training set
Cov_mean_train = Sig_Cov/M;

% Whitenning, Logarithm transform, and Vectorization
Cov_whiten = zeros(M, KC, KC);
for m = 1:M
    temp_cov = Cov_mean_train^(-1/2)*Cov{1,m}*Cov_mean_train^(-1/2);% whiten
    Cov_whiten(m,:,:)  = (temp_cov + temp_cov')/2; 
    R_m =logm(squeeze(Cov_whiten(m,:,:))); % logarithm transform
    R_m = R_m(:); % column-wise vectorization
    R_train(m,:) =  R_m';
end 
end

