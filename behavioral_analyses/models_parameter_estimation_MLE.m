%This code is the wrapper that fits all 10 models to the data 
%and estimate parameters for each subject using maximum likelihood
%------------ Caroline Charpentier ----------------

clear all
close all
addpath('model_functions')

data_dir = pwd; %replace with relevant directory
load('Data_for_models_S1.mat','data')
data_s1 = data;
n_s1 = length(data_s1);
load('Data_for_models_S2.mat','data')
data_s2 = data;
n_s2 = length(data_s2);
ntot = n_s1 + n_s2;
data_all = [data_s1;data_s2];

% init the randomization screen
RandStream.setGlobalStream(RandStream('mt19937ar','Seed','shuffle'));

%Setup solver opt for fminunc
npar = 5; %max number of parameters
opts = optimoptions(@fminunc, ...
        'Algorithm', 'quasi-newton', ...
        'Display','off', ...
        'MaxFunEvals', 50 * npar, ...
        'MaxIter', 50 * npar,...
        'TolFun', 0.01, ...
        'TolX', 0.01);

fitResult_s1 = struct();
fitResult_s2 = struct();

%Emulation Optimal Model
Model1_Params         = zeros(ntot,1);
Model1_Loglikelihood  = zeros(ntot,1);
Model1_PseudoR2       = zeros(ntot,1);

%Emulation with Lambda
Model2_Params         = zeros(ntot,2);
Model2_Loglikelihood  = zeros(ntot,1);
Model2_PseudoR2       = zeros(ntot,1);
Model2_LL_percond     = zeros(ntot,4);

%Imitation with fixed learning rate
Model3_Params         = zeros(ntot,2);
Model3_Loglikelihood  = zeros(ntot,1);
Model3_PseudoR2       = zeros(ntot,1);
Model3_LL_percond     = zeros(ntot,4);

%Imitation with dynamic learning rate
Model4_Params         = zeros(ntot,3);
Model4_Loglikelihood  = zeros(ntot,1);
Model4_PseudoR2       = zeros(ntot,1);

%Emulation RL - fixed learning rate, one beta
Model5_Params         = zeros(ntot,2);
Model5_Loglikelihood  = zeros(ntot,1);
Model5_PseudoR2       = zeros(ntot,1);

%Emulation RL - dynamic learning rate
Model6_Params         = zeros(ntot,3);
Model6_Loglikelihood  = zeros(ntot,1);
Model6_PseudoR2       = zeros(ntot,1);

%Arbitration model without lambda
Model7_Params         = zeros(ntot,4);
Model7_Loglikelihood  = zeros(ntot,1);
Model7_PseudoR2       = zeros(ntot,1);
Model7_Weight_percond = zeros(ntot,4);

%Arbitration model with lambda
Model8_Params         = zeros(ntot,5);
Model8_Loglikelihood  = zeros(ntot,1);
Model8_PseudoR2       = zeros(ntot,1);
Mode8_Weight_percond = zeros(ntot,4);

%Token Shown RL
Model9_Params         = zeros(ntot,2);
Model9_Loglikelihood  = zeros(ntot,1);
Model9_PseudoR2       = zeros(ntot,1);

%Revised arbitration model (with 1-step imitation strategy instead of
%imitation RL)
Model10_Params         = zeros(ntot,3);
Model10_Loglikelihood  = zeros(ntot,1);
Model10_PseudoR2       = zeros(ntot,1);
Model10_Weight_percond = zeros(ntot,4);

numcores = feature('numcores'); %detects number of core (parallel processing toolbox)
parfor (i = 1:ntot,numcores) %if parallel processing toolbox not available, this can be replaced by 'for i=1:ntot'
    %Load behavioral data
    P = data_all(i).P;
    %summary of P matrix
    %col 1: run order
    %col 2: run ID
    %col 3: trial number
    %col 4: Observe (1), Play (2)
    %col 5: goal token
    %col 6: stable (1), volatile (2)
    %col 7: low uncertainty (1), high uncertainty (2)
    %col 8: unavailable action
    %col 9: correct action (also partner's action)
    %col 10: best action
    %col 11: subject choice (coded as 1, 2 or 3 for left, down, right)
    %col 12: subject choice (coded as 1 for left and 0 for right)
    %col 13: subject is correct (1) or not (0)
    %col 14: choice RT
    %col 15: vertOrd (1:G/R/B, 2: R/B/G, 3: B/G/R)
    %col 16: horizOrd (1: G-R-B, 2: R-B-G, 3: B-G-R)
    %col 17: token shown
    
    tr_nb = sum(~isnan(P(:,12))); %number of non-missed play trials, used to calculate BIC and pseudo-R2
    i_play = P(:,4)==2;
    ista = P(:,6)==1; %stable trials
    ilu = P(:,7)==1; %low uncertainty trials
    
    %% Fit Model 1
    %EM model with optimal lambda=0.9999
    disp(['Sub ' num2str(i) '- Fitting Mod 1'])
    npar = 1;
    params_rand=[]; 
    for i_rand=1:20*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_model1_EM, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<20*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 30/(1+exp(-best_params(1)));   % decision softmax beta [0 30]
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);
    
    Model1_Params(i,:)        = best_params(1:npar);
    Model1_Loglikelihood(i,1) = best_params(npar+1);
    Model1_PseudoR2(i,1)      = pseudoR2;
    
    %% Fit Model 2
    %EM model with estimated lambda
    disp(['Sub ' num2str(i) '- Fitting Mod 2'])
    npar = 2;
    params_rand=[]; 
    for i_rand=1:20*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_model2_EM, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<20*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 30/(1+exp(-best_params(1)));   % decision softmax beta [0 30]
    best_params(2) = 1/(1+exp(-best_params(2)));   % lambda [0 1]
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);
    
    Model2_Params(i,:)        = best_params(1:npar);
    Model2_Loglikelihood(i,1) = best_params(npar+1);
    Model2_PseudoR2(i,1)      = pseudoR2;
    
    %calculate mean loglikelihood per trial for each of the 4 conditions
    P_ll = generate_choice_model2_EM(best_params(1:npar),P);
    ll = P_ll(:,17);
    Model2_LL_percond(i,:) = [nanmean(ll(ista & ilu)) nanmean(ll(~ista & ilu)) ...
        nanmean(ll(ista & ~ilu)) nanmean(ll(~ista & ~ilu))];

    %% Fit Model 3
    %IM model with fixed learning rate
    disp(['Sub ' num2str(i) '- Fitting Mod 3'])
    npar = 2;
    params_rand=[]; 
    for i_rand=1:20*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_model3_IM, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<20*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 30/(1+exp(-best_params(1)));  % decision softmax beta [0 30]
    best_params(2) = 1/(1+exp(-best_params(2)));   % alpha [0 1]
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);
    
    Model3_Params(i,:)        = best_params(1:npar);
    Model3_Loglikelihood(i,1) = best_params(npar+1);
    Model3_PseudoR2(i,1)      = pseudoR2;
    
    %calculate mean loglikelihood per trial for each of the 4 conditions
    P_ll = generate_choice_model3_IM(best_params(1:npar),P);
    ll = P_ll(:,11);
    Model3_LL_percond(i,:) = [nanmean(ll(ista & ilu)) nanmean(ll(~ista & ilu)) ...
        nanmean(ll(ista & ~ilu)) nanmean(ll(~ista & ~ilu))];
    
    %% Fit Model 4
    %IM model with dynamic learning rate
    disp(['Sub ' num2str(i) '- Fitting Mod 4'])
    npar = 3;
    params_rand=[]; 
    for i_rand=1:20*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_model4_IM, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<20*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 30/(1+exp(-best_params(1)));  % decision softmax beta [0 30]
    best_params(2) = 1/(1+exp(-best_params(2)));   % weight prev PE [0 1]
    best_params(3) = 1/(1+exp(-best_params(3)));   % initial alpha [0 1]
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);
    
    Model4_Params(i,:)        = best_params(1:npar);
    Model4_Loglikelihood(i,1) = best_params(npar+1);
    Model4_PseudoR2(i,1)      = pseudoR2;
    
    %% Fit Model 5
    %EM RL model, fixed learning rate, one beta
    disp(['Sub ' num2str(i) '- Fitting Mod 5'])
    npar = 2;
    params_rand=[]; 
    for i_rand=1:20*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_model5_EM_RL, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<20*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 30/(1+exp(-best_params(1)));  % decision softmax beta [0 30]
    best_params(2) = 1/(1+exp(-best_params(2)));   % alpha [0 1]
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);
    
    Model5_Params(i,:)        = best_params(1:npar);
    Model5_Loglikelihood(i,1) = best_params(npar+1);
    Model5_PseudoR2(i,1)      = pseudoR2;
        
    %% Fit Model 6
    %EM RL model, dynamic learning rate
    disp(['Sub ' num2str(i) '- Fitting Mod 6'])
    npar = 3;
    params_rand=[]; 
    for i_rand=1:20*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_model6_EM_RL, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<20*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 30/(1+exp(-best_params(1)));  % decision softmax beta [0 30]
    best_params(2) = 1/(1+exp(-best_params(2)));   % weight of previous PE on current trial learning rate [0 1]
    best_params(3) = 1/(1+exp(-best_params(3)));   % inital value of learning rate [0 1]
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);
    
    Model6_Params(i,:)        = best_params(1:npar);
    Model6_Loglikelihood(i,1) = best_params(npar+1);
    Model6_PseudoR2(i,1)     = pseudoR2;
    
    %% Fit Model 7
    %Arbitration Model, no lambda
    disp(['Sub ' num2str(i) '- Fitting Mod 7'])
    npar = 4;
    params_rand=[]; 
    for i_rand=1:20*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_model7_arbitration, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<20*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 30/(1+exp(-best_params(1)));  % emulation decision softmax beta [0 30]
    best_params(2) = 30/(1+exp(-best_params(2)));  % imitation decision softmax beta [0 30]
    best_params(3) = 10/(1+exp(-best_params(3)))-5; % bias [-5 5]
    best_params(4) = 1/(1+exp(-best_params(4)));  % learning rate for imitation values [0 1]
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);

    Model7_Params(i,:)          = best_params(1:npar);
    Model7_Loglikelihood(i,1)   = best_params(npar+1);
    Model7_PseudoR2(i,1)        = pseudoR2;
    
    %extract weight for each of the 4 conditions
    P_ll = generate_choice_model7_arbitration(best_params(1:npar),P);
    w = P_ll(:,18);
    Model7_Weight_percond(i,:) = [nanmean(w(ista & ilu)) nanmean(w(~ista & ilu)) ...
        nanmean(w(ista & ~ilu)) nanmean(w(~ista & ~ilu))];
    
    %% Fit Model 8
    %Arbitration Model, lambda
    disp(['Sub ' num2str(i) '- Fitting Mod 8'])
    npar = 5;
    params_rand=[]; 
    for i_rand=1:20*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_model8_arbitration, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<20*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 30/(1+exp(-best_params(1)));  % emulation decision softmax beta [0 30]
    best_params(2) = 30/(1+exp(-best_params(2)));  % imitation decision softmax beta [0 30]
    best_params(3) = 10/(1+exp(-best_params(3)))-5; % bias [-5 5]
    best_params(4) = 1/(1+exp(-best_params(4)));  % learning rate for imitation values [0 1]
    best_params(5) = 1/(1+exp(-best_params(5)));  % lambda [0 1]
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);

    Model8_Params(i,:)          = best_params(1:npar);
    Model8_Loglikelihood(i,1)   = best_params(npar+1);
    Model8_PseudoR2(i,1)        = pseudoR2;
    
    %extract weight for each of the 4 conditions
    P_ll = generate_choice_model8_arbitration(best_params(1:npar),P);
    w = P_ll(:,18);
    Model8_Weight_percond(i,:) = [nanmean(w(ista & ilu)) nanmean(w(~ista & ilu)) ...
        nanmean(w(ista & ~ilu)) nanmean(w(~ista & ~ilu))];
    
    %% Fit Model 9
    %Token shown RL 
    disp(['Sub ' num2str(i) '- Fitting Mod 9'])
    npar = 2;
    params_rand=[]; 
    for i_rand=1:20*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_model9_tokenRL, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<20*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 30/(1+exp(-best_params(1)));  % decision softmax beta [0 30]
    best_params(2) = 1/(1+exp(-best_params(2)));   % learning rate [0 1]
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);
    
    Model9_Params(i,:)        = best_params(1:npar);
    Model9_Loglikelihood(i,1) = best_params(npar+1);
    Model9_PseudoR2(i,1)      = pseudoR2;
    
    %% Fit Model 10
    %Arbitration Model, lambda
    disp(['Sub ' num2str(i) '- Fitting Mod 10'])
    npar = 3;
    params_rand=[]; 
    for i_rand=1:20*npar
        start = randn(1,npar); %also possible: sample from a gamma distribution (rather than normal)
        [paramtracker, lltracker, ex] = fminunc(@LL_function_model10_arbitration, start, opts, P);
        params_rand=[params_rand; paramtracker lltracker ex];
    end
    bad_fit = params_rand(:,npar+2)<=0;
    if sum(bad_fit)<20*npar
        params_rand(bad_fit,:)=[];
    end
    [~,ids] = sort(params_rand(:,npar+1));
    best_params = params_rand(ids(1),:);
    best_params(1) = 30/(1+exp(-best_params(1)));  % emulation decision softmax beta [0 30]
    best_params(2) = 30/(1+exp(-best_params(2)));  % imitation decision softmax beta [0 30]
    best_params(3) = 10/(1+exp(-best_params(3)))-5; % bias [-5 5]
    pseudoR2 = 1 + best_params(npar+1)/(log(0.5)*tr_nb);

    Model10_Params(i,:)          = best_params(1:npar);
    Model10_Loglikelihood(i,1)   = best_params(npar+1);
    Model10_PseudoR2(i,1)        = pseudoR2;
    
    %extract weight for each of the 4 conditions
    P_ll = generate_choice_model10_arbitration(best_params(1:npar),P);
    w = P_ll(:,14);
    Model10_Weight_percond(i,:) = [nanmean(w(ista & ilu)) nanmean(w(~ista & ilu)) ...
        nanmean(w(ista & ~ilu)) nanmean(w(~ista & ~ilu))];
end

fitResult_s1.Model1.Params          = Model1_Params(1:n_s1,:);
fitResult_s1.Model1.Loglikelihood   = Model1_Loglikelihood(1:n_s1);
fitResult_s1.Model1.PseudoR2        = Model1_PseudoR2(1:n_s1);
fitResult_s1.Model2.Params          = Model2_Params(1:n_s1,:);
fitResult_s1.Model2.Loglikelihood   = Model2_Loglikelihood(1:n_s1);
fitResult_s1.Model2.PseudoR2        = Model2_PseudoR2(1:n_s1);
fitResult_s1.Model2.LL_percond      = Model2_LL_percond(1:n_s1,:);
fitResult_s1.Model3.Params          = Model3_Params(1:n_s1,:);
fitResult_s1.Model3.Loglikelihood   = Model3_Loglikelihood(1:n_s1);
fitResult_s1.Model3.PseudoR2        = Model3_PseudoR2(1:n_s1);
fitResult_s1.Model3.LL_percond      = Model3_LL_percond(1:n_s1,:);
fitResult_s1.Model4.Params          = Model4_Params(1:n_s1,:);
fitResult_s1.Model4.Loglikelihood   = Model4_Loglikelihood(1:n_s1);
fitResult_s1.Model4.PseudoR2        = Model4_PseudoR2(1:n_s1);
fitResult_s1.Model5.Params          = Model5_Params(1:n_s1,:);
fitResult_s1.Model5.Loglikelihood   = Model5_Loglikelihood(1:n_s1);
fitResult_s1.Model5.PseudoR2        = Model5_PseudoR2(1:n_s1);
fitResult_s1.Model6.Params          = Model6_Params(1:n_s1,:);
fitResult_s1.Model6.Loglikelihood   = Model6_Loglikelihood(1:n_s1);
fitResult_s1.Model6.PseudoR2        = Model6_PseudoR2(1:n_s1);
fitResult_s1.Model7.Params          = Model7_Params(1:n_s1,:);
fitResult_s1.Model7.Loglikelihood   = Model7_Loglikelihood(1:n_s1);
fitResult_s1.Model7.PseudoR2        = Model7_PseudoR2(1:n_s1);
fitResult_s1.Model8.Params          = Model8_Params(1:n_s1,:);
fitResult_s1.Model8.Loglikelihood   = Model8_Loglikelihood(1:n_s1);
fitResult_s1.Model8.PseudoR2        = Model8_PseudoR2(1:n_s1);
fitResult_s1.Model9.Params          = Model9_Params(1:n_s1,:);
fitResult_s1.Model9.Loglikelihood   = Model9_Loglikelihood(1:n_s1);
fitResult_s1.Model9.PseudoR2        = Model9_PseudoR2(1:n_s1);
fitResult_s1.Model10.Params         = Model10_Params(1:n_s1,:);
fitResult_s1.Model10.Loglikelihood  = Model10_Loglikelihood(1:n_s1);
fitResult_s1.Model10.PseudoR2       = Model10_PseudoR2(1:n_s1);

fitResult_s2.Model1.Params          = Model1_Params(n_s1+1:ntot,:);
fitResult_s2.Model1.Loglikelihood   = Model1_Loglikelihood(n_s1+1:ntot);
fitResult_s2.Model1.PseudoR2        = Model1_PseudoR2(n_s1+1:ntot);
fitResult_s2.Model2.Params          = Model2_Params(n_s1+1:ntot,:);
fitResult_s2.Model2.Loglikelihood   = Model2_Loglikelihood(n_s1+1:ntot);
fitResult_s2.Model2.PseudoR2        = Model2_PseudoR2(n_s1+1:ntot);
fitResult_s2.Model2.LL_percond      = Model2_LL_percond(n_s1+1:ntot,:);
fitResult_s2.Model3.Params          = Model3_Params(n_s1+1:ntot,:);
fitResult_s2.Model3.Loglikelihood   = Model3_Loglikelihood(n_s1+1:ntot);
fitResult_s2.Model3.PseudoR2        = Model3_PseudoR2(n_s1+1:ntot);
fitResult_s2.Model3.LL_percond      = Model3_LL_percond(n_s1+1:ntot,:);
fitResult_s2.Model4.Params          = Model4_Params(n_s1+1:ntot,:);
fitResult_s2.Model4.Loglikelihood   = Model4_Loglikelihood(n_s1+1:ntot);
fitResult_s2.Model4.PseudoR2        = Model4_PseudoR2(n_s1+1:ntot);
fitResult_s2.Model5.Params          = Model5_Params(n_s1+1:ntot,:);
fitResult_s2.Model5.Loglikelihood   = Model5_Loglikelihood(n_s1+1:ntot);
fitResult_s2.Model5.PseudoR2        = Model5_PseudoR2(n_s1+1:ntot);
fitResult_s2.Model6.Params          = Model6_Params(n_s1+1:ntot,:);
fitResult_s2.Model6.Loglikelihood   = Model6_Loglikelihood(n_s1+1:ntot);
fitResult_s2.Model6.PseudoR2        = Model6_PseudoR2(n_s1+1:ntot);
fitResult_s2.Model7.Params          = Model7_Params(n_s1+1:ntot,:);
fitResult_s2.Model7.Loglikelihood   = Model7_Loglikelihood(n_s1+1:ntot);
fitResult_s2.Model7.PseudoR2        = Model7_PseudoR2(n_s1+1:ntot);
fitResult_s2.Model8.Params          = Model8_Params(n_s1+1:ntot,:);
fitResult_s2.Model8.Loglikelihood   = Model8_Loglikelihood(n_s1+1:ntot);
fitResult_s2.Model8.PseudoR2        = Model8_PseudoR2(n_s1+1:ntot);
fitResult_s2.Model9.Params          = Model9_Params(n_s1+1:ntot,:);
fitResult_s2.Model9.Loglikelihood   = Model9_Loglikelihood(n_s1+1:ntot);
fitResult_s2.Model9.PseudoR2        = Model9_PseudoR2(n_s1+1:ntot);
fitResult_s2.Model10.Params         = Model10_Params(n_s1+1:ntot,:);
fitResult_s2.Model10.Loglikelihood  = Model10_Loglikelihood(n_s1+1:ntot);
fitResult_s2.Model10.PseudoR2       = Model10_PseudoR2(n_s1+1:ntot);
fitResult_s2.Model10.Weight_percond = Model10_Weight_percond(n_s1+1:ntot,:);

save('Models_Parameters_MLE.mat','fitResult_s1','fitResult_s2')
