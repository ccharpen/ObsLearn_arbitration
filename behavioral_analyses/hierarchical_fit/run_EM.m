function run_EM(model,study)

mod = str2num(char(model));

%% This file runs EM algorithm. You need to specify a function, as well as parameters for this. Made by Kyo Iigaya, kiigaya@gmail.com.
dir_data = pwd;
cd(dir_data)

if study == 1
    load([dir_data 'Models_Parameters_MLE.mat'], 'fitResult_s1')
    fitResult = fitResult_s1;
    load([dir_data 'Data_for_models_S1.mat'],'data')
elseif study == 2
    load([dir_data 'Models_Parameters_MLE.mat'], 'fitResult_s2')
    fitResult = fitResult_s2;
    load([dir_data 'Data_for_models_S2.mat'],'data')
end

%options that remain the same for all models:
n_file   = 30;                          % number of subjects, or sessions, depending on what you do.
%probably need to re-run this for 30 subjects
criteria = 10^(-6);                     % When to stop algorithm (diff < criteria)
tmax     = 1000;                        % When to stop algorithm (number of loops > tmax)
n_initials = 3;                         % number of initial conditions for E step.
n_startpoints = 5;                      % number of starting points for hyper parameters (good to start with 1 to check that things are working
n_search_from_best = 1;                 % within n_startpoints, how many times you want to search from "initial_values" that you set here. 
warning('off','all')
samplesize = 10000;                     % number of samples for BIC calculation

if mod == 1
    %% Model 1: Emulation model, fixed lambda
    disp('Model 1')
    functionname = 'LL_function_Mod1_forEM';  % specify function here
    functionname_bic = 'LL_function_Mod1_forBIC';
    n_variables  = 1;                       % number of variables, this is the number of parameters for that model

    %initial parameter value (here inverse temperature)
    initial_values_subs_r = fitResult.Model1.Params;      % I normally fit indivisually first, and then use those parameters as one of the initial conditions. N of subs x N of parameters. Put randome matrix to start with. 
    initial_values_subs = log(initial_values_subs_r+eps); %transform so that they range from -Inf to +Inf

    [em_results_m1,~,~] = EM_loop(functionname,n_file,criteria,tmax,n_initials,n_startpoints,n_variables,initial_values_subs,n_search_from_best,data);
    
    %transform params
    for l=1:n_startpoints
        em_results_m1(l).pstr = exp(em_results_m1(l).ps);
        em_results_m1(l).meantr = mean(em_results_m1(l).pstr);
        em_results_m1(l).stdtr = std(em_results_m1(l).pstr);
        em_results_m1(l).semtr = std(em_results_m1(l).pstr)/sqrt(n_file);
        thetamean = em_results_m1(l).mean;
        thetasigma = em_results_m1(l).cov;
        em_results_m1(l).bic = bic_score(functionname_bic,thetamean,thetasigma,samplesize,n_variables,n_file,data);
    end  
    if study == 1
        save('em_fit_m1_s1.mat','em_results_m1')
    elseif study == 2
        save('em_fit_m1_s2.mat','em_results_m1')
    end

elseif mod == 2
    %% Model 2: Emulation model, estimated lambda
    disp('Model 2')
    functionname = 'LL_function_Mod2_forEM';  % specify function here
    functionname_bic = 'LL_function_Mod2_forBIC';
    n_variables  = 2;                       % number of variables, this is the number of parameters for that model

    %initial parameter value (here inverse temperature)
    initial_values_subs_r = fitResult.Model2.Params;      % I normally fit indivisually first, and then use those parameters as one of the initial conditions. N of subs x N of parameters. Put randome matrix to start with. 
    initial_values_subs = log(initial_values_subs_r(:,1)+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,2) = -log(1./initial_values_subs_r(:,2)-1+eps); %transform so that they range from -Inf to +Inf

    [em_results_m2,~,~] = EM_loop(functionname,n_file,criteria,tmax,n_initials,n_startpoints,n_variables,initial_values_subs,n_search_from_best,data);
    
    %transform params
    for l=1:n_startpoints
        em_results_m2(l).pstr = exp(em_results_m2(l).ps(:,1)); %inverse temperature
        em_results_m2(l).pstr(:,2) = 1./(1+exp(-em_results_m2(l).ps(:,2))); %lambda
        em_results_m2(l).meantr = mean(em_results_m2(l).pstr);
        em_results_m2(l).stdtr = std(em_results_m2(l).pstr);
        em_results_m2(l).semtr = std(em_results_m2(l).pstr)/sqrt(n_file);
        thetamean = em_results_m2(l).mean;
        thetasigma = em_results_m2(l).cov;
        em_results_m2(l).bic = bic_score(functionname_bic,thetamean,thetasigma,samplesize,n_variables,n_file,data);
    end
    if study == 1
        save('em_fit_m2_s1.mat','em_results_m2')
    elseif study == 2
        save('em_fit_m2_s2.mat','em_results_m2')
    end

elseif mod == 3
    %% Model 3: Imitation model, estimated learning rate
    disp('Model 3')
    functionname = 'LL_function_Mod3_forEM';  % specify function here
    functionname_bic = 'LL_function_Mod3_forBIC';
    n_variables  = 2;                       % number of variables, this is the number of parameters for that model

    %initial parameter value (here inverse temperature)
    initial_values_subs_r = fitResult.Model3.Params;      % I normally fit indivisually first, and then use those parameters as one of the initial conditions. N of subs x N of parameters. Put randome matrix to start with. 
    initial_values_subs = log(initial_values_subs_r(:,1)+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,2) = -log(1./initial_values_subs_r(:,2)-1+eps); %transform so that they range from -Inf to +Inf

    [em_results_m3,~,~] = EM_loop(functionname,n_file,criteria,tmax,n_initials,n_startpoints,n_variables,initial_values_subs,n_search_from_best,data);
    
    %transform params
    for l=1:n_startpoints
        em_results_m3(l).pstr = exp(em_results_m3(l).ps(:,1)); %inverse temperature
        em_results_m3(l).pstr(:,2) = 1./(1+exp(-em_results_m3(l).ps(:,2))); %learning rate
        em_results_m3(l).meantr = mean(em_results_m3(l).pstr);
        em_results_m3(l).stdtr = std(em_results_m3(l).pstr);
        em_results_m3(l).semtr = std(em_results_m3(l).pstr)/sqrt(n_file);
        thetamean = em_results_m3(l).mean;
        thetasigma = em_results_m3(l).cov;
        em_results_m3(l).bic = bic_score(functionname_bic,thetamean,thetasigma,samplesize,n_variables,n_file,data);
    end
    if study == 1
        save('em_fit_m3_s1.mat','em_results_m3')
    elseif study == 2
        save('em_fit_m3_s2.mat','em_results_m3')
    end                     

elseif mod == 4
    %% Model 4: Imitation model, dynamic learning rate
    disp('Model 4')
    functionname = 'LL_function_Mod4_forEM';  % specify function here
    functionname_bic = 'LL_function_Mod4_forBIC';
    n_variables  = 3;                       % number of variables, this is the number of parameters for that model 

    %initial parameter value (here inverse temperature)
    initial_values_subs_r = fitResult.Model4.Params;      % I normally fit indivisually first, and then use those parameters as one of the initial conditions. N of subs x N of parameters. Put randome matrix to start with. 
    initial_values_subs = log(initial_values_subs_r(:,1)+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,2) = -log(1./initial_values_subs_r(:,2)-1+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,3) = -log(1./initial_values_subs_r(:,3)-1+eps); %transform so that they range from -Inf to +Inf

    [em_results_m4,~,~] = EM_loop(functionname,n_file,criteria,tmax,n_initials,n_startpoints,n_variables,initial_values_subs,n_search_from_best,data);
    
    %transform params
    for l=1:n_startpoints
        em_results_m4(l).pstr = exp(em_results_m4(l).ps(:,1)); %inverse temperature
        em_results_m4(l).pstr(:,2) = 1./(1+exp(-em_results_m4(l).ps(:,2))); %weight
        em_results_m4(l).pstr(:,3) = 1./(1+exp(-em_results_m4(l).ps(:,3))); %initial alpha
        em_results_m4(l).meantr = mean(em_results_m4(l).pstr);
        em_results_m4(l).stdtr = std(em_results_m4(l).pstr);
        em_results_m4(l).semtr = std(em_results_m4(l).pstr)/sqrt(n_file);
        thetamean = em_results_m4(l).mean;
        thetasigma = em_results_m4(l).cov;
        em_results_m4(l).bic = bic_score(functionname_bic,thetamean,thetasigma,samplesize,n_variables,n_file,data);
    end
    if study == 1
        save('em_fit_m4_s1.mat','em_results_m4')
    elseif study == 2
        save('em_fit_m4_s2.mat','em_results_m4')
    end   

elseif mod == 5
    %% Model 5: Emulation RL, fixed learning rate
    disp('Model 5')
    functionname = 'LL_function_Mod5_forEM';  % specify function here
    functionname_bic = 'LL_function_Mod5_forBIC';
    n_variables  = 2;                       % number of variables, this is the number of parameters for that model

    %initial parameter value (here inverse temperature)
    initial_values_subs_r = fitResult.Model5.Params;      % I normally fit indivisually first, and then use those parameters as one of the initial conditions. N of subs x N of parameters. Put randome matrix to start with. 
    initial_values_subs = log(initial_values_subs_r(:,1)+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,2) = -log(1./initial_values_subs_r(:,2)-1+eps); %transform so that they range from -Inf to +Inf

    [em_results_m5,~,~] = EM_loop(functionname,n_file,criteria,tmax,n_initials,n_startpoints,n_variables,initial_values_subs,n_search_from_best,data);
    
    %transform params
    for l=1:n_startpoints
        em_results_m5(l).pstr = exp(em_results_m5(l).ps(:,1)); %inverse temperature
        em_results_m5(l).pstr(:,2) = 1./(1+exp(-em_results_m5(l).ps(:,2))); %learning rate
        em_results_m5(l).meantr = mean(em_results_m5(l).pstr);
        em_results_m5(l).stdtr = std(em_results_m5(l).pstr);
        em_results_m5(l).semtr = std(em_results_m5(l).pstr)/sqrt(n_file);
        thetamean = em_results_m5(l).mean;
        thetasigma = em_results_m5(l).cov;
        em_results_m5(l).bic = bic_score(functionname_bic,thetamean,thetasigma,samplesize,n_variables,n_file,data);
    end
    if study == 1
        save('em_fit_m5_s1.mat','em_results_m5')
    elseif study == 2
        save('em_fit_m5_s2.mat','em_results_m5')
    end   

elseif mod == 6
    %% Model 6: Emulation RL, dynamic learning rate
    disp('Model 6')
    functionname = 'LL_function_Mod6_forEM';  % specify function here
    functionname_bic = 'LL_function_Mod6_forBIC';
    n_variables  = 3;                       % number of variables, this is the number of parameters for that model 

    %initial parameter value (here inverse temperature)
    initial_values_subs_r = fitResult.Model6.Params;      % I normally fit indivisually first, and then use those parameters as one of the initial conditions. N of subs x N of parameters. Put randome matrix to start with. 
    initial_values_subs = log(initial_values_subs_r(:,1)+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,2) = -log(1./initial_values_subs_r(:,2)-1+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,3) = -log(1./initial_values_subs_r(:,3)-1+eps); %transform so that they range from -Inf to +Inf

    [em_results_m6,~,~] = EM_loop(functionname,n_file,criteria,tmax,n_initials,n_startpoints,n_variables,initial_values_subs,n_search_from_best,data);
    
    %transform params
    for l=1:n_startpoints
        em_results_m6(l).pstr = exp(em_results_m6(l).ps(:,1)); %inverse temperature
        em_results_m6(l).pstr(:,2) = 1./(1+exp(-em_results_m6(l).ps(:,2))); %weight
        em_results_m6(l).pstr(:,3) = 1./(1+exp(-em_results_m6(l).ps(:,3))); %initial alpha
        em_results_m6(l).meantr = mean(em_results_m6(l).pstr);
        em_results_m6(l).stdtr = std(em_results_m6(l).pstr);
        em_results_m6(l).semtr = std(em_results_m6(l).pstr)/sqrt(n_file);
        thetamean = em_results_m6(l).mean;
        thetasigma = em_results_m6(l).cov;
        em_results_m6(l).bic = bic_score(functionname_bic,thetamean,thetasigma,samplesize,n_variables,n_file,data);
    end
    if study == 1
        save('em_fit_m6_s1.mat','em_results_m6')
    elseif study == 2
        save('em_fit_m6_s2.mat','em_results_m6')
    end   

elseif mod == 7
    %% Model 7: Arbitration model, no lambda
    disp('Model 7')
    functionname = 'LL_function_Mod7_forEM';  % specify function here
    functionname_bic = 'LL_function_Mod7_forBIC';
    n_variables  = 4;                       % number of variables, this is the number of parameters for that model

    %initial parameter value (here inverse temperature)
    initial_values_subs_r = fitResult.Model7.Params;     
    % I normally fit indivisually first, and then use those parameters as one of the initial conditions. N of subs x N of parameters. Put randome matrix to start with. 
    initial_values_subs = log(initial_values_subs_r(:,1)+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,2) = log(initial_values_subs_r(:,2)+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,3) = initial_values_subs_r(:,3); %no transformation needed for bias parameter
    initial_values_subs(:,4) = -log(1./initial_values_subs_r(:,4)-1+eps); %transform so that they range from -Inf to +Inf

    [em_results_m7,~,~] = EM_loop(functionname,n_file,criteria,tmax,n_initials,n_startpoints,n_variables,initial_values_subs,n_search_from_best,data);

    %transform params
    for l=1:n_startpoints
        em_results_m7(l).pstr = exp(em_results_m7(l).ps(:,1));      %inverse temperature emulation
        em_results_m7(l).pstr(:,2) = exp(em_results_m7(l).ps(:,2)); %inverse temperature imitation
        em_results_m7(l).pstr(:,3) = em_results_m7(l).ps(:,3);      %bias (no transformation)
        em_results_m7(l).pstr(:,4) = 1./(1+exp(-em_results_m7(l).ps(:,4))); %alpha
        em_results_m7(l).meantr = mean(em_results_m7(l).pstr);
        em_results_m7(l).stdtr = std(em_results_m7(l).pstr);
        em_results_m7(l).semtr = std(em_results_m7(l).pstr)/sqrt(n_file);
        thetamean = em_results_m7(l).mean;
        thetasigma = em_results_m7(l).cov;
        em_results_m7(l).bic = bic_score(functionname_bic,thetamean,thetasigma,samplesize,n_variables,n_file,data);
    end 
    if study == 1
        save('em_fit_m7_s1.mat','em_results_m7')
    elseif study == 2
        save('em_fit_m7_s2.mat','em_results_m7')
    end   

elseif mod == 8
    %% Model 8: Arbitration model, lambda
    disp('Model 8')
    functionname = 'LL_function_Mod8_forEM';  % specify function here
    functionname_bic = 'LL_function_Mod8_forBIC';
    n_variables  = 5;                       % number of variables, this is the number of parameters for that model

    %initial parameter value (here inverse temperature)
    initial_values_subs_r = fitResult.Model8.Params;     
    % I normally fit first, and then use those parameters as one of the initial conditions. N of subs x N of parameters. Put randome matrix to start with. 
    initial_values_subs      = log(initial_values_subs_r(:,1)+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,2) = log(initial_values_subs_r(:,2)+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,3) = initial_values_subs_r(:,3); %no transformation needed for bias parameter
    initial_values_subs(:,4) = -log(1./initial_values_subs_r(:,4)-1+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,5) = -log(1./initial_values_subs_r(:,5)-1+eps); %transform so that they range from -Inf to +Inf

    [em_results_m8,~,~] = EM_loop(functionname,n_file,criteria,tmax,n_initials,n_startpoints,n_variables,initial_values_subs,n_search_from_best,data);
    
    %transform params
    for l=1:n_startpoints
        em_results_m8(l).pstr = exp(em_results_m8(l).ps(:,1));      %inverse temperature emulation
        em_results_m8(l).pstr(:,2) = exp(em_results_m8(l).ps(:,2)); %inverse temperature imitation
        em_results_m8(l).pstr(:,3) = em_results_m8(l).ps(:,3);      %bias (no transformation)
        em_results_m8(l).pstr(:,4) = 1./(1+exp(-em_results_m8(l).ps(:,4))); %alpha
        em_results_m8(l).pstr(:,5) = 1./(1+exp(-em_results_m8(l).ps(:,5))); %lambda
        em_results_m8(l).meantr = mean(em_results_m8(l).pstr);
        em_results_m8(l).stdtr = std(em_results_m8(l).pstr);
        em_results_m8(l).semtr = std(em_results_m8(l).pstr)/sqrt(n_file);
        thetamean = em_results_m8(l).mean;
        thetasigma = em_results_m8(l).cov;
        em_results_m8(l).bic = bic_score(functionname_bic,thetamean,thetasigma,samplesize,n_variables,n_file,data);
    end 
    if study == 1
        save('em_fit_m8_s1.mat','em_results_m8')
    elseif study == 2
        save('em_fit_m8_s2.mat','em_results_m8')
    end 

elseif mod == 9
    %% Model 9: RL token shown
    disp('Model 9')
    functionname = 'LL_function_Mod9_forEM';  % specify function here
    functionname_bic = 'LL_function_Mod9_forBIC';
    n_variables  = 2;                       % number of variables, this is the number of parameters for that model

    %initial parameter value (here inverse temperature)
    initial_values_subs_r = fitResult.Model9.Params;      % I normally fit indivisually first, and then use those parameters as one of the initial conditions. N of subs x N of parameters. Put randome matrix to start with. 
    initial_values_subs = log(initial_values_subs_r(:,1)+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,2) = -log(1./initial_values_subs_r(:,2)-1+eps); %transform so that they range from -Inf to +Inf

    [em_results_m9,~,~] = EM_loop(functionname,n_file,criteria,tmax,n_initials,n_startpoints,n_variables,initial_values_subs,n_search_from_best,data);
    
    %transform params
    for l=1:n_startpoints
        em_results_m9(l).pstr = exp(em_results_m9(l).ps(:,1)); %inverse temperature
        em_results_m9(l).pstr(:,2) = 1./(1+exp(-em_results_m9(l).ps(:,2))); %learning rate
        em_results_m9(l).meantr = mean(em_results_m9(l).pstr);
        em_results_m9(l).stdtr = std(em_results_m9(l).pstr);
        em_results_m9(l).semtr = std(em_results_m9(l).pstr)/sqrt(n_file);
        thetamean = em_results_m9(l).mean;
        thetasigma = em_results_m9(l).cov;
        em_results_m9(l).bic = bic_score(functionname_bic,thetamean,thetasigma,samplesize,n_variables,n_file,data);
    end
    if study == 1
        save('em_fit_m9_s1.mat','em_results_m9')
    elseif study == 2
        save('em_fit_m9_s2.mat','em_results_m9')
    end 

elseif mod==10
    %% new Model 10: Arbitration model with 1-step imitation, no lambda
    disp('Model 10')
    functionname = 'LL_function_newMod10_forEM';  % specify function here
    functionname_bic = 'LL_function_newMod10_forBIC';
    n_variables  = 3;                       % number of variables, this is the number of parameters for that model

    %initial parameter value (here inverse temperature)
    initial_values_subs_r = fitResult.Model10.Params;     
    % I normally fit indivisually first, and then use those parameters as one of the initial conditions. N of subs x N of parameters. Put randome matrix to start with. 
    initial_values_subs = log(initial_values_subs_r(:,1)+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,2) = log(initial_values_subs_r(:,2)+eps); %transform so that they range from -Inf to +Inf
    initial_values_subs(:,3) = initial_values_subs_r(:,3); %no transformation needed for bias parameter

    [em_results_m10,~,~] = EM_loop(functionname,n_file,criteria,tmax,n_initials,n_startpoints,n_variables,initial_values_subs,n_search_from_best,data);

    %transform params
    for l=1:n_startpoints
        em_results_m10(l).pstr = exp(em_results_m10(l).ps(:,1));      %inverse temperature emulation
        em_results_m10(l).pstr(:,2) = exp(em_results_m10(l).ps(:,2)); %inverse temperature imitation
        em_results_m10(l).pstr(:,3) = em_results_m10(l).ps(:,3);      %bias (no transformation)
        em_results_m10(l).meantr = mean(em_results_m10(l).pstr);
        em_results_m10(l).stdtr = std(em_results_m10(l).pstr);
        em_results_m10(l).semtr = std(em_results_m10(l).pstr)/sqrt(n_file);
        thetamean = em_results_m10(l).mean;
        thetasigma = em_results_m10(l).cov;
        disp(['Mod10 BIC - start point ' num2str(l)])
        em_results_m10(l).bic = bic_score(functionname_bic,thetamean,thetasigma,samplesize,n_variables,n_file,data);
    end 
    if study == 1
        save('em_fit_newm10_s1.mat','em_results_m10')
    elseif study == 2
        save('em_fit_newm10_s2.mat','em_results_m10')
    end 
end