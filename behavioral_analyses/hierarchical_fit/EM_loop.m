function [em_results,mean_initials,cov_initials] = EM_loop(functionname,n_file, criteria,tmax,n_initials,n_startpoints,n_variables,initial_values_subs,n_search_from_best,data)
%% This function loops over EM algorithm. K Iigaya (kiigaya@gmail.com)

mean_initials = -3+6*rand(n_startpoints,n_variables);  % set initial hyper parameters
mean_initials(1,:) = median(initial_values_subs,1);    % for first try set median of subjects as the initial hyper parameter

%initial_values_sessions(initial_values_sessions>3)=3;   % eliminate outliers-- may be useful.
%initial_values_sessions(initial_values_sessions<-3)=-3;

cov_initials = cell(1,n_startpoints); %start with n_startpoints = 1
b2=3; a2=1;     % the range of covariance for generating initial conditions
for l=1:n_startpoints
    A1=(b2-a2).*rand(1,n_variables) + a2*ones(1,n_variables);
    cov_initials{l}=A1; %initial thetasigma
end

% Run EM for n_startpoints times, you'll get n_startpoints sets of results.
parfor l=1:n_startpoints 
    
    disp(['EM start point ' num2str(l)])
    if l>n_search_from_best
        previousvalues=-3+6*rand(size(initial_values_subs));    %%asign random values to initial parameters for indivisuals. -3 to 3. 
    else 
        previousvalues = initial_values_subs;                   % initial input from individual fits for that subject
    end
    
    %initialise things
    thetamean0 = mean_initials(l,:);
    thetasigma0 = cov_initials{l};
    maxtt = 10000;     
    theta0s = NaN(n_variables,maxtt);
    thetacvs = NaN(n_variables,maxtt);
    logposterior = NaN(1,maxtt);
    loglikelihood = NaN(1,maxtt);
    tt=0;
    difference=100;
    previous_logposterior=1000000000;

    while difference>criteria && tt<tmax && min(thetasigma0)>0.025
   
        tt=tt+1;
        initials=-2+4*rand(n_initials,n_variables);
        
        %run one E-step to update the value of the hyperparameters (thetamean and thetasigma)
        [thetamean1,thetasigma1,logposterior(tt),previousvalues,sigmas] = EM_fit_indv(functionname,n_variables,n_file,thetamean0,thetasigma0,previousvalues,initials,n_initials,data);
        thetamean0=thetamean1;
        thetasigma0=thetasigma1;
   
        theta0s(:,tt)=thetamean0;
   
        %plot logposterior of the distribution
        %figure(1)
        %subplot(2,1,1)
        %plot(logposterior)
        %drawnow
  
        %plot updated hyperparameters (means and variances)
        %figure(2)
        %subplot(3,1,1)
        %plot(theta0s')
        %subplot(3,1,2)
        %bar(thetasigma0)
     
        %update posterior and calculate difference to see if model converged (difference<criteria)
        current_logposterior  = logposterior(tt);
        difference            = abs(previous_logposterior-current_logposterior)/previous_logposterior;
        previous_logposterior = current_logposterior;

    end

    em_results(l).logp = current_logposterior;
    em_results(l).ps = previousvalues;
    em_results(l).sigmas = sigmas;
    em_results(l).mean = thetamean0;
    em_results(l).cov = thetasigma0;
    em_results(l).iterations = tt;

end            

end

