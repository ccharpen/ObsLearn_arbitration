function [newthetamean,newthetasigma,logposterior,ps,sigmas] = EM_fit_indv(functionname,n_variables,n_file,thetamean,thetasigma,previousvalues,initials,n_initials,data)
%% This function does E and M step. K Iigaya (kiigaya@gmail.com)

diagadd = 0;  %% This is to insure a stability. This in fact introduces a wishart prior for the covariance. You probably don't need this. Set 0. 
%diagadd = 0.005; %if 0 doesn't work (error -> negative variance) then try to set to 0.005

%initialise variables
ps = zeros(n_file,n_variables); %vector to save posterior estimates
hessians = cell(1,n_file);
sigmas = zeros(n_file,n_variables);
options = optimoptions(@fminunc, ...
        'Algorithm', 'quasi-newton', ...
        'Display','off', ...
        'MaxFunEvals', 200, ...
        'MaxIter', 200,...
        'TolFun', 0.01, ...
        'TolX', 0.01);

%% E-step
for fi=1:n_file %looping through subjects (also possible to loop over sessions)
    
    %extract initial value and data for that subject
    initials(1,:) = previousvalues(fi,:);    %% I always serach from pervious estimates, as the first candidate
    P = data(fi).P;
    
    %initialise variables
    ps_sample = ones(n_initials,n_variables);
    likeh = NaN(1,n_initials);
    hessian_candidate = cell(1,n_initials);
    
    ok=0;
    for t=1:n_initials %n_initials should be 2 or 3
        initial=initials(t,:);
        %fit function to data that maximises posterior
        [ps_sample(t,:),current_likeh,exitflag,output,grad,current_hessian] = fminunc(@(x) feval(functionname,x,thetamean,thetasigma,P),initial,options);
        
        if isreal(current_likeh) && isempty(find(diag(current_hessian)<=0.001,1))    %% to make sure that it's not a saddle point.        
            likeh(t) = current_likeh;
            hessian_candidate{t} = current_hessian;
            ok = 1;
        else
            likeh(t) = 10000000000000000000000000000000000000;
            hessian_candidate{t} = current_hessian;
        end
    end
    
    while ok<1
        initial=-3+6*rand(1,n_variables);   %% try other initials
        %fit function to data that maximises posterior    
        [ps_sample(1,:),current_likeh,exitflag,output,grad,current_hessian] = fminunc(@(x) feval(functionname,x,thetamean,thetasigma,P),initial,options);
        
        if isreal(current_likeh) && isempty(find(diag(current_hessian)<=0.001,1))
            likeh(1) = current_likeh;
            hessian_candidate{1} = current_hessian;
            ok = 1;
        end
    end
   
    argmin = find(likeh==min(likeh),1);
    ps(fi,:) = ps_sample(argmin,:);
    full_hessian = hessian_candidate{argmin};
    diagonals = diag(full_hessian)';
    hessians{fi} = diagonals; %% take only diagonals, assuming independence between parameters.
    
    clearvars ps_sample likeh
end

%% M-step

newthetamean = mean(ps);

A=zeros(1,n_variables);
for i=1:n_file %loop over subjects
    sigma = hessians{i}.^(-1);
    sigmas(i,:) = sigma;    
    A = A + ps(i,:).*ps(i,:) + sigma;
end
newthetasigma = A/n_file - newthetamean.*newthetamean + diagadd*ones(1,n_variables);

% calculate joint distribution
logposterior = log(prod(2*pi*newthetasigma)) + sum(sum(0.5*(ones(n_file,1)*newthetamean-ps).*(ones(n_file,1)*newthetasigma).^(-1).*(ones(n_file,1)*newthetamean-ps),2)) + 0.5*sum(sum(ones(n_file,1)*newthetasigma.^(-1).*sigmas)); 
% this is the part that contains theta. you also have log p (A|h) but ignored it in m-step
%sum(log(p(Datai|hi)p(hi|H))) integration into full posterior

end