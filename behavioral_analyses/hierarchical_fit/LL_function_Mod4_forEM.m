function post = LL_function_Mod4_forEM(params,thetamean,thetasigma,P)

params(1) = exp(params(1));        % softmax beta [0 +Inf]
params(2) = 1/(1+exp(-params(2))); % weight of previous PE on current trial learning rate [0 1]
params(3) = 1/(1+exp(-params(3))); % inital value of learning rate [0 1]

tr_nb = length(P(:,1));

%initialize variables
V = zeros(tr_nb,3); %each row=trial, each column=action
alpha = zeros(tr_nb,1); %dynamic learning rate - each row=trial

%careful, the column indices may change
tr_type = P(:,4); %obs(1)/play(2)
unav_act = P(:,8);
part_act = P(:,9); %partner's action (always = correct action)
choice = P(:,12); %subject's choice (1: left, 0: right)

P_left   = NaN(tr_nb,1);

for t=1:tr_nb
    
    UA = unav_act(t); %unavailable action
    PA = part_act(t); %partner's action
    UnchA = [1 2 3];
    UnchA([UA PA]) = []; %unchosen action
        
    if tr_type(t)==1 %observe trial

        %second alternative: no kappa (i.e. assuming kappa = 1)
        if P(t,3)==1 %initialize prior on first trial of each run            
            V(t,UA) = 0; %do not update value of unavailable options
            alpha(t) = params(3); %estimate starting learning rate as a free parameter
            V(t,PA) = alpha(t); %positively update value of action chosen by partner
            V(t,UnchA) = -alpha(t); %negatively update value of unchosen
        else      
            V(t,UA) = V(t-1,UA); %do not update value of unavailable options
            alpha(t) = params(2)*abs(1-V(t-1,PA)) + (1-params(2))*alpha(t-1);
            V(t,PA) = V(t-1,PA) + alpha(t)*(1-V(t-1,PA)); %positively update value of action chosen by partner
            V(t,UnchA) = V(t-1,UnchA) + alpha(t)*(-1-V(t-1,UnchA)); %negatively update value of unchosen  
        end
        
    elseif tr_type(t)==2 %play trial
        
        V(t,:) = V(t-1,:);
        alpha(t) = alpha(t-1);
        
        %calculate probability of choosing left-most option
        AV_av = V(t,:);
        AV_av(UA)=[]; %isolate the probabilities of the 2 available actions
        val_diff = AV_av(1) - AV_av(2); %always left minus right difference
        
        %if choice value is 1, use one part of likelihood contribution.
        if choice(t) == 1
            P_left(t) = (1+exp(-params(1)*val_diff))^-1;                  
        %if choice value is 0, use other part of likelihood contribution    
        elseif choice(t) == 0
            P_left(t) = 1-(1+exp(-params(1)*val_diff))^-1;          
        end   
    end       
end
% adjust 0 probability trials
P_left(P_left < 1e-10) = 1e-10;

f = -nansum(log(P_left)); %negative value of loglikelihood

post = f + sum(0.5.*(params-thetamean).*thetasigma.^(-1).*(params-thetamean)); %posterior
%mathematically equivalent to 0.5*(params - thetamean)^2/thetasigma