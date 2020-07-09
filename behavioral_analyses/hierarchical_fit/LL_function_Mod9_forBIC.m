function f = LL_function_Mod9_forBIC(params,P)

params(1) = exp(params(1));        % decision softmax beta [0 Inf]
params(2) = 1/(1+exp(-params(2))); % learning rate [0 1]

tr_nb = length(P(:,1));

%initialize variables
V = zeros(tr_nb,3); %each row=trial, each column=token
AV = zeros(tr_nb,3); %each row=trial, each column=action

%contingencies of the slot machine (each row=token, each column=action)
SM_struct = cell(2,3);
SM_struct{1,1} = [0.75 0.05 0.2; 0.2 0.75 0.05; 0.05 0.2 0.75]; %probability distribution of EASY slot machine (low BU, horizOrd 1)
SM_struct{1,2} = [0.05 0.2 0.75; 0.75 0.05 0.2; 0.2 0.75 0.05]; %probability distribution of EASY slot machine (low BU, horizOrd 2)
SM_struct{1,3} = [0.2 0.75 0.05; 0.05 0.2 0.75; 0.75 0.05 0.2]; %probability distribution of EASY slot machine (low BU, horizOrd 3)
SM_struct{2,1} = [0.5 0.2 0.3; 0.3 0.5 0.2; 0.2 0.3 0.5]; %probability distribution of HARD slot machine (high BU, horizOrd 1)
SM_struct{2,2} = [0.2 0.3 0.5; 0.5 0.2 0.3; 0.3 0.5 0.2]; %probability distribution of HARD slot machine (high BU, horizOrd 2)
SM_struct{2,3} = [0.3 0.5 0.2; 0.2 0.3 0.5; 0.5 0.2 0.3]; %probability distribution of HARD slot machine (high BU, horizOrd 3)
  
%careful, the column indices may change
tr_type = P(:,4); %obs(1)/play(2)
tr_bu = P(:,7); %low BU(1)/high BU(2)
unav_act = P(:,8);
choice = P(:,12); %subject's choice (1: left, 0: right)
hord = P(:,16); %horizontal order
token_s = P(:,17);

P_left   = NaN(tr_nb,1);

for t=1:tr_nb
    
    UA = unav_act(t); %unavailable action
    
    if P(t,3)>=2
        TS = token_s(t-1); %update based on token shown on previous trial
        if TS == 1 %green token shown
            O = [1 0 0];
        elseif TS == 2 %red token shown
            O = [0 1 0];
        elseif TS == 3 %blue token shown
            O = [0 0 1];
        end
    end

    if tr_type(t)==1 %only update after observe trials
        
        if P(t,3)==1 %initialize prior on first trial of each block
            V(t,:) = [1/3 1/3 1/3];
        else
            V(t,:) = V(t-1,:) + params(2)*(O-V(t-1,:));
        end

        AV(t,:) = [NaN NaN NaN];

    elseif tr_type(t)==2 %play trial
        
        V(t,:) = V(t-1,:) + params(2)*(O-V(t-1,:));
        
        %calculate value of each action by multipyling matrix of slot machine
        %contingencies (token to action mapping) by token values
        AV(t,:) = V(t,:)*SM_struct{tr_bu(t),hord(t)}; 
        AV(t,UA) = 0;
        
        %calculate probability of choosing left-most option
        AV_av = AV(t,:);
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