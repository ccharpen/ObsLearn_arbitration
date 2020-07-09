function f = LL_function_model2_EM(params,P)

%this function calculates loglikelihood for a model inferring the value of each
%token using approximate Bayesian update
%evidence given by partner action [P(partner action|goal token & available
%actions)] is binary (either 1 or 0)
%computation of action values from token values are based on the
%probability of getting each token given each slot machine/action chosen
%------------ Caroline Charpentier ----------------

%transform parameters to make sure they are constrained between values that
%make sense
params(1) = 30/(1+exp(-params(1)));   % softmax beta [0 30]
params(2) = 1/(1+exp(-params(2))); % lambda [0 1]

tr_nb = length(P(:,1));

%initialize variables
prior_V = zeros(tr_nb,3); %each row=trial, each column=token
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
tr_unc = P(:,7); %low uncertainty(1)/high uncertainty(2)
unav_act = P(:,8); %unavailable action
part_act = P(:,9); %partner's action (always = correct action)
choice = P(:,12); %subject's choice (1: left, 0: right)
hord = P(:,16); %horizontal order

%task structure matrices
P_PA_Tok{1,1} = [0 0 1;0 1 0;0 0 1]; %ua 1, horizOrd 1
P_PA_Tok{1,2} = [0 0 1;0 0 1;0 1 0]; %ua 1, horizOrd 2
P_PA_Tok{1,3} = [0 1 0;0 0 1;0 0 1]; %ua 1, horizOrd 3
P_PA_Tok{2,1} = [1 0 0;1 0 0;0 0 1]; %ua 2, horizOrd 1
P_PA_Tok{2,2} = [0 0 1;1 0 0;1 0 0]; %ua 2, horizOrd 2
P_PA_Tok{2,3} = [1 0 0;0 0 1;1 0 0]; %ua 2, horizOrd 3
P_PA_Tok{3,1} = [1 0 0;0 1 0;0 1 0]; %ua 3, horizOrd 1
P_PA_Tok{3,2} = [0 1 0;1 0 0;0 1 0]; %ua 3, horizOrd 2
P_PA_Tok{3,3} = [0 1 0;0 1 0;1 0 0]; %ua 3, horizOrd 3
%here we detail the structure of probability of each partner's action given
%set of available actions and given goal token
%each row of the cell structure {1}, {2}, or {3} represents the unavailable action.
%each column of the cell structure {1}, {2}, or {3} represents the horizontal order.
%within each cell, each column represents the action performed by the partner
%and each row represents the conditional goal token (1:green, 2:red, 3:blue)

P_left   = NaN(tr_nb,1);

for t=1:tr_nb
    
    UA = unav_act(t); %unavailable action #1
    PA = part_act(t); %partner's action
    HO = hord(t); %horizontal order

    if tr_type(t)==1 %only update after observe trials
        %compute prior at trial t for each token: prior combines previous
        %trial's posterior such that there is a probability lambda of it
        %being the same (no switch) and a probability 1-lambda that one of the
        %other two tokens is now valuable
        if P(t,3)==1 %initialize prior on first trial of each block
            prior_V(t,:) = [1/3 1/3 1/3];
        else %no switch possible on first trial
            prior_V(t,1) = params(2)*V(t-1,1) + (1-params(2))*(1/2)*(V(t-1,2) + V(t-1,3));
            prior_V(t,2) = params(2)*V(t-1,2) + (1-params(2))*(1/2)*(V(t-1,1) + V(t-1,3));
            prior_V(t,3) = params(2)*V(t-1,3) + (1-params(2))*(1/2)*(V(t-1,1) + V(t-1,2));
        end

        %Calculate likelihood of partner's action given slot machine and
        %valuable token
        P_PA_V = P_PA_Tok{UA,HO}(:,PA)'; %each row=token

        %update probabilities of each token being valuable
        V(t,:) = prior_V(t,:).*P_PA_V;

        %scale probas such that Vg(t) + Vr(t) + Vb(t) = 1
        scaling = sum(V(t,:));
        if scaling == 0
            scaling = eps; % eps added to make sure scaling is never 0, otherwise scaled values become NaNs
        end
        V(t,:) = V(t,:)/scaling;

    elseif tr_type(t)==2 %play trial

        prior_V(t,:) = V(t-1,:);
        V(t,:) = prior_V(t,:);
        
        %calculate value of each action by multipyling matrix of slot machine
        %contingencies (token to action mapping) by token values
        AV(t,:) = V(t,:)*SM_struct{tr_unc(t),HO}; 
        AV(t,UA) = 0;
        
        sca = sum(AV(t,:)); 
        if sca == 0
            sca = eps; % eps added to make sure scaling is never 0, otherwise scaled values become NaNs
        end
        AV(t,:) = AV(t,:)/sca;
        
        %calculate probability of choosing left-most option
        AV_av = AV(t,:);
        AV_av(UA)=[]; %isolate the probabilities of the 2 available actions
        val_diff = AV_av(1) - AV_av(2); %always left minus right difference
        
        %if choice value is 1, use one part of likelihood contribution (P=1/(1+exp))
        %if choice value is 0, use other part of likelihood contribution (P=1-1/(1+exp))
        P_left(t) = 1-choice(t) + (2*choice(t)-1)*(1+exp(-params(1)*val_diff))^-1;      
    end 
end
% adjust 0 probability trials
P_left(P_left < 1e-10) = 1e-10;

f = -nansum(log(P_left)); %negative value of loglikelihood
