function P_acc = generate_choice_model5_EM_RL(params,P)

%this function generate choices for a model inferring the value of each
%token (emulation), but using RL instead of mulplicative (approx. Bayesian)
%inference. Learning rate is a free parameter estimated for each subject.
%evidence given by partner action [P(partner action|goal token & available
%actions)] is binary (either 1 or -1)
%computation of action values from token values are based on the
%probability of getting each token given each slot machine/action chosen
%------------ Caroline Charpentier ----------------

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
tr_unc = P(:,7); %low uncertainty(1)/high uncertainty(2)
unav_act = P(:,8); %unavailable action
part_act = P(:,9); %partner's action (always = correct action)
SC = P(:,11); %subject's choice
Slr = P(:,12); %subject's choice (left 1/right 0)
hord = P(:,16); %horizontal order

%task structure matrices
P_PA_Tok{1,1} = [-1 -1 1;-1 1 -1;-1 -1 1]; %ua 1, horizOrd 1
P_PA_Tok{1,2} = [-1 -1 1;-1 -1 1;-1 1 -1]; %ua 1, horizOrd 2
P_PA_Tok{1,3} = [-1 1 -1;-1 -1 1;-1 -1 1]; %ua 1, horizOrd 3
P_PA_Tok{2,1} = [1 -1 -1;1 -1 -1;-1 -1 1]; %ua 2, horizOrd 1
P_PA_Tok{2,2} = [-1 -1 1;1 -1 -1;1 -1 -1]; %ua 2, horizOrd 2
P_PA_Tok{2,3} = [1 -1 -1;-1 -1 1;1 -1 -1]; %ua 2, horizOrd 3
P_PA_Tok{3,1} = [1 -1 -1;-1 1 -1;-1 1 -1]; %ua 3, horizOrd 1
P_PA_Tok{3,2} = [-1 1 -1;1 -1 -1;-1 1 -1]; %ua 3, horizOrd 2
P_PA_Tok{3,3} = [-1 1 -1;-1 1 -1;1 -1 -1]; %ua 3, horizOrd 3
%here we detail the structure of probability of each partner's action given
%set of available actions and given goal token
%each row of the cell structure {1}, {2}, or {3} represents the unavailable action.
%each column of the cell structure {1}, {2}, or {3} represents the horizontal order.
%within each cell, each column represents the action performed by the partner
%and each row represents the conditional goal token (1:green, 2:red, 3:blue)

P_left   = NaN(tr_nb,1);
pred_ch  = NaN(tr_nb,1); %predicted choice: left (1), right (0)
corr     = NaN(tr_nb,1);
corr2    = NaN(tr_nb,1);
ll       = NaN(tr_nb,1);
PC       = NaN(tr_nb,1); %predicted action (1 to 3)
ll_corr  = NaN(tr_nb,1);
ll_S     = NaN(tr_nb,1);

for t=1:tr_nb

    UA = unav_act(t); %unavailable action #1
    PA = part_act(t); %partner's action
    HO = hord(t); %horizontal order

    if tr_type(t)==1 %only update after observe trials
        
        %Calculate evidence for each token being valuable given partner's action
        token_ev = P_PA_Tok{UA,HO}(:,PA)'; %each row=token
        
        if P(t,3)==1 
            V(t,:) = [0 0 0] + params(2)*(token_ev - [0 0 0]);
        else
            V(t,:) = V(t-1,:) + params(2)*(token_ev-V(t-1,:)); %value update 
        end

    elseif tr_type(t)==2 %play trial
        
        V(t,:) = V(t-1,:);
        
    end 
    
    %calculate value of each action by multipyling matrix of slot machine
    %contingencies (token to action mapping) by token values
    AV(t,:) = V(t,:)*SM_struct{tr_unc(t),HO}; 
    AV(t,UA) = 0;
       
    %calculate probability of choosing left-most option
    AV_av = AV(t,:);
    AV_av(UA)=[]; %isolate the probabilities of the 2 available actions
    val_diff = AV_av(1) - AV_av(2); %always left minus right difference

    P_left(t) = (1+exp(-params(1)*val_diff))^-1;    

    n=rand();
    ind_pos_act = [1 2 3]; %find index of available actions
    ind_pos_act(UA) = [];
    if n<P_left(t)
        pred_ch(t) = 1; %predicted choice is left   
        PC(t) = ind_pos_act(1); %translate into which action (from 1 to 4)
        ll(t) = P_left(t); %loglikelihood
    else
        pred_ch(t) = 0; %predicted choice is right
        PC(t) = ind_pos_act(2); %translate into which action (from 1 to 4)
        ll(t) = 1-P_left(t); %loglikelihood
    end
    if PC(t) == PA
        corr2(t) = 1; %simulated choice is correct
    else
        corr2(t) = 0;
    end        
    if PA==1 || (PA==2 && UA==1) %correct choice is left
        ll_corr(t) = P_left(t);
    elseif PA==3 || (PA==2 && UA==3) %correct choice is right
        ll_corr(t) = (1-P_left(t));
    end   
    if ~isnan(SC(t)) %look at whether model predicts subject choice
        if PC(t) == SC(t)
            corr(t) = 1;
        else
            corr(t) = 0;
        end
    else
        corr(t) = NaN;
    end
    if Slr(t)==1 %calculate likelihood of predicting subject choice
        ll_S(t) = P_left(t);
    elseif Slr(t)==0
        ll_S(t) = 1-P_left(t);
    end
end

P_acc = [V AV P_left pred_ch PC corr2 ll ll_corr corr ll_S];
