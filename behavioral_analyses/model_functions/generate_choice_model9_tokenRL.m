function P_acc = generate_choice_model9_tokenRL(params,P)

%model that learns from the token shown at the end of a trial
%------------ Caroline Charpentier ----------------

tr_nb = length(P(:,1));

%initialize variables
V = zeros(tr_nb,3); %token values
AV = zeros(tr_nb,3); %action values

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
Slr = P(:,12); %subject's choice (1: left, 0: right)
hord = P(:,16); %horizontal order
token_s = P(:,17); %token shown

P_left   = NaN(tr_nb,1);
pred_ch  = NaN(tr_nb,1); %predicted choice: left (1), right (0)
corr     = NaN(tr_nb,1);
corr2    = NaN(tr_nb,1);
ll       = NaN(tr_nb,1);
PC       = NaN(tr_nb,1); %predicted action (1 to 3)
ll_corr  = NaN(tr_nb,1);
ll_S     = NaN(tr_nb,1);

for t=1:tr_nb
    
    UA = unav_act(t); %unavailable action
    PA = part_act(t);
    
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

    elseif tr_type(t)==2 %play trial
        
        V(t,:) = V(t-1,:) + params(2)*(O-V(t-1,:));
        
    end

    %calculate value of each action by multipyling matrix of slot machine
    %contingencies (token to action mapping) by token values
    AV(t,:) = V(t,:)*SM_struct{tr_unc(t),hord(t)}; 
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
