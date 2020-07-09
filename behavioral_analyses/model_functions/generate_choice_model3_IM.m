function P_acc = generate_choice_model3_IM(params,P)

%this function generates choice for an imitation RL model learning
%the value of each action (slot machines)
%update is based on actions taken by the partner in the past:
%- value of chosen action updated positively
%- value of unchosen action updated negatively
%- value of unavailable actions not updated
%Learning rate is a free parameter estimated for each subject.
%------------ Caroline Charpentier ----------------

tr_nb = length(P(:,1));

%initialize variables
V = zeros(tr_nb,3); %each row=trial, each column=action

%careful, the column indices may change
tr_type = P(:,4); %obs(1)/play(2)
unav_act = P(:,8); %unavailable action
part_act = P(:,9); %partner's action (always = correct action)
SC = P(:,11); %subject's choice
Slr = P(:,12); %subject's choice (left 1/right 0)

P_left   = NaN(tr_nb,1);
pred_ch  = NaN(tr_nb,1);
corr     = NaN(tr_nb,1);
corr2    = NaN(tr_nb,1);
ll       = NaN(tr_nb,1);
PC       = NaN(tr_nb,1); %predicted action (1 to 3)
ll_corr  = NaN(tr_nb,1);
ll_S     = NaN(tr_nb,1);

for t=1:tr_nb
    
    UA = unav_act(t); %unavailable action
    PA = part_act(t); %partner's action
    UnchA = [1 2 3];
    UnchA([UA PA]) = []; %unchosen action
        
    if tr_type(t)==1 %observe trial
        
        if P(t,3)==1 %initialize prior on first trial of each run            
            V(t,UA) = 0; %do not update value of unavailable options
            V(t,PA) = params(2); %positively update value of action chosen by partner
            V(t,UnchA) = -params(2); %negatively update value of unchosen
        else            
            V(t,UA) = V(t-1,UA); %do not update value of unavailable options
            V(t,PA) = V(t-1,PA) + params(2)*(1-V(t-1,PA)); %positively update value of action chosen by partner
            V(t,UnchA) = V(t-1,UnchA) + params(2)*(-1-V(t-1,UnchA)); %negatively update value of unchosen                                      
        end
        
    elseif tr_type(t)==2 %play trial
        
        V(t,:) = V(t-1,:);
        
    end
    
    %calculate probability of choosing left-most option
    AV_av = V(t,:);
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

P_acc = [V P_left pred_ch PC corr2 ll ll_corr corr ll_S];
