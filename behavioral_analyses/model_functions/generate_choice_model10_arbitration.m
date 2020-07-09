function P_acc = generate_choice_model10_arbitration(params,P)

%This function describes a model to estimate arbitration
%between 1-trial imitation and emulation.
%------------ Caroline Charpentier ----------------

lambda=0.99999; %fixed to optimal lambda parameter
tr_nb = length(P(:,1));

%initialize variables
prior_V = zeros(tr_nb,3); %prior token values for emulation (each row=trial, each column=token)
V = zeros(tr_nb,3); %posterior token values for emulation (each row=trial, each column=token)
Vim = zeros(tr_nb,3); %action values for imitation (each row=trial, each column=action)
AV_EM = zeros(tr_nb,3); %available action values for emulation (each row=trial, each column=action)
Pl_EM = NaN(tr_nb,1); %probability of choosing left according to emulation
Pl_IM = NaN(tr_nb,1); %probability of choosing right according to imitation
w = zeros(tr_nb,1); %arbitration weight (each row=trial)
entropy = zeros(tr_nb,1); %entropy (each row=trial)
min_ent = zeros(tr_nb,1); %keep track of minimum entropy (each row=trial)
max_ent = zeros(tr_nb,1); %keep track of maximum entropy (each row=trial)
xEM = zeros(tr_nb,1); %unreliability of emulation (each row=trial)

%contingencies of the slot machine (each row=token, each column=action)
SM_struct = cell(2,3);
SM_struct{1,1} = [0.75 0.05 0.2; 0.2 0.75 0.05; 0.05 0.2 0.75]; %probability distribution of EASY slot machine (low BU, horizOrd 1)
SM_struct{1,2} = [0.05 0.2 0.75; 0.75 0.05 0.2; 0.2 0.75 0.05]; %probability distribution of EASY slot machine (low BU, horizOrd 2)
SM_struct{1,3} = [0.2 0.75 0.05; 0.05 0.2 0.75; 0.75 0.05 0.2]; %probability distribution of EASY slot machine (low BU, horizOrd 3)
SM_struct{2,1} = [0.5 0.2 0.3; 0.3 0.5 0.2; 0.2 0.3 0.5]; %probability distribution of HARD slot machine (high BU, horizOrd 1)
SM_struct{2,2} = [0.2 0.3 0.5; 0.5 0.2 0.3; 0.3 0.5 0.2]; %probability distribution of HARD slot machine (high BU, horizOrd 2)
SM_struct{2,3} = [0.3 0.5 0.2; 0.2 0.3 0.5; 0.5 0.2 0.3]; %probability distribution of HARD slot machine (high BU, horizOrd 3)

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

%careful, the column indices may change
tr_type = P(:,4); %obs(1)/play(2)
tr_unc = P(:,7); %low uncertainty(1)/ high uncertainty(2)
unav_act = P(:,8); %unavailable action
part_act = P(:,9); %partner's action (always = correct action)
SC = P(:,11); %subject's choice
Slr = P(:,12); %subject's choice (left 1/right 0)
hord = P(:,16); %horizontal order of slot machines on the screen

%variable to save at the end
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
    UnchA = [1 2 3];
    UnchA([UA PA]) = []; %unchosen action
    
    if tr_type(t)==1 %observe
        
        %Calculate likelihood of partner's action given slot machine and
        %valuable token
        P_PA_V = P_PA_Tok{UA,HO}(:,PA)'; %each row=token
        
        if P(t,3)==1 %initialize prior on first trial of each block
            
            prior_V(t,:) = [1/3 1/3 1/3];
            
            %update value of two available choices
            Vim(t,UA) = 0; %no value of unchosen action
            Vim(t,PA) = 1; %positive value of action chosen by partner
            Vim(t,UnchA) = -1; %negative value of unchosen
            
        else %no switch possible on first trial
            
            prior_V(t,1) = lambda*V(t-1,1) + (1-lambda)*(1/2)*(V(t-1,2) + V(t-1,3));
            prior_V(t,2) = lambda*V(t-1,2) + (1-lambda)*(1/2)*(V(t-1,1) + V(t-1,3));
            prior_V(t,3) = lambda*V(t-1,3) + (1-lambda)*(1/2)*(V(t-1,1) + V(t-1,2));
            
            %update value of two available choices
            Vim(t,UA) = Vim(t-1,UA); %keep previous value
            Vim(t,PA) = 1; %positive value of action chosen by partner
            Vim(t,UnchA) = -1; %negative value of unchosen
            
        end

        %emulation update prior * likelihood
        V(t,:) = prior_V(t,:).*P_PA_V;

        %scale probas such that Vg(t) + Vr(t) + Vb(t) = 1
        scaling = sum(V(t,:)); 
        if scaling == 0
            scaling = eps; % eps added to make sure scaling is never 0, otherwise scaled values become NaNs
        end
        V(t,:) = V(t,:)/scaling;
        
        %Emulation - calculate value of each action by multipyling matrix of slot machine
        %contingencies (token to action mapping) by token values
        AV_EM(t,:) = V(t,:)*SM_struct{tr_unc(t),HO};
        AV_EM(t,UA)=0; %isolate the probabilities of the 2 available actions
        sca = sum(AV_EM(t,:)); 
        if sca == 0
            sca = eps; % eps added to make sure scaling is never 0, otherwise scaled values become NaNs
        end
        AV_EM(t,:) = AV_EM(t,:)/sca;
        
        %Calculate unreliability of emulation
        %first calculate entropy
        if sum(AV_EM(t,[PA UnchA]))~=0
            entropy(t) = -(sum(AV_EM(t,[PA UnchA]).*log2(AV_EM(t,[PA UnchA]))));
        else
            entropy(t) = 0;
        end

        %unreliability based on min/max normalized entropy
        if t==1
            min_ent(t) = 0.3373; %true min(entropy) from practice trials (averaged across subjects)
            max_ent(t) = 0.9680; %true max(entropy) from practice trials (averaged across subjects)
            if entropy(t)<min_ent(t)
                min_ent(t) = entropy(t);
            elseif entropy(t)>max_ent(t)
                max_ent(t) = entropy(t);
            end
        else
            if entropy(t)<min_ent(t-1) %if entropy is smaller than previous min_ent then it becomes the new min
                min_ent(t) = entropy(t);
            else
                min_ent(t) = min_ent(t-1);
            end
            if entropy(t)>max_ent(t-1) %if ent is larger than previous max_ent then it becomes the new max
                max_ent(t) = entropy(t);
            else
                max_ent(t) = max_ent(t-1);
            end
        end
        if max_ent(t)>min_ent(t)
            xEM(t) = (entropy(t)-min_ent(t))/(max_ent(t)-min_ent(t));
        else
            xEM(t) = 0;
        end
        
        %Calculate arbitration weight
        %Since xEM represents the UNreliability, take minus of this to get reliability
        rEM = 1 - 2*xEM(t); %reliability of EM (now ranges from -1 to +1)
        w(t) = (1+exp(-(params(3) + rEM)))^-1; 
       	%params(3) is the bias towards one strategy over the other
        
    elseif tr_type(t)==2 %play
        
        %values under bayesian inference model
        prior_V(t,:) = V(t-1,:);
        V(t,:) = prior_V(t,:);
        
        %values under RL token shown model
        Vim(t,:) = Vim(t-1,:);

        min_ent(t) = min_ent(t-1);
        max_ent(t) = max_ent(t-1);

        xEM(t) = xEM(t-1);
        w(t) = w(t-1);
        
        %Emulation - calculate value of each action by multipyling matrix of slot machine
        %contingencies (token to action mapping) by token values
        AV_EM(t,:) = V(t,:)*SM_struct{tr_unc(t),HO};
        AV_EM(t,UA)=0; %isolate the probabilities of the 2 available actions
        sca = sum(AV_EM(t,:)); 
        if sca == 0
            sca = eps; % eps added to make sure scaling is never 0, otherwise scaled values become NaNs
        end
        AV_EM(t,:) = AV_EM(t,:)/sca;

        %Calculate action values
        AV_EM_c = AV_EM(t,:);
        AV_EM_c(UA) = [];
        val_diff_EM = AV_EM_c(1) - AV_EM_c(2);
        Pl_EM(t) = (1+exp(-params(1)*val_diff_EM))^-1;

        %IMIT action values
        %if action chosen by partner on trial t-1 is available now then
        %this action should be chosen
        if UA ~= part_act(t-1) 
            pred_choice = part_act(t-1);
        else
            t1b = find(P(1:t,3)==1); %trial number of first trial of current block
            tr_cb = [(t1b(end):t-2)' part_act(t1b(end):t-2)~=UA]; %trial list current block 
            l = find(tr_cb(:,2)==1);
            if ~isempty(l)
                tr_rep = tr_cb(l(end),1); %number of trial to be repeated
                pred_choice = part_act(tr_rep);
            else
                pred_choice = 0; %no past evidence to be repeated
            end
        end

        if pred_choice==1 || (pred_choice==2 && UA==1) %predicted choice is left
            val_diff_IM = 1;
        elseif pred_choice==3 || (pred_choice==2 && UA==3) %predicted choice is right
            val_diff_IM = -1;
        elseif pred_choice==0
            val_diff_IM = 0;
        end 
        Pl_IM(t) = (1+exp(-params(2)*val_diff_IM))^-1;

        P_left(t) = w(t)*Pl_EM(t) + (1-w(t))*Pl_IM(t);

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
end
P_acc = [prior_V V AV_EM Vim xEM w Pl_EM Pl_IM P_left pred_ch PC corr2 ll ll_corr corr ll_S entropy min_ent max_ent];