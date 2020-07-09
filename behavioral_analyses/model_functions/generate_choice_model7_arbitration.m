function P_acc = generate_choice_model7_arbitration(params,P)

%This function describes a model to estimate arbitration
%between imitation and emulation.
%Includes 4 parameters: emulation decision softmax beta, imitation decision
%softmax beta, bias towards one strategy, and imitation learning rate.
%------------ Caroline Charpentier ----------------

lambda=0.99999;
tr_nb = length(P(:,1));

%initialize variables
prior_V = zeros(tr_nb,3); %prior token values for EM (each row=trial, each column=token)
V = zeros(tr_nb,3); %posterior token values for EM (each row=trial, each column=token)
Vim = zeros(tr_nb,3); %action values for IM (each row=trial, each column=action)
AVem = zeros(tr_nb,3); %available action values for EM (each row=trial, each column=action)
AVim = zeros(tr_nb,3); %available action values for IM (each row=trial, each column=action)
Pl_em = zeros(tr_nb,1); %probability of choosing left according to emulation
Pl_im = zeros(tr_nb,1); %probability of choosing right according to imitation
w = zeros(tr_nb,1); %arbitration weight (each row=trial)
PEim = zeros(tr_nb,1); %imitation PEs (each row=trial)
min_ape = zeros(tr_nb,1); %keep track of minimum abs APE (each row=trial)
max_ape = zeros(tr_nb,1); %keep track of maximum abs APE (each row=trial)
xIM = zeros(tr_nb,1); %unreliability of imitation (each row=trial)
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
tr_unc = P(:,7); %low uncertainty(1)/high uncertainty(2)
unav_act = P(:,8); %unavailable action
part_act = P(:,9); %partner's action (always = correct action)
SC = P(:,11); %subject's choice
Slr = P(:,12); %subject's choice (left 1/right 0)
hord = P(:,16); %horizontal order

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
            Vim(t,UA) = 0; %do not update value of unavailable options
            Vim(t,PA) = params(4); %positively update value of action chosen by partner
            Vim(t,UnchA) = -params(4); %negatively update value of unchosen
            
        else %no switch possible on first trial
            
            prior_V(t,1) = lambda*V(t-1,1) + (1-lambda)*(1/2)*(V(t-1,2) + V(t-1,3));
            prior_V(t,2) = lambda*V(t-1,2) + (1-lambda)*(1/2)*(V(t-1,1) + V(t-1,3));
            prior_V(t,3) = lambda*V(t-1,3) + (1-lambda)*(1/2)*(V(t-1,1) + V(t-1,2));
            
            %update value of two available choices
            Vim(t,UA) = Vim(t-1,UA); %do not update value of unavailable options
            Vim(t,PA) = Vim(t-1,PA) + params(4)*(1-Vim(t-1,PA)); %positively update value of action chosen by partner
            Vim(t,UnchA) = Vim(t-1,UnchA) + params(4)*(-1-Vim(t-1,UnchA)); %negatively update value of unchosen  
            
        end

        %baysian update prior * likelihood
        V(t,:) = prior_V(t,:).*P_PA_V;

        %scale probas such that Vg(t) + Vr(t) + Vb(t) = 1
        scaling = sum(V(t,:)); 
        if scaling == 0
            scaling = eps; % eps added to make sure scaling is never 0, otherwise scaled values become NaNs
        end
        V(t,:) = V(t,:)/scaling;
        
    elseif tr_type(t)==2 %play
        
        %values under emulation model
        prior_V(t,:) = V(t-1,:);
        V(t,:) = prior_V(t,:);
        
        %values under imitation model
        Vim(t,:) = Vim(t-1,:);

        min_ape(t) = min_ape(t-1);
        max_ape(t) = max_ape(t-1);
        min_ent(t) = min_ent(t-1);
        max_ent(t) = max_ent(t-1);

        xIM(t) = xIM(t-1);
        xEM(t) = xEM(t-1);
        w(t) = w(t-1);
        
    end
        
    %Emulation - calculate value of each action by multipyling matrix of slot machine
    %contingencies (token to action mapping) by token values
    AVem(t,:) = V(t,:)*SM_struct{tr_unc(t),HO};
    AVem(t,UA)=0; %isolate the probabilities of the 2 available actions
    sca = sum(AVem(t,:)); 
    if sca == 0
        sca = eps; % eps added to make sure scaling is never 0, otherwise scaled values become NaNs
    end
    AVem(t,:) = AVem(t,:)/sca;
    
    %Imitation - calculate action values
    AVim(t,:) = Vim(t,:);
    AVim(t,UA)=0; %isolate the probabilities of the 2 available actions
    
    %Calculate reliability values for each strategy
    if tr_type(t)==1 %observe
        %Calculate PE (of the partner's action value)
        %PEim calculated from chosen action PE
        if P(t,3)==1
            %on the first trial, actions are equiprobable
            PEim(t) = 1;
        else
            Vprev = Vim(t-1,:);
            Vprev(UA) = 0; 
            PEim(t) = 1-Vprev(PA); %calculate PE  (always positive)
        end
        
        %calculate unreliability of imitation strategy
        %unreliability based on min/max normalized unsigned action prediction error
        if t==1
            min_ape(t) = 0; %true min(PE) from practice trials (averaged across subjects)
            max_ape(t) = 1.91614; %true max(PE) from practice trials (averaged across subjects)
            if PEim(t)<min_ape(t)
                min_ape(t) = PEim(t);
            elseif PEim(t)>max_ape(t)
                max_ape(t) = PEim(t);
            end
        else
            if PEim(t)<min_ape(t-1) %if PE is smaller than previous min_ape then it becomes the new min
                min_ape(t) = PEim(t);
            else
                min_ape(t) = min_ape(t-1);
            end
            if PEim(t)>max_ape(t-1) %if PE is larger than previous max_ape then it becomes the new max
                max_ape(t) = PEim(t);
            else
                max_ape(t) = max_ape(t-1);
            end
        end
        if max_ape(t)>min_ape(t)
            xIM(t) = (PEim(t)-min_ape(t))/(max_ape(t)-min_ape(t));
        else
            xIM(t) = 0;
        end
                
        %Calculate unreliability of emulation strategy
        %first calculate entropy
        if sum(AVem(t,[PA UnchA]))~=0
            entropy(t) = -(sum(AVem(t,[PA UnchA]).*log2(AVem(t,[PA UnchA]))));
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
        %Difference in reliability pushed through a sigmoid to bound  the value between 0-->1
        %Since xEM and xIM represent the UNreliability, take minus of these to get reliability
        rdiff = -(xEM(t) - xIM(t)); %reliability of EM over IM
        w(t) = (1+exp(-(params(3) + rdiff)))^-1; 
        %params(3) is the bias towards one strategy over the other
        
    end
        
    %Calculate action values and choice probabilities for each strategy
    AVem_c = AVem(t,:);
    AVem_c(UA) = [];
    val_diff_em = AVem_c(1) - AVem_c(2);
    Pl_em(t) = (1+exp(-params(1)*val_diff_em))^-1;

    AVim_c = AVim(t,:);
    AVim_c(UA) = [];
    val_diff_im = AVim_c(1) - AVim_c(2);
    Pl_im(t) = (1+exp(-params(2)*val_diff_im))^-1;
    
    %Combine choice probabilities with arbitration weight
    P_left(t) = w(t)*Pl_em(t) + (1-w(t))*Pl_im(t);

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
P_acc = [prior_V V AVem Vim AVim xIM xEM w Pl_em Pl_im P_left pred_ch PC corr2 ll ll_corr corr ll_S PEim min_ape max_ape entropy min_ent max_ent];