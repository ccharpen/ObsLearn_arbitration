function f = LL_function_Mod8_forBIC(params,P)

params(1) = exp(params(1));  % emulation decision softmax beta [0 Inf]
params(2) = exp(params(2));  % imitation decision softmax beta [0 Inf]
%params(3) = bias, no transformation [-Inf Inf]
params(4) = 1/(1+exp(-params(4)));  % learning rate for imitation values [0 1]
params(5) = 1/(1+exp(-params(5)));  % lambda [0 1]

tr_nb = length(P(:,1));

%initialize variables
prior_V = zeros(tr_nb,3); %prior token values for BI (each row=trial, each column=token)
V = zeros(tr_nb,3); %posterior token values for BI (each row=trial, each column=token)
Vim = zeros(tr_nb,3); %action values for imit (each row=trial, each column=action)
AVbi = zeros(tr_nb,3); %available action values for BI (each row=trial, each column=action)
AVim = zeros(tr_nb,3); %available action values for IM (each row=trial, each column=action)
w = zeros(tr_nb,1); %arbitration weight (each row=trial)
PEim = zeros(tr_nb,1); %imitation PEs (each row=trial)
min_ape = zeros(tr_nb,1); %keep track of minimum abs APE (each row=trial)
max_ape = zeros(tr_nb,1); %keep track of maximum abs APE (each row=trial)
xIM = zeros(tr_nb,1); %unreliability of imitation (each row=trial)
entropy = zeros(tr_nb,1); %entropy (each row=trial)
min_ent = zeros(tr_nb,1); %keep track of minimum entropy (each row=trial)
max_ent = zeros(tr_nb,1); %keep track of maximum entropy (each row=trial)
xBI = zeros(tr_nb,1); %unreliability of BI (each row=trial)

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

%extract relevant columns from P matrix
unav_act = P(:,8);
part_act = P(:,9); %partner's action (always = correct action)
hord     = P(:,16); %horizontal order
tr_type  = P(:,4); %obs(1)/play(2)
tr_bu    = P(:,7); %low BU(1)/high BU(2)
choice   = P(:,12); %subject's choice (1: left, 0: right)

P_left   = NaN(tr_nb,1);

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
             
            prior_V(t,1) = params(5)*V(t-1,1) + (1-params(5))*(1/2)*(V(t-1,2) + V(t-1,3));
            prior_V(t,2) = params(5)*V(t-1,2) + (1-params(5))*(1/2)*(V(t-1,1) + V(t-1,3));
            prior_V(t,3) = params(5)*V(t-1,3) + (1-params(5))*(1/2)*(V(t-1,1) + V(t-1,2));
            
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
        
        %BI - calculate value of each action by multipyling matrix of slot machine
        %contingencies (token to action mapping) by token values
        AVbi(t,:) = V(t,:)*SM_struct{tr_bu(t),HO};
        AVbi(t,UA)=0; %isolate the probabilities of the 2 available actions
        sca = sum(AVbi(t,:)); 
        if sca == 0
            sca = eps; % eps added to make sure scaling is never 0, otherwise scaled values become NaNs
        end
        AVbi(t,:) = AVbi(t,:)/sca;
        
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
        
        %calculate unreliability of imitation model
        %unreliability based on min/max normalized action prediction error
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
                
        %Calculate unreliability of Bayesian model
        %first calculate entropy
        if sum(AVbi(t,[PA UnchA]))~=0
            entropy(t) = -(sum(AVbi(t,[PA UnchA]).*log2(AVbi(t,[PA UnchA]))));
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
            xBI(t) = (entropy(t)-min_ent(t))/(max_ent(t)-min_ent(t));
        else
            xBI(t) = 0;
        end
        
        %Calculate arbitration weight (p(BI)) as the ratio between
        %reliability of BI over reliability of Imit
        %Difference in reliability pushed through a sigmoid to bound  the value between 0-->1
        %Since xBI and xIM represent the UNreliability, take minus of these to get reliability
        rdiff = -(xBI(t) - xIM(t)); %reliability of BI over imit
        w(t) = (1+exp(-(params(3) + rdiff)))^-1; %do not estimate inverse temperature
        %params(2) is the bias towards one strategy over the other
        
    elseif tr_type(t)==2 %play
        
        %values under bayesian inference model
        prior_V(t,:) = V(t-1,:);
        V(t,:) = prior_V(t,:);
        
        %values under RL token shown model
        Vim(t,:) = Vim(t-1,:);

        min_ape(t) = min_ape(t-1);
        max_ape(t) = max_ape(t-1);
        min_ent(t) = min_ent(t-1);
        max_ent(t) = max_ent(t-1);

        xIM(t) = xIM(t-1);
        xBI(t) = xBI(t-1);
        w(t) = w(t-1);
                                
        %BI - calculate value of each action by multipyling matrix of slot machine
        %contingencies (token to action mapping) by token values
        AVbi(t,:) = V(t,:)*SM_struct{tr_bu(t),HO};
        AVbi(t,UA)=0; %isolate the probabilities of the 2 available actions
        sca = sum(AVbi(t,:)); 
        if sca == 0
            sca = eps; % eps added to make sure scaling is never 0, otherwise scaled values become NaNs
        end
        AVbi(t,:) = AVbi(t,:)/sca;
        
        %IMIT action values
        AVim(t,:) = Vim(t,:);
        AVim(t,UA)=0; %isolate the probabilities of the 2 available actions
        
        %Calculate action valueS
        AVbi_c = AVbi(t,:);
        AVbi_c(UA) = [];
        val_diff_bi = AVbi_c(1) - AVbi_c(2);
        Pl_bi = (1+exp(-params(1)*val_diff_bi))^-1;

        AVim_c = AVim(t,:);
        AVim_c(UA) = [];
        val_diff_im = AVim_c(1) - AVim_c(2);
        Pl_im = (1+exp(-params(2)*val_diff_im))^-1;

        Pl = w(t)*Pl_bi + (1-w(t))*Pl_im;

        %calculate likelihood
        if choice(t) == 1
            P_left(t) = Pl;
        elseif choice(t) == 0
            P_left(t) = 1-Pl;
        end
    end 
end
% adjust 0 probability trials
P_left(P_left < 1e-10) = 1e-10;

f = -nansum(log(P_left)); %negative value of loglikelihood