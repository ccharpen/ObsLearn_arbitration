function [bic_score]=bic_score(functionname_bic,thetamean,thetasigma,samplesize,n_variables,n_file,data)
% bic score, kiigaya@gmail.com

rng default;  % For reproducibility
hs = mvnrnd(thetamean,thetasigma,samplesize); %sample from random distribution with given mean and variance
%sample size = 10000 is usually ok (maybe 100000 for paper), check how long it takes

for fi=1:n_file %loop over subjects
    
    bic_sub = fi;
    P = data(fi).P;
    parfor i=1:samplesize
        h_current = hs(i,:);
        likelihood(i) = feval(functionname_bic,h_current,P); %likelihood function (not posterior)
    end

    L = -likelihood;
    delL = L-max(L); %% L=(L-max(L))+max(L)
    A = exp(delL);
    bics(fi) = log(mean(A))+max(L); %mean of likelihood
    l_session(fi) = sum(~isnan(P(:,12))); %number of choice trials
    
end

bic_score = -2*sum(bics) + n_variables*2*log(sum(l_session));

end