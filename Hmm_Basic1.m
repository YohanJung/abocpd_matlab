%% Demo Regime Switching HMM
clc;
close all;
clear all;
%%cd('C:\Users\YOHAN\Documents\MATLAB\Paper_Journal\');
addpath(genpath('C:\Users\YOHAN\Documents\MATLAB\Paper_Journal'));

%addpath(genpath('C:\Users\YOHAN\Documents\MATLAB\Paper_Journal\EXP_Code_Organization'));
%addpath(genpath('C:\Users\YOHAN\Documents\MATLAB\Paper_Journal\Reference_Code\GPdyn-master'));
%addpath(genpath('C:\Users\YOHAN\Documents\MATLAB\Paper_Journal\Utility'))

%%
Syn_Option = 'Syn2';
NameSeason =  strcat('Journal_',Syn_Option);

formatOut = 'yymmdd_HHMMSS';
datestr(datetime('now'),formatOut);
foldername = strcat('EXP',datestr(datetime('now'),formatOut),'_BOCPD_',NameSeason);
%mkdir(foldername);
%address_file = strcat(pwd,'\',foldername,'\');
address_file = pwd;
%%
[Input_Pair Index_Info Period Day Corrupted_Input_Pair aday_length]= Data_Input_Synthesis(address_file,Syn_Option);
%cd(address_file)

%% Data Input Processing
Obs = Input_Pair(Index_Info.Training_Index,4)';
Obs2 = [0 (Input_Pair(2:Index_Info.Training_Index(end),4)-Input_Pair(1:Index_Info.Training_Index(end)-1,4))'];


%% Data Input Figure
fig_size = [100,10,1000,600];
fig = figure();
set(gca(), 'LooseInset', get(gca(), 'TightInset'));
set(fig,'position',fig_size);
set(0,'defaulttextinterpreter','latex')

subplot(2,1,1)
plot(Obs)
xlim([0,1000])
subplot(2,1,2)
plot(Obs2)
xlim([0,1000])


%% EM-Setup
[Dim_Obs,Num_Obs] = size(Obs);
Num_state = 2;
Hmm_Model.Num_state = Num_state;
Hmm_Model.A = rand(Hmm_Model.Num_state,Hmm_Model.Num_state) ;
Hmm_Model.A = Hmm_Model.A./sum(Hmm_Model.A,2);

Hmm_Model.Pi = rand(Hmm_Model.Num_state,1);
Hmm_Model.Pi = Hmm_Model.Pi./sum(Hmm_Model.Pi,1);

Hmm_Model.E = cell(Hmm_Model.Num_state,2);
Hmm_Model.E{1,1} = randn(Dim_Obs,1); Hmm_Model.E{1,2} = abs(randn(Dim_Obs,Dim_Obs));
Hmm_Model.E{2,1} = randn(Dim_Obs,1); Hmm_Model.E{2,2} = abs(randn(Dim_Obs,Dim_Obs));

% Hmm_Model.E{1,1} = randn(Dim_Obs,1); Hmm_Model.E{1,2} = abs(randn(Dim_Obs,Dim_Obs));
% Hmm_Model.E{2,1} = randn(Dim_Obs,1); Hmm_Model.E{2,2} = abs(randn(Dim_Obs,Dim_Obs));
 
% A_T = Hmm_Model.A';
% A = Hmm_Model.A;
E = zeros(Hmm_Model.Num_state ,Num_Obs);

%% EM-Algorithm
Max_Num_Iter = 500;
tol = 1e-3;
llh(1) = -inf;

%%
[Hmm_Model2 llh] = Hmm_EM(Obs,Hmm_Model,Max_Num_Iter)

%%

for it = 2:Max_Num_Iter

    A = Hmm_Model.A; A_T = A';
    Pi = Hmm_Model.Pi ;
    % EM - Expectation

    for i = 1:Num_Obs
        E(1,i) = mvnpdf(Obs(i),Hmm_Model.E{1,1},Hmm_Model.E{1,2});
        E(2,i) = mvnpdf(Obs(i),Hmm_Model.E{2,1},Hmm_Model.E{2,2});
    end
    alpha(:,1) = Hmm_Model.Pi.*E(:,1);
    c(1) = sum(alpha(:,1));
    alpha(:,1) = alpha(:,1)./c(1);    
    for i = 2:Num_Obs
        alpha(:,i) = E(:,i).*A_T*alpha(:,i-1); 
        c(i) = sum(alpha(:,i));
        alpha(:,i) = alpha(:,i)./c(i);
    end
    beta(:,Num_Obs) = ones(Num_state,1);
    %beta(:,Num_Obs) = beta(:,Num_Obs)./sum(beta(:,Num_Obs));
    for i = Num_Obs:-1:2
        beta(:,i-1) = A*(E(:,i).*beta(:,i))./c(i); 
        %beta(:,i-1) = beta(:,i-1)./sum(beta(:,i-1));
    end

    gamma =  alpha.*beta;
    llh(it) = sum(log(c(c>0)));
    if llh(it)-llh(it-1) < tol*llh(it-1) 
        break
    end
    % EM - Maximization
    Pi = gamma(:,1)./sum(gamma(:,1));
    for i = 2:Num_Obs
        xi{i} = A.*(alpha(:,i-1)*(beta(:,i).*E(:,i))');
    end
    Anew = sum(cat(3, xi{:}),3);
    Anew = Anew./sum(Anew,2);

    for i = 1:Num_state
        Mu_new{i,1} = gamma(i,:)*Obs';
        Mu_new{i,1} = Mu_new{i,1}./sum(gamma(i,:));    
        Sigma_new{i,1} = (gamma(i,:).*(Obs-Mu_new{i,1}))*(Obs-Mu_new{i,1})';
        Sigma_new{i,1} = Sigma_new{i,1}./sum(gamma(i,:));
    end
    
    %llh(it);
    
    Hmm_Model.A = Anew ;
    Hmm_Model.Pi = Pi;
    Hmm_Model.E{1,1} = Mu_new{1,1}; Hmm_Model.E{1,2} = Sigma_new{1,1};
    Hmm_Model.E{2,1} = Mu_new{2,1}; Hmm_Model.E{2,2} = Sigma_new{2,1};
    
end

%% EM-llh
fig_size = [100,10,1000,600];
fig = figure();
set(gca(), 'LooseInset', get(gca(), 'TightInset'));
set(fig,'position',fig_size);
set(0,'defaulttextinterpreter','latex')

plot(llh)

%% Hmm_Filter

alpha(:,1) = Hmm_Model.Pi.*E(:,1);
c(1) = sum(alpha(:,1));
alpha(:,1) = alpha(:,1)./c(1);    

for i = 2:Num_Obs


    alpha(:,i) = E(:,i).*A_T*alpha(:,i-1); 
    c(i) = sum(alpha(:,i));
    alpha(:,i) = alpha(:,i)./c(i);
end

index_Hidden = argmax(alpha);
id1 = find(index_Hidden == 1);
id2 = find(index_Hidden == 2);

%% Hmm_Filter Figure
fig_size = [100,10,1000,600];
fig = figure();
set(gca(), 'LooseInset', get(gca(), 'TightInset'));
set(fig,'position',fig_size);
set(0,'defaulttextinterpreter','latex')

subplot(2,1,1)
plot(Obs,'k')
hold on
scatter(id1,Obs(id1),'*')
hold on
scatter(id2,Obs(id2),'filled')
xlim([0,200])

subplot(2,1,2)
plot(Obs2)
xlim([0,200])

%%  Hmm_Filter_TestSet
ObsT =  Input_Pair(Index_Info.Test_Index,4)';
[Dim_ObsT,Num_ObsT] = size(ObsT);

ET(1,1) = mvnpdf(ObsT(1),Hmm_Model.E{1,1},Hmm_Model.E{1,2});
ET(2,1) = mvnpdf(ObsT(1),Hmm_Model.E{2,1},Hmm_Model.E{2,2});

alphaT(:,1) = Hmm_Model.Pi.*ET(:,1);
c(1) = sum(alphaT(:,1));
alphaT(:,1) = alphaT(:,1)./c(1);    


for i = 2:Num_ObsT
    ET(1,i) = mvnpdf(ObsT(i),Hmm_Model.E{1,1},Hmm_Model.E{1,2});
    ET(2,i) = mvnpdf(ObsT(i),Hmm_Model.E{2,1},Hmm_Model.E{2,2});
    
    alphaT(:,i) = ET(:,i).*A_T*alphaT(:,i-1); 
    c(i) = sum(alphaT(:,i));
    alphaT(:,i) = alphaT(:,i)./c(i);
end

index_Hidden = argmax(alphaT);
id1 = find(index_Hidden == 1);
id2 = find(index_Hidden == 2);

%%
alphaTT(:,1) = Hmm_Model.Pi.*ET(:,1);
c(1) = sum(alphaTT(:,1));
alphaTT(:,1) = alphaTT(:,1)./c(1);    

for i = 2:Num_ObsT
    alphaTT(:,i) = Hmm_filter(ObsT(i),Hmm_Model2,alphaTT(:,i-1));

end
index_Hidden = argmax(alphaTT);
id1 = find(index_Hidden == 1);
id2 = find(index_Hidden == 2);
%% Hmm_Filter Figure_TestSet
fig_size = [100,10,1000,600];
fig = figure();
set(fig,'position',fig_size);
set(gca(), 'LooseInset', get(gca(), 'TightInset'));
set(0,'defaulttextinterpreter','latex')

plot(ObsT,'k','LineWidth',1.0)
hold on
scatter(id1,ObsT(id1),'filled')
hold on
scatter(id2,ObsT(id2),'filled')


