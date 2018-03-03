%% Data Input Processing
clc;
close all;
clear all;
addpath(genpath('C:\Users\YOHAN\Documents\MATLAB\Paper_Journal'));
Syn_Option = 'Syn11';
address_file = pwd;

[Input_Pair Index_Info Period Day Corrupted_Input_Pair aday_length]= Data_Input_Synthesis(address_file,Syn_Option);
Obs = Input_Pair(Index_Info.Training_Index,4);
lag = 1;
[Full_Input,Full_Output]= Preprocessing_BenchMarkData(Obs,lag,[]);

start = 1; Middle = 500; num_data = 2000; 
Training_Input = Full_Input(start:start + Middle-1,:); Training_Output = Full_Output(start : start + Middle -1,:);
Test_Input = Full_Input(start + Middle:end,:); Test_Output = Full_Output(start + Middle:end,:);

%% Switch Hmm SetUp
SHmm_Model.Obs_dim = lag;
SHmm_Model.Swiching_Num = 2;
Num_state_List = [2,2];
for ii = 1:SHmm_Model.Swiching_Num
    SHmm_Model.Num_state(ii,1) = Num_state_List(ii);
    SHmm_Model.A{ii,1} = rand(SHmm_Model.Num_state(ii,1),SHmm_Model.Num_state(ii,1));
    SHmm_Model.Pi{ii,1} = rand(SHmm_Model.Num_state(ii,1),1);
end
%% Emission Setup individually

% swich state 1 - Emisson setup
SHmm_Model.E{1,1,1} = randn(SHmm_Model.Obs_dim,1);   SHmm_Model.E{1,2,1} = eye(SHmm_Model.Obs_dim);
SHmm_Model.E{2,1,1} = randn(SHmm_Model.Obs_dim,1);   SHmm_Model.E{2,2,1} = eye(SHmm_Model.Obs_dim);

% swich state 2 - Emisson setup

SHmm_Model.E{1,1,2} = randn(SHmm_Model.Obs_dim,1);   SHmm_Model.E{1,2,2} = eye(SHmm_Model.Obs_dim);
SHmm_Model.E{2,1,2} = randn(SHmm_Model.Obs_dim,1);   SHmm_Model.E{2,2,2} = eye(SHmm_Model.Obs_dim);

%% Switching Variable 
SHmm_Model.SPi = rand(SHmm_Model.Swiching_Num,1) ;
SHmm_Model.SA = rand(SHmm_Model.Num_state(ii,1),SHmm_Model.Num_state(ii,1));

%% Switch Hmm EM
Obs = Obs';
[Num_Dim , Num_Obs] = size(Obs);


Max_Num_Iter = 500;
tol = 1e-3;
llh(1) = -inf;
SEmission_Prob = zeros(SHmm_Model.Num_state,Num_Obs,SHmm_Model.Swiching_Num);

for it = 2:Max_Num_Iter

    % Expectation Part in EM Algorithm
    for kk = 1:SHmm_Model.Swiching_Num
        for jj = 1:SHmm_Model.Num_state
            for ii = 1:Num_Obs
                SEmission_Prob(jj,ii,kk) = Hmm_Gaussian_Emission(Obs(:,ii),SHmm_Model.E{jj,1,kk},SHmm_Model.E{jj,2,kk});
            end
        end
        
        
        Temp_A = SHmm_Model.A{kk,1}; Temp_A_t = Temp_A';
        
        
        Salpha(:,1,kk) = SHmm_Model.Pi{kk,1}.*SEmission_Prob(jj,ii,kk);
        
        for ii = 2:Num_Obs
            Salpha(:,ii,kk) =  SEmission_Prob(:,ii,kk).*(Temp_A_t*Salpha(:,ii-1,kk));
        end
        
        Sbeta(:,Num_Obs,kk) = ones(SHmm_Model.Num_state(kk,1),1);
        for ii = Num_Obs-1:-1:1
            Sbeta(:,ii,kk) = SEmission_Prob(:,ii,kk).*(Temp_A*Sbeta(:,ii+1,kk));
       
        end
        
        Sgamma = Salpha.*Sbeta; 
                 
    end
    
    llh(it) = sum(log(c(c>0)));
    if llh(it)-llh(it-1) < tol*llh(it)
        break;
    end 

    % Maximization Part in EM Algorithm
    
    
    
    
    
    



end
















