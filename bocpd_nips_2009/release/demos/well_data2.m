% Ryan Turner (rt324@cam.ac.uk)
% Evaluate the BOCPS learning procedure on the well data and compare to RPA hand
% picked hyper-parameters.

clear
close all
% dbstop if naninf

disp('Trying well log data');

% I think all this code is deterministic, but let's fix the seed to be safe.
randn('state', 4);
rand('state', 4);

load data/well.dat;

% We don't know the physical interpretation, so lets just standardize the
% readings to make them cleaner.
X = standardize_cols(well);
X = X(1:1050,1);
Tlearn = 500;
Ttest = size(X, 1) - Tlearn;

useLogistic = true;

assert(isVector(X));
assert(size(X, 2) == 1);

%%

% TODO compare logistic h and constant h

% Can try learn_IFM usinf IFMfast to speed this up
disp('Starting learning');
tic;
[well_hazard, well_model, well_learning] = learn_bocpd(X(1:Tlearn), useLogistic);
toc;
disp('Learning Done');
%%
disp('Testing');
tic;
[well_R, well_S, well_nlml, Z] = bocpd(X, 'gpr', well_model', 'logistic_h', well_hazard');
toc;
disp('Done Testing');

nlml_score = -sum(log(Z(Tlearn + 1:end))) / Ttest;
%%
% TODO assert we get the same results as [RPA] without doing standardizing
% from the paper sec 3.1
rpa_hazard = 1 / 250;
rpa_mu0 = 1.15e5;
rpa_mu_sigma = 1e4;

% correct for the effects of standardizing
rpa_mu0 = (rpa_mu0 - mean(well)) / std(well);
rpa_mu_sigma = rpa_mu_sigma / std(well);

% convert to precision
rpa_kappa = 1 / rpa_mu_sigma ^ 2;

% unstated what he uses for variance parameters.
% some reasonable defaults
rpa_alpha = 1;
rpa_beta = rpa_kappa;

rpa_model = [rpa_mu0 1 rpa_alpha rpa_beta];
[well_R, well_S_rpa, well_nlml_rpa, Z_rpa] = ...
  bocpd(X, 'gpr', rpa_model, 'constant_h', rpa_hazard);

nlml_score_rpa = -sum(log(Z_rpa(Tlearn + 1:end))) / Ttest;

TIM_nlml = -sum(normlogpdf(X(Tlearn + 1:end))) / Ttest;

save well_results
%%
plotS(well_S, X);
title(['RDT ' num2str(nlml_score)]);
%%
plotS(well_S_rpa, X);
title(['RPA ' num2str(nlml_score_rpa)]);
