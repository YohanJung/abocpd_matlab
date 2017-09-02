% Ryan Turner (rt324@cam.ac.uk)
% Evaluate the BOCPS learning procedure on the industry portfolio data

clear
close all
% dbstop if naninf

disp('Trying industry portfolio data');

Tlearn = 3000;
useReal = true;

% I think all this code is deterministic, but let's fix the seed to be safe.
randn('state', 4);
rand('state', 4);

if useReal
  load data/30_industry.mat;

  % get the time stamp out of the matrix
  year = floor(thirty_industry(:, 1) / 10000);
  month = mod(floor(thirty_industry(:, 1) / 100), 100);
  day = mod(thirty_industry(:, 1), 100);
  time = datenum([year month day]);
  % TODO only fit the covariance and mean from the training part ONLY
  X = whitten(thirty_industry(:, 2:end), Tlearn);
else
  T = 3 * Tlearn;
  D = 10;

  % sample some true parameters
  true_hazard_params = randn(1, 3);
  true_hazard_params(1) = logistic(true_hazard_params - 2);
  true_model_params = randn(D, 4);
  true_model_params(:, 2:end) = exp(true_model_params(:, 2:end));
  true_model_params = true_model_params(:)';

  [X changePoints] = IFMrnd(T, D, true_model_params, true_hazard_params);
end


[T D] = size(X);
Ttest = T - Tlearn;

disp('learn the models independently');
industry_hazard = zeros(3, D);
industry_model = zeros(4, D);
industry_learning = cell(D, 1);
Z = zeros(T, D);
theta_init = [];
%%

for ii = 1:D
  disp(ii);

  tic;
  [industry_hazard(:, ii), industry_model(:, ii), industry_learning{ii}] = ...
    learn_IFM(X(1:Tlearn, ii), true, theta_init);
  toc

  [R, S, nlml, Z(:, ii)] = bocpd_sparse( ...
    industry_hazard(:, ii)', industry_model(:, ii)', X(:, ii), ...
    'logistic_h', 'IFM', .001);

  % initilializer for next iteration
  theta_init = [mean(industry_hazard(:, 1:ii), 2); ...
    mean(industry_model(:, 1:ii), 2)];
end

nlml_score = -sum2(log(Z(Tlearn + 1:end, :))) / Ttest;

%%
disp('Learn the joint');

tmp = industry_model';
theta_init = [mean(industry_hazard, 2); tmp(:)];

%%
tic;
[industry_hazard_joint, industry_model_joint, industry_learning_joint] = ...
  learn_IFM(X(1:Tlearn, :), true, theta_init);
toc
%%
disp('Testing the joint');
[industry_R, industry_S, industry_nlml, Zjoint] = bocpd_sparse( ...
  industry_hazard_joint', industry_model_joint', X, ...
  'logistic_h', 'IFM', .001);

nlml_score_joint = -sum(log(Zjoint(Tlearn + 1:end))) / Ttest;
TIM_nlml = -sum2(normlogpdf(X(Tlearn + 1:end, :))) / Ttest;

% remove the heavy tails of the data
% TODO adapt df depending on the data
df = 4;
Xprime = tcdf(X, df);
Xprime = norminv(Xprime);

[industry_hazard_heavy, industry_model_heavy, industry_learning_heavy] = ...
  learn_IFM(Xprime(1:Tlearn, :), true, ...
  [industry_hazard_joint; industry_model_joint]);

[industry_RH, industry_SH, industry_nlml, Zheavy] = bocpd_sparse( ...
  industry_hazard_heavy', industry_model_heavy', Xprime, ...
  'logistic_h', 'IFM', .001);

% Warning: can't directly compare nlml of heavy tail to non-heavy tail corrected
% versions.  Must correct for non-linear warping first.
nlml_score_heavy = -sum(log(Zheavy(Tlearn + 1:end))) / Ttest;
TIM_prime_nlml = -sum2(normlogpdf(Xprime(Tlearn + 1:end, :))) / Ttest;

save industry_results



%%

%Plot the results for standard and heavy-tailed
figure; 

colormap gray;
imagesc(time, 1:size(industry_S,1), cumsum(industry_S));
Mrun = getMedianRunLength(industry_S);
hold on;
plot(time, Mrun, 'r-');
hold off;
datetick;

%Annotated results for heavy-tailed
figure; 
colormap gray;
start = datenum('20.07.1998','dd.mm.yyyy');
finish = time(end);
startind = find(time >= start);
startind = startind(1);
finishind = find(time >= finish);
finishind = finishind(1);
imagesc(time(startind:finishind), 1:size(industry_SH,1), cumsum(industry_SH(:,startind:finishind)));
historic = [datenum('05.08.1998', 'dd.mm.yyyy'), datenum('10.03.2000','dd.mm.yyyy'), datenum('11.09.2001','dd.mm.yyyy'), ...
    datenum('26.06.2003','dd.mm.yyyy'), datenum('02.11.2004','dd.mm.yyyy'), datenum('01.09.2007','dd.mm.yyyy'), datenum('15.09.2008', 'dd.mm.yyyy')];
%05/08/1998 -> Asia crisis and dot-com bubble
%10/03/2000 -> dot-com bubble burst
%11/09/2001
%26/06/2003 -> interest rate cut
%02/11/2004 -> US election
%01/09/2007 -> Northern Rock bank run, circa beginning of credit crunch
%15/09/2008 -> Lehman Brothers collapse, height of credit crunch
hold on; 
plot(historic, ones(size(historic)), 'g*', 'MarkerSize', 10);
datetick('x');
[Mrun, MT] = getMedianRunLength(industry_SH);
plot(time(startind:finishind), Mrun(startind:finishind), 'r-');
axis tight;
hold off;

figure;
timesub = time(startind:finishind);
tMT = time(MT);
MTsub = tMT(startind:finishind);
plot(timesub, MTsub);
set(gca, 'YTick', timesub(1):7:timesub(end));
datetick('x'); datetick('y',20,'keepticks');
grid;

% plotS(industry_S, X, [], time);
% title(['Joint ' num2str(nlml_score_joint)]);
% 
% plotS(industry_SH, Xprime, [], time);
% title(['Heavy ' num2str(nlml_score_heavy)]);
