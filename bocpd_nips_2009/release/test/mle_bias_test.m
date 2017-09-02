
clear
close all

rand('state', 5);
randn('state', 4);

model_params = rand(1,4)+1;
hazard_param = .01;
N = 50;

hazard_params_learned = zeros(N, 1);
model_params_learned = zeros(N, 4);
nlml = zeros(N, 1);
for ii = 1:N
  [X, changePoints] = IFMrnd(1000, 1, model_params, .01);
  [hazard_params_learned(ii), model_params_learned(ii, :), nlml(ii)] = learn_bocpd(X);
end



[h_haz, p_haz, ci_haz] = ttest(x, hazard_param);
[h_model, p_model, ci_model] = ttest(x, model_params);
