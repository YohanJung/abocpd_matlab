% Ryan Turner (rt324@cam.ac.uk)
% Yunus Saatci (ys267@cam.ac.uk)
% Ryan Adams (rpa23@cam.ac.uk)
% Modularized version of fast Bayesian online change point detection (BOCP)
% adapted from: http://www.inference.phy.cam.ac.uk/rpa23/cp/gaussdemo.m
% Based on the paper:
% http://www.inference.phy.cam.ac.uk/rpa23/papers/rpa-changepoint.pdf
% paper refered to as [RPA].
%
% Input:
% X : T x 1 variable containing the time series.
% model_f : function handle to update the posteriors and evaluate the posterior
% predictive density.
% theta_m : 1 x param_count parameters of the priors on new distributions drawn at
% each time point.
% hazard_f : function handle calculating the hazard of a change point.
% theta_h : 1 x hazard_param_count parameters to the hazard function
%
% Outputs:
% R : T + 1 x T + 1 Object giving the run-length probabilities, R(r, t) =
% P(runlength_t-1 = r-1|X_1:t-1). [P]
% lml : log marginal likelihood of the data, X(1:end), under the model
% = P(X_1:T), integrating out all the runlengths. [log P]
%
% Might want to implement pruning for efficiency.
% TODO specify required interface for model_f() and hazard_f()
function [R, S, nlml, Z] = bocpd(X, model_f, theta_m, hazard_f, theta_h)

num_hazard_params = length(theta_h);
num_model_params = length(theta_m);

assert(isKosher(X));
% assert they are row vec
assert(all(size(theta_h) == [1 num_hazard_params]));
assert(all(size(theta_m) == [1 num_model_params]));

% assert that the function handles passed in are valid
if exist(hazard_f) == 0 || exist([model_f, '_update']) == 0 || ...
    exist([model_f, '_predict']) == 0


  error('Required functions do not exist. See preamble.');
end

hazard_f = str2func(hazard_f);
model_update_f = str2func([model_f, '_update']);
model_predict_f = str2func([model_f, '_predict']);
model_init_f = str2func([model_f, '_init']);

[T,D] = size(X); % Number of time point observed. 1 x 1. [s]

% R(r, t) = P(runlength_t-1 = r-1|X_1:t-1).
R = zeros(T + 1, T + 1); % pre-load the run length distribution. [P]
S = zeros(T, T);

% At time t = 1, we actually have complete knowledge about the run
% length.  It is definitely zero.  See the paper for other possible
% boundary conditions. This assumes there was surely a change point right before
% the first data point not at the first data point.
% Implements step 1, alg 1, of [RPA].
% => P(runglenth_0 = 0|nothing) = 1
R(1, 1) = 1; % 1 x 1. [P]

Z = zeros(T, 1); % The evidence at each time step => Z(t) = P(X_t|X_1:t-1). [P]
% Find parameters of p(X_1|nothing). 1 x param_count. units depend on model_f.

% initialize the posterior with the prior.
if exist([model_f, '_init'])
  post_params = feval(model_init_f, T + 1, D, theta_m);
else
  post_params = theta_m;
end

for t = 1:T
  % Implictly Implements step 2, alg 1, of [RPA]: oberserve new datum, simply by
  % incrementing the loop index.

  % Evaluate the predictive distribution for the new datum under each of
  % the parameters.  This is the standard thing from Bayesian inference.
  % Implements step 3, alg 1, of [RPA].
  % predprobs(r) = p(X(t)|X(1:t-1), runlength_t-1 = r-1). t x 1. [P]
  predprobs = feval(model_predict_f, post_params, X(t, :));

  % Evaluate the hazard function for this interval.
  % H(r) = P(runlength_t = 0|runlength_t-1 = r-1)
  % Pre-computed the hazard in preperation for steps 4 & 5, alg 1, of [RPA]
  H = feval(hazard_f, (1:t)', theta_h); % t x 1. [P]

  % Evaluate the growth probabilities - shift the probabilities up and to
  % the right, scaled by the hazard function and the predictive
  % probabilities.
  % Implements step 4, alg 1, of [RPA].
  % Assigning P(runlength_t = 1|X_1:t) to P(runlength_t = t|X_1:t):
  % P(runlength_t = r|X_1:t) propto P(runlength_t-1 = r-1|X_1:t-1) *
  % p(X_t|X_1:t-1,runlength_t-1 = r-1) * P(runlength_t = r|runlength_t-1 = r-1).
  R(2:t + 1, t + 1) = R(1:t, t) .* predprobs .* (1 - H); % t x 1. [P]

  % Evaluate the probability that there *was* a changepoint and we're
  % accumulating the mass back down at r = 0.
  % Implements step 5, alg 1, of [RPA].
  % Assigning P(runlength_t = 0|X_1:t)
  % P(runlength_t = 0|X_1:t) propto sum_r=0^t-1 P(runlength_t-1 = r|X_1:t-1) *
  % p(X_t|X_1:t-1, runlength_t-1 = r) * P(runlength_t = 0|runlength_t-1 = r).
  R(1, t + 1) = sum(R(1:t, t) .* predprobs .* H); % 1 x 1. [P]

  % Renormalize the run length probabilities for improved numerical
  % stability.
  % note that unlike in [RPA] which keeps track of P(r_t, X_1:t), we keep track
  % of P(r_t|X_1:t) => unnormalized R(i, t+1) = P(runlength_t = i-1|X_1:t) *
  % P(X_t|X_1:t-1) => normalization const Z(t) = P(X_t|X_1:t-1).
  % Sort of Implements step 6, alg 1, of [RPA].
  % Could sum R(:, t+1), but we only sum R(1:t+1,t+1) to avoid wasting time
  % summing zeros.
  Z(t) =  sum(R(1:t + 1, t + 1)); % 1 x 1. [P]
  % Implements step 7, alg 1, of [RPA].
  % After normalization, R(i, t+1) = P(runlength_t = i-1|X_1:t).
  R(1:t + 1, t + 1) = R(1:t + 1, t + 1) ./  Z(t); % (T + 1) x 1. [P]

  S(1:t, t) = R(1:t, t) .* predprobs;
  S(:, t) = S(:, t) ./ sum(S(:, t));

  % post_params(r, i) = current state of hyper-parameter i given runlength = r-1
  % Implements step 8, alg 1, of [RPA].
  % (t + 1) x param_count. units depend on model_f.
  post_params = feval(model_update_f, theta_m, post_params, X(t, :));

  % Could implement step 9, alg 1, of [RPA] is already taken care of by Z(t).
end

% Get the log marginal likelihood of the data, X(1:end), under the model
% = P(X_1:T), integrating out all the runlengths. 1 x 1. [log P]
nlml = -sum(log(Z));
