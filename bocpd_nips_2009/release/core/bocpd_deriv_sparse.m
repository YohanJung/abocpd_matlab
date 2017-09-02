% Ryan Turner (rt324@cam.ac.uk)
% Yunus Saatci (ys267@cam.ac.uk)
% Ryan Adams (rpa23@cam.ac.uk)
% Modularized version of fast Bayesian online change point detection (BOCP)
% adapted from: http://www.inference.phy.cam.ac.uk/rpa23/cp/gaussdemo.m
% Based on the paper:
% http://www.inference.phy.cam.ac.uk/rpa23/papers/rpa-changepoint.pdf
% paper refered to as [RPA].
%
% Inputs:
% X : T x 1 variable containing the time series.
% model_f : observation model e.g. 'gaussian1D' (this will call
% gaussian1D_update for update of posterior parameters, and
% gaussian1D_predict for evaluation of predictive densities)
% theta_m : 1 x model_param_count parameters of the priors on new distributions
% drawn at each time point.
% hazard_f : hazard model e.g. 'constant'
% theta_h : 1 x hazard_param_count parameters to the hazard function
%
% Outputs:
% nlml : negative log marginal likelihood of the data, X(1:end), under the model
% = -log(P(X_1:T)), integrating out all the runlengths. [log P]
% dnlml_h : derivative of the negative log marginal likelihood w.r.t the
% hazard (hyper)parameters.
% dnlml_m :  derivative of the negative log marginal likelihood w.r.t the
% observation model (hyper)parameters.
% Z : "online" evidence at each time step  Z(t) = P(X_t|X_1:t-1)
% dZ_h : derivative of the online evidence w.r.t hazard parameters.
% dZ_m : derivative of the online evidence w.r.t. observation model


% parameters.
% R : distribution over runlengths at each time step (of size T+1-by-T+1)
% dR_h : derivative of R w.r.t. hazard parameters
% dR_m : derivative of R w.r.t. observation model parameters
%
% Might want to implement pruning for efficiency.
% TODO specify required interface for model_f() and hazard_f()
function [nlml, dnlml_h, dnlml_m, Z, dZ_h, dZ_m, R] = ...
  bocpd_deriv_sparse(theta_h, theta_m, X, hazard_f, model_f, epsilon)

num_hazard_params = length(theta_h);
num_model_params = length(theta_m);

assert(isKosher(theta_m));
assert(isKosher(theta_h));
assert(isKosher(X));
% assert they are row vec
assert(all(size(theta_h) == [1 num_hazard_params]));
assert(all(size(theta_m) == [1 num_model_params]));

% assert that the function handles passed in are valid
if exist(hazard_f) == 0 || exist([model_f, '_update']) == 0 || ...
    exist([model_f, '_predict']) == 0 || exist([model_f, '_init']) == 0
  error('Required functions do not exist. See preamble.');
end

hazard_f = str2func(hazard_f);
model_update_f = str2func([model_f, '_update']);
model_predict_f = str2func([model_f, '_predict']);
init_f = str2func([model_f, '_init']);

[T D] = size(X); % Number of time point observed. 1 x 1. [s]


% R(r, t) = P(runlength_t-1 = r-1|X_1:t-1).
R = cell(T + 1, 1); % pre-load the run length distribution. [P]

% At time t = 1, we actually have complete knowledge about the run
% length.  It is definitely zero.  See the paper for other possible
% boundary conditions. This assumes there was surely a change point right before
% the first data point not at the first data point.
% Implements step 1, alg 1, of [RPA].
% => P(runglenth_0 = 0|nothing) = 1
R{1} = 1; % 1 x 1. [P]

Z = zeros(T, 1); % The evidence at each time step => Z(t) = P(X_t|X_1:t-1). [P]
dZ_h = zeros(T, num_hazard_params);
dZ_m = zeros(T, num_model_params);

% Find parameters of p(X_1|nothing). 1 x param_count. units depend on model_f.
[post_params, dmessage] = feval(init_f, 1, D, theta_m);

maxRunConsidered = 1;
% last = zeros on first iteration is equivalent to full matrix version
% TODO pass these back out for chackgrad to check
dR_h_last = zeros(maxRunConsidered, num_hazard_params);
dR_m_last = zeros(maxRunConsidered, num_model_params);
dR_h_curr = zeros(maxRunConsidered + 1, num_hazard_params);
dR_m_curr = zeros(maxRunConsidered + 1, num_model_params);
for t = 1:T
  Rnew = zeros(maxRunConsidered + 1, 1);
  % Implictly Implements step 2, alg 1, of [RPA]: oberserve new datum, simply by
  % incrementing the loop index.

  % Evaluate the predictive distribution for the new datum under each of
  % the parameters.  This is the standard thing from Bayesian inference.
  % Implements step 3, alg 1, of [RPA].
  % predprobs(r) = p(X(t)|X(1:t-1), runlength_t-1 = r-1). MRC x 1. [P]
  [predprobs, dpredprobs] = feval(model_predict_f, post_params, X(t, :), dmessage);

  % Evaluate the hazard function for this interval.
  % H(r) = P(runlength_t = 0|runlength_t-1 = r-1)
  % Pre-computed the hazard in preperation for steps 4 & 5, alg 1, of [RPA]
  [H, dH] = feval(hazard_f, (1:maxRunConsidered)', theta_h); % MRC x 1. [P]

  % Evaluate the growth probabilities - shift the probabilities up and to
  % the right, scaled by the hazard function and the predictive
  % probabilities.
  % Implements step 4, alg 1, of [RPA].
  % Assigning P(runlength_t = 1|X_1:t) to P(runlength_t = t|X_1:t):
  % P(runlength_t = r|X_1:t) propto P(runlength_t-1 = r-1|X_1:t-1) *
  % p(X_t|X_1:t-1,runlength_t-1 = r-1) * P(runlength_t = r|runlength_t-1 = r-1).
  Rnew(2:end) = R{t} .* predprobs .* (1 - H); % MRC x 1. [P]
  % derivatives wrt to hazard
  % predprobs .* (dR_h_last(:, ii) .* (1 - H) - R(1:MRC, t) .* dH(:, ii));
  PRpart = rmult(dR_h_last, 1 - H) - rmult(dH, R{t});
  dR_h_curr(2:end, :) = rmult(PRpart, predprobs);
  % derivatives wrt to model
  % (1 - H) .* {(dR_m_last(:, ii) .* predprobs + R(1:MRC, t) .* dpredprobs(:, ii))};
  PRpart = rmult(dR_m_last, predprobs) + rmult(dpredprobs, R{t});
  dR_m_curr(2:end, :) = rmult(PRpart, 1 - H);

  % Evaluate the probability that there *was* a changepoint and we're
  % accumulating the mass back down at r = 0.
  % Implements step 5, alg 1, of [RPA].
  % Assigning P(runlength_t = 0|X_1:t)
  
  % P(runlength_t = 0|X_1:t) propto sum_r=0^t-1 P(runlength_t-1 = r|X_1:t-1) *
  % p(X_t|X_1:t-1, runlength_t-1 = r) * P(runlength_t = 0|runlength_t-1 = r).
  Rnew(1) = sum(R{t} .* predprobs .* H); % 1 x 1. [P]
  % derivatives wrt to hazard
  % sum(predprobs .* (dR_h_last(:, ii) .* H + R(1:MRC, t) .* dH(:, ii)), 1);
  PRpart = rmult(dR_h_last, H) + rmult(dH, R{t});
  dR_h_curr(1, :) = sum(rmult(PRpart, predprobs), 1);
  % derivatives wrt to model
  % sum(H .* (dR_m_last(:, ii) .* predprobs + R(1:MRC, t) .* dpredprobs(:, ii)));
  PRpart = rmult(dR_m_last, predprobs) + rmult(dpredprobs, R{t});
  dR_m_curr(1, :) = sum(rmult(PRpart, H), 1);

  % Renormalize the run length probabilities for improved numerical
  % stability.
  % note that unlike in [RPA] which keeps track of P(r_t, X_1:t), we keep track
  % of P(r_t|X_1:t) => unnormalized R(i, t+1) = P(runlength_t = i-1|X_1:t) *
  % P(X_t|X_1:t-1) => normalization const Z(t) = P(X_t|X_1:t-1).
  % Sort of Implements step 6, alg 1, of [RPA].
  % Could sum R(:, t+1), but we only sum R(1:t+1,t+1) to avoid wasting time
  % summing zeros.
  Z(t) =  sum(Rnew); % 1 x 1. [P]
  dZ_h(t, :) = sum(dR_h_curr, 1);
  dZ_m(t, :) = sum(dR_m_curr, 1);

  % Implements step 7, alg 1, of [RPA].
  % After normalization, R(i, t+1) = P(runlength_t = i-1|X_1:t).
  % (dR_h_curr(:, ii) / Z(t)) - (dZ_h(t, ii) * Rnew) / (Z(t) ^ 2);
  dR_h_curr = dR_h_curr / Z(t) - Rnew * dZ_h(t, :) / (Z(t) ^ 2);
  % (dR_m_curr(:, ii) / Z(t)) - (dZ_m(t, ii) * Rnew) / (Z(t) ^ 2);
  dR_m_curr = dR_m_curr / Z(t) - Rnew * dZ_m(t, :) / (Z(t) ^ 2);
  Rnew = Rnew ./  Z(t); % MRC x 1. [P]

  % Do pruning
  [Rpruned, renorm] = pruneR(Rnew, epsilon); % [new MRC x 1, 1 x 1]. [P] [P]
  maxRunConsidered = length(Rpruned); % 1 x 1. [points]
  R{t + 1} = Rpruned; % MRC x 1. [P]
  % MRC x hazard_params. [P/[hazard_params]]
  dR_h_curr = dR_h_curr(1:maxRunConsidered, :);
  % MRC x model_params. [P/[model_params]]
  dR_m_curr = dR_m_curr(1:maxRunConsidered, :);

  % Compensate for renormalization in derivatives
  % TODO combine this quotient rule correction with the normalization with Z
  % dR_h_curr(:, ii) = (dR_h_curr(:, ii) ./ renorm) - ...
  % (sum(dR_h_curr(1:maxRunConsidered, ii)) .* Rnew(1:maxRunConsidered))
  % ./ (renorm ^ 2);
  dR_h_curr = dR_h_curr / renorm - ...
    Rnew(1:maxRunConsidered) * sum(dR_h_curr, 1) / (renorm ^ 2);
  % dR_m_curr(:, ii) = (dR_m_curr(:, ii) ./ renorm) - ...
  % (sum(dR_m_curr(1:maxRunConsidered, ii)) .* Rnew(1:maxRunConsidered))
  % ./ (renorm ^ 2);
  dR_m_curr = dR_m_curr / renorm - ...
    Rnew(1:maxRunConsidered) * sum(dR_m_curr, 1) / (renorm ^ 2);

  % post_params(r, i) = current state of hyper-parameter i given runlength = r-1
  % Implements step 8, alg 1, of [RPA].
  % (t + 1) x param_count. units depend on model_f.
  [post_params, dmessage] = feval( ...
    model_update_f, theta_m, post_params, X(t, :), dmessage, maxRunConsidered);

  % remember: loop invariant: number of nnz rows of R(t + 1, :) = number of rows
  % in post_params and dmessage
  % TODO assert this

  % assert sum of run length derivatives = 0
  % assert(all(abs(sum(dR_m_curr, 1)) <= 1e-9));
  % assert(all(abs(sum(dR_h_curr, 1)) <= 1e-9));

  % Could implement step 9, alg 1, of [RPA] is already taken care of by Z(t).

  % MRC x hazard_params. [P/[hazard_params]]
  dR_h_last = dR_h_curr;
  % MRC x hazard_params. [P/[model_params]]
  dR_m_last = dR_m_curr;
  % probably will work without this but good to be safe
  dR_h_curr = zeros(maxRunConsidered + 1, num_hazard_params);
  dR_m_curr = zeros(maxRunConsidered + 1, num_model_params);
end

% Get the log marginal likelihood of the data, X(1:end), under the model
% = P(X_1:T), integrating out all the runlengths. 1 x 1. [log P]
nlml = -sum(log(Z));
dnlml_h = zeros(num_hazard_params, 1);
dnlml_m = zeros(num_model_params, 1);

for ii = 1:num_hazard_params
  % 1 x 1. [log P/[units hazard]]
  dnlml_h(ii) = -sum(dZ_h(:, ii) ./ Z);
end

for ii = 1:num_model_params
  % 1 x 1. [log P/[units model]]
  dnlml_m(ii) = -sum(dZ_m(:, ii) ./ Z);
end

assert(isKosher(nlml));
assert(isKosher(dnlml_h));
assert(isKosher(dnlml_m));

% TODO write convert R from cell to matrix

function [Rpruned, renorm] = pruneR(R, epsilon)

assert(approxeq(sum(R), 1, 1e-8, false));

prunes = find(R >= epsilon); % * x 1. []
assert(~isempty(prunes));
Rpruned = R(1:prunes(end)); % new MRC x 1. [unnormalized P]
renorm = sum(Rpruned); % 1 x 1, [Z]
Rpruned = Rpruned ./ renorm; % MRC x 1. [P]
