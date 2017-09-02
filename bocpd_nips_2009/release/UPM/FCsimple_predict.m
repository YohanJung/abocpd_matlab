function [predprobs, dpredprobs] = FCsimple_predict(post_params, xnew, dpost_params)

% work with column vector observations here
xnew = xnew'; % D x 1
D = length(xnew); % 1 x 1

%Unpack sufficient statistics
ns = post_params(1, :); % 1 X T
kappas = post_params(2, :); % 1 x T
nus = post_params(3, :); % 1 x T
sums = post_params(4:D + 3, :); % D x T
outers = post_params(D + 4:end, :); % D^2 x T
T = size(post_params, 2); % 1 x 1

% TODO: vectorize this
predprobs = zeros(T, 1);
dpredprobs = zeros(T, 2);
for t = 1:T
  kappa = kappas(t); % 1 x 1
  nu = nus(t); % 1 x 1
  sum_t = sums(:, t); % D x 1
  % TODO assert the reshape if done correctly
  outer = reshape(outers(:, t), D, D); % D x D
  % Gelman p. 87, 88
  % Yunus use sums, Gelman use means
  mu = sum_t ./ kappa; % Simpler than Gelman's way. D x 1
  lambda_n = (eye(D)./2) + outer - ((sum_t * sum_t') ./ kappa); % D x D
  lambda = lambda_n .* ((kappa + 1) ./ (kappa * (nu - D + 1))); % D x D
  % Gelman p. 576
  [predprobs(t), dmu, dlambda, dnu] = multi_student(xnew, mu, lambda, nu - D + 1); % 1 x 1
  dlambda_dkappa = ((kappa + 1)/(kappa*(nu-D+1)) .* ((sum_t * sum_t') ./ (kappa*kappa))) - (lambda_n ./ (kappa*kappa*(nu-D+1))); 
  dkappa0 = sum(sum(dlambda .* dlambda_dkappa)) - (dmu' * (sum_t ./ (kappa*kappa)));
  dnu0 = dnu - sum(sum(dlambda .* (lambda./(nu-D+1))));
  dpredprobs(t , :) = [dkappa0, dnu0]; 
end
predprobs = exp(predprobs); % T x 1
dpredprobs = dpredprobs .* [predprobs, predprobs]; % convert from logspace
