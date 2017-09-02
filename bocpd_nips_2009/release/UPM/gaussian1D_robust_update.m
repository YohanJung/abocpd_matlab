
% TODO document interface to this function
function [post_params_new, dpost_params] = ...
  gaussian1D_robust_update(theta_prior, post_params, xt, dpost_params)

% Valid calls:
% post_params = gaussian1D_update(theta_prior, post_params, xt)
% or
% [post_params, dpost_params] =
% gaussian1D_update(theta_prior, post_params, xt, dpost_params)

global isJunk;

mus = post_params(:, 1); % N x 1. [x]
kappas = post_params(:, 2); % N x 1. [points]
alphas = post_params(:, 3); % N x 1. [points]
betas = post_params(:, 4); % N x 1. [x^2]

% Posterior update rules found in p. 29 of
% http://www.stat.columbia.edu/~cook/movabletype/mlm/CONJINTRnew%2BTEX.pdf
% except their betas are scale and we use inverse scale. Set n = 1 to attain
% the online update rules. Note that SS = 0 when n = 1.
% TODO rename kappas taus for consistency with document.

% p(mean X|X) = N(mus, kappas), mus = mean, kappas = precision on E[X]
% Bayesian update on mus when n = 1. N + 1 x 1. [x]
mus_new = [theta_prior(1); (kappas .* mus + xt) ./ (kappas + 1)];
kappas_new = [theta_prior(2); kappas + 1]; % N + 1 x 1. [points]
% p(precision X|X) = Gamma(alphas, betas), alphas = shape, betas = inv scale
% => E[precision X] = alpha / beta => E[var X] = beta / alpha.
alphas_new = [theta_prior(3); alphas + 0.5]; % N + 1 x 1. [points]
% N + 1 x 1. [x^2]
betas_new = [theta_prior(4); ...
  betas + (kappas .* (xt - mus) .^ 2) ./ (2 * (kappas + 1))];

post_params_new = [mus_new, kappas_new, alphas_new, betas_new]; % N + 1 x 4. [mixed]

% throw out updates for data points deemed junk by predict
tmp = [zeros(1, 4); post_params];
post_params_new([false; isJunk], :) = tmp([false; isJunk], :);

if nargout == 1
  return;
end
% else: get the derivatives too

dmu_dmu0 = squeeze(dpost_params(1, 1, :)); % N x 1
dmu_dkappa0 = squeeze(dpost_params(1, 2, :)); % N x 1
dkappa_dkappa0 = squeeze(dpost_params(2, 2, :)); % N x 1
dbeta_dmu0 = squeeze(dpost_params(4, 1, :)); % N x 1
dbeta_dkappa0 = squeeze(dpost_params(4, 2, :)); % N x 1

% Update the components which need updating
dmu_dmu0_new = (kappas ./ (kappas + 1)) .* dmu_dmu0; % d_mu / d_mu0. N x 1
dmu_dkappa0_new = ... %d_mu / d_kappa0
  (dkappa_dkappa0 .* mus + dmu_dkappa0 .* kappas) ./ (kappas + 1) - ...
  ((kappas .* mus + xt) .* dkappa_dkappa0) ./ ((kappas + 1) .^ 2); % N x 1
dbeta_dmu0_new = dbeta_dmu0 - ...
  ((kappas .* (xt - mus)) ./ (kappas + 1) .* dmu_dmu0); % N x 1

den = 2 * (kappas + 1); % N x 1
num = kappas .* (xt - mus) .^ 2; % N x 1
dden_dkappa0 = 2 * dkappa_dkappa0; % N x 1
dnum_dkappa0 = dkappa_dkappa0 .* (xt - mus) .^ 2 + ...
  2 * kappas .* (mus - xt) .* dmu_dkappa0; % N x 1
QR = (dnum_dkappa0 .* den - dden_dkappa0 .* num) ./ den .^ 2; % N x 1
dbeta_dkappa0_new = dbeta_dkappa0 + QR; % N x 1

dpost_params(1, 1, :) = dmu_dmu0_new; % 1 x 1 x N
dpost_params(1, 2, :) = dmu_dkappa0_new; % 1 x 1 x N
dpost_params(4, 1, :) =  dbeta_dmu0_new; % 1 x 1 x N
dpost_params(4, 2, :) =  dbeta_dkappa0_new; % 1 x 1 x N

dpost_params(:, :, 2:size(dpost_params, 3) + 1) = dpost_params;  % 1 x 1 x N
dpost_params(:, :, 1) = eye(size(dpost_params, 1)); % 4 x 4 x 1
