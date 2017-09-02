
% TODO document interface to this function

% dpost_params(i,j,k,l) = d(parameter k on channel l at runlength
% i)/d(hyper-parameter j). A more natural representation would have been
% d(param i on chan l at run k)/d(hyper j) but this requires squeezing and
% permuting slowed things down
function [post_params, dpost_params] = ...
  IFMfast_update(theta_prior, post_params, xt, dpost_params, maxRunConsidered)

% Valid calls:
% post_params = gaussian1D_update(theta_prior, post_params, xt)
% or
% [post_params, dpost_params] =
% gaussian1D_update(theta_prior, post_params, xt, dpost_params)

% If seperating out post_params and theta_prior get much more complicated we'll
% need to move to using unpackFromVector (because of x N replication we'll need
% to make a function unpackFromMatrix).

% suff stats version:
% D = length(xt);
% 
% xsum = [zeros(1, D); cplus(xsum, xt)];
% xsum2 = [zeros(1, D); cplus(xsum2, xt .^ 2)];
% countn = [zeros(1, D); countn + 1];

N = size(post_params, 1);
D = length(xt);
num_params = size(post_params, 2);
assert(num_params == 4 * D);

% Organize these in a 3D matrix so we don't need to do this crap??
mus = post_params(:, 1:D); % N x D. [x]
kappas = post_params(:, (D + 1):2 * D); % N x D. [points]
alphas = post_params(:, (2 * D + 1):3 * D); % N x D. [points]
betas = post_params(:, (3 * D + 1):4 * D); % N x D. [x^2]

% Posterior update rules found in p. 29 of
% http://www.stat.columbia.edu/~cook/movabletype/mlm/CONJINTRnew%2BTEX.pdf
% except their betas are scale and we use inverse scale. Set n = 1 to attain
% the online update rules. Note that SS = 0 when n = 1.
% TODO rename kappas taus for consistency with document.

% TODO organize these in a matrix so we don't need to do this crap
mu_prior = theta_prior(1:D); % 1 x D. [x]
kappa_prior = theta_prior((D + 1):2 * D); % 1 x D. [points]
alpha_prior = theta_prior((2 * D + 1):3 * D); % 1 x D. [points]
beta_prior = theta_prior((3 * D + 1):4 * D); % 1 x D. [x^2]

mus_new = zeros(N + 1, D);
kappas_new = zeros(N + 1, D);
alphas_new = zeros(N + 1, D);
betas_new = zeros(N + 1, D);
% Could probably do this loop with rplus (bsxfun) instead.
for ii = 1:D
  % TODO pre-allocate post_params so it doesn't need to grow every iteration.
  % If we permute updates: beta, mu, kappa, alpha then we can work in place and
  % not need to create mus_new ect.
  %
  % p(mean X|X) = N(mus, kappas), mus = mean, kappas = precision on E[X]
  % Bayesian update on mus when n = 1. N + 1 x 1. [x]
  mus_new(:, ii) = [mu_prior(ii); (kappas(:, ii) .* mus(:, ii) + xt(ii)) ./ (kappas(:, ii) + 1)];
  kappas_new(:, ii) = [kappa_prior(ii); kappas(:, ii) + 1]; % N + 1 x 1. [points]
  % p(precision X|X) = Gamma(alphas, betas), alphas = shape, betas = inv scale
  % => E[precision X] = alpha / beta => E[var X] = beta / alpha.
  alphas_new(:, ii) = [alpha_prior(ii); alphas(:, ii) + 0.5]; % N + 1 x 1. [points]
  % N + 1 x 1. [x^2]
  betas_new(:, ii) = [beta_prior(ii); ...
    betas(:, ii) + (kappas(:, ii) .* (xt(ii) - mus(:, ii)) .^ 2) ./ (2 * (kappas(:, ii) + 1))];
end

post_params = [mus_new, kappas_new, alphas_new, betas_new]; % N + 1 x 4. [mixed]

if exist('maxRunConsidered', 'var')
post_params = post_params(1:maxRunConsidered, :);
end

if nargout == 1
  return;
end
% else: get the derivatives too

% replace this loop with sub-routine??
for ii = 1:D
  % TODO permute the indices so we don't need the squeeze command
  % this slows down the program
  dmu_dmu0 = dpost_params(1:N, 1, 1, ii); % N x 1
  dmu_dkappa0 = dpost_params(1:N, 2, 1, ii); % N x 1
  dkappa_dkappa0 = dpost_params(1:N, 2, 2, ii); % N x 1
  dbeta_dmu0 = dpost_params(1:N, 1, 4, ii); % N x 1
  dbeta_dkappa0 = dpost_params(1:N, 2, 4, ii); % N x 1

  % Update the components which need updating
  dmu_dmu0_new = (kappas(:, ii) ./ (kappas(:, ii) + 1)) .* dmu_dmu0; % d_mu / d_mu0. N x 1
  dmu_dkappa0_new = ... %d_mu / d_kappa0
    (dkappa_dkappa0 .* mus(:, ii) + dmu_dkappa0 .* kappas(:, ii)) ./ (kappas(:, ii) + 1) - ...
    ((kappas(:, ii) .* mus(:, ii) + xt(ii)) .* dkappa_dkappa0) ./ ((kappas(:, ii) + 1) .^ 2); % N x 1
  dbeta_dmu0_new = dbeta_dmu0 - ...
    ((kappas(:, ii) .* (xt(ii) - mus(:, ii))) ./ (kappas(:, ii) + 1) .* dmu_dmu0); % N x 1

  den = 2 * (kappas(:, ii) + 1); % N x 1
  num = kappas(:, ii) .* (xt(ii) - mus(:, ii)) .^ 2; % N x 1
  dden_dkappa0 = 2 * dkappa_dkappa0; % N x 1
  dnum_dkappa0 = dkappa_dkappa0 .* (xt(ii) - mus(:, ii)) .^ 2 + ...
    2 * kappas(:, ii) .* (mus(:, ii) - xt(ii)) .* dmu_dkappa0; % N x 1
  QR = (dnum_dkappa0 .* den - dden_dkappa0 .* num) ./ den .^ 2; % N x 1
  dbeta_dkappa0_new = dbeta_dkappa0 + QR; % N x 1

  dpost_params(2:N + 1, 1, 1, ii) = dmu_dmu0_new; % N x 1 x 1 x 1
  dpost_params(2:N + 1, 2, 1, ii) = dmu_dkappa0_new; % N x 1 x 1 x 1
  dpost_params(2:N + 1, 1, 4, ii) =  dbeta_dmu0_new; % N x 1 x 1 x 1
  dpost_params(2:N + 1, 2, 4, ii) =  dbeta_dkappa0_new; % N x 1 x 1 x 1
  
  % These lines should not make a difference if dpost_params is pre-loaded with
  % identity matrix. However, we need to add these terms if pre-loaded with
  % zeros or not pre-load at all.
  dpost_params(2:N + 1, 2, 2, ii) = ones(N, 1);
  dpost_params(2:N + 1, 3, 3, ii) = ones(N, 1);
  dpost_params(2:N + 1, 4, 4, ii) = ones(N, 1);
end

for ii = 1:D
  dpost_params(1, :, :, ii) = eye(size(dpost_params, 2)); % 1 x 4 x 4 x 1
end

if exist('maxRunConsidered', 'var')
  % This will defeat the purpose of pre-loaded, but this line is only used when
  % pruning and you don't pre-load then anyway.
dpost_params = dpost_params(1:maxRunConsidered, :, :, :);
end
