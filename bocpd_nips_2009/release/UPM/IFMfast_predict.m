
% TODO document interface to this function
function [pred, dpred] = IFMfast_predict(post_params, xnew, dpost_params)

% Valid calls:
% pred = gaussian1D_predict(post_params, xnew)
% or
% [pred, dpred] = gaussian1D_predict(post_params, xnew, dpost_params)

% Version to convert to suff statistics
%{
N = size(post_params, 1);
D = length(xnew);
assert(size(post_params, 2) == 4 * D);

meanX = xsum ./ countn; % N x D
meanX2 = xsum2 ./ coutn; % N x D
SS = countn .* (meanX2 - meanX .^ 2); % N x D

mus = cplus(countn .* meanX, kappa_prior .* mu_prior) ./ cplus(countn, kappa_prior);  % N x D
kappas = cplus(countn, kappa_prior); % N x D
alphas = cplus(countn / 2, alpha_prior); % N x D

beta_den = 2 * cplus(countn, kappa_prior); % N x D
sq_error = countn .* cplus(meanX, -mu_prior) .^ 2; % N x D
beta_num = cmult(sq_error, kappa_prior); % N x D
betas = cplus(SS / 2, beta_prior) + beta_num ./ beta_den; % N x D

% The following operations should also work for the matrix case because they are
% all element-wise.
predictive_variance = betas .* (kappas + 1) ./ (alphas .* kappas); % N x D
df = 2 * alphas; % N x D. [points]

predIF = studentpdffast(xnew, mus, predictive_variance, df); % N x D. [P/x]
pred = prod(predIF, 2);

if nargout == 1
  return;
end
% else: get the derivatives too

dp_dtheta = zeros(N, D, 4);

[tmp, dtpdf] = studentpdffast(xnew, mus, predictive_variance, df); % N x 1. [P/x]

for ii = 1:D
% mu derivatives
dmu_dmu = kappa_prior(ii) ./ (kappa_prior(ii) + countn(:, ii));
dmu_dkappa = countn(:, ii) .* (mu_prior(ii) - meanX(:, ii)) ./ (kappa_prior(ii) + countn(:, ii)) .^ 2;

% kappa derivatives
% dkappa_dkappa = 1;

% alpha derivative
% dalpha_dalpha = 1;

% beta derivatives
dbeta_dmu = (kappa_prior(ii) .* countn(:, ii) .* (mu_prior(ii) - meanX(:, ii))) ./ (kappa_prior(ii) + countn(:, ii));
dbeta_dkappa = .5 * ((countn(:, ii) * (meanX(:, ii) - mu_prior(ii))) / (kappa_prior(ii) + countn(:, ii))) .^ 2;
% dbeta_dbeta = 1;

dmu_dtheta = [dmu_dmu dmu_dkappa zeros(N, 2)]; % N x 4
dkappa_dtheta = [zeros(N, 1) ones(N, 1) zeros(N, 2)]; % N x 4
dalpha_dtheta = [zeros(N, 2) ones(N, 1) zeros(N, 1)]; % N x 4
dbeta_dtheta = [dbeta_dmu dbeta_dkappa zeros(N, 1) dbeta_dbeta]; % N x 4

dnu_dtheta = 2 * dalpha_dtheta; % N x 4

% TODO use rmult and eliminate the for loop
dpv_dtheta = zeros(N, 4);
for jj = 1:4
  QRpart = (dbeta_dtheta(:, jj) .* alphas(:, ii) - betas(:, ii) .* dalpha_dtheta(:, jj)) ...
    ./ alphas(:, ii) .^ 2; % N x 1
  dpv_dtheta(:, jj) = -(betas(:, ii) ./ (alphas(:, ii) .* kappas(:, ii) .^ 2)) .* ...
    dkappa_dtheta(:, jj) + (1 + 1 ./ kappas(:, ii)) .* QRpart; % N x 1
end

% TODO use rmult and eliminate the for loop
for jj = 1:4
  % dp/dtheta_i = dp/dmu * dmu/dtheta_i + dp/dsigma2 * dsigma2/dtheta_i +
  % dp/dnu + dnu/dtheta_i
  % N x 1
  dp_dtheta(:, ii, jj) = dtpdf(:, 1) .* dmu_dtheta(:, jj) + ...
    dtpdf(:, 2) .* dpv_dtheta(:, jj) + ...
    dtpdf(:, 3) .* dnu_dtheta(:, jj);
  dp_dtheta(:, ii, jj) = dp_dtheta(:, ii, jj) .* (pred ./ predIF(:, ii));
end
end

dpred = zeros(N, 4 * D);
dpred(:, 1:D) = dp_dtheta(:, :, 1);
dpred(:, D + 1:2 * D) = dp_dtheta(:, :, 2);
dpred(:, 2 * D + 1:3 * D) = dp_dtheta(:, :, 3);
dpred(:, 3 * D + 1:4 * D) = dp_dtheta(:, :, 4);
%}

N = size(post_params, 1);
D = length(xnew);
assert(size(post_params, 2) == 4 * D);

mus = post_params(:, 1:D); % N x D. [x]
kappas = post_params(:, (D + 1):2 * D); % N x D. [points]
alphas = post_params(:, (2 * D + 1):3 * D); % N x D. [points]
betas = post_params(:, (3 * D + 1):4 * D); % N x D. [x^2]

% The following operations should also work for the matrix case because they are
% all element-wise.
% N x 1. [x^2]
predictive_variance = betas .* (kappas + 1) ./ (alphas .* kappas);
df = 2 * alphas; % N x 1. [points]

predIF = studentpdffast(xnew, mus, predictive_variance, df); % N x 1. [P/x]
pred = prod(predIF, 2);

if nargout == 1
  return;
end
% else: get the derivatives too

[tmp, dtpdf] = studentpdffast(xnew, mus, predictive_variance, df); % N x 1. [P/x]
dp_dtheta = zeros(N, D, 4);
for ii = 1:D
  dmu_dtheta = dpost_params(1:N, :, 1, ii); % N x 4
  dkappa_dtheta = dpost_params(1:N, :, 2, ii); % N x 4
  dalpha_dtheta = dpost_params(1:N, :, 3, ii); % N x 4
  dbeta_dtheta = dpost_params(1:N, :, 4, ii); % N x 4

  dnu_dtheta = 2 * dalpha_dtheta; % N x 4

  % TODO use rmult and eliminate the for loop
  dpv_dtheta = zeros(N, 4);
  for jj = 1:4
    QRpart = (dbeta_dtheta(:, jj) .* alphas(:, ii) - betas(:, ii) .* dalpha_dtheta(:, jj)) ...
      ./ alphas(:, ii) .^ 2; % N x 1
    dpv_dtheta(:, jj) = -(betas(:, ii) ./ (alphas(:, ii) .* kappas(:, ii) .^ 2)) .* ...
      dkappa_dtheta(:, jj) + (1 + 1 ./ kappas(:, ii)) .* QRpart; % N x 1
  end

  % TODO use rmult and eliminate the for loop
  for jj = 1:4
    % dp/dtheta_i = dp/dmu * dmu/dtheta_i + dp/dsigma2 * dsigma2/dtheta_i +
    % dp/dnu + dnu/dtheta_i
    % N x 1
    dp_dtheta(:, ii, jj) = dtpdf(:, ii, 1) .* dmu_dtheta(:, jj) + ...
      dtpdf(:, ii, 2) .* dpv_dtheta(:, jj) + ...
      dtpdf(:, ii, 3) .* dnu_dtheta(:, jj);
    dp_dtheta(:, ii, jj) = dp_dtheta(:, ii, jj) .* (pred ./ predIF(:, ii));
  end
end

dpred = zeros(N, 4 * D);
dpred(:, 1:D) = dp_dtheta(:, :, 1);
dpred(:, D + 1:2 * D) = dp_dtheta(:, :, 2);
dpred(:, 2 * D + 1:3 * D) = dp_dtheta(:, :, 3);
dpred(:, 3 * D + 1:4 * D) = dp_dtheta(:, :, 4);

function [p, dp] = studentpdffast(x, mu, var, nu)
%
% p = studentpdf(x, mu, var, nu)
%
% Can be made equivalent to MATLAB's tpdf() by:
% tpdf((y - mu) ./ sqrt(var), nu) ./ sqrt(var)
% Equations found in p. 577 of Gelman

% TODO assert size consistency
global Gcache;
global Pcache;

D = length(x);

c = Gcache(1:size(nu, 1), :) .* (nu .* pi .* var) .^ (-0.5); % N x D
%ctest = exp(gammaln(nu / 2 + 0.5) - gammaln(nu / 2)) .* (nu .* pi .* var) .^ (-0.5);
%assert(max2(abs(ctest-c)) <= 1e-12);
% This form is taken from Kevin Murphy's lecture notes.
sq_error = cplus(mu, -x) .^ 2; % N x D
p = c .* (1 + (1 ./ (nu .* var)) .* sq_error) .^ (-(nu + 1) / 2); % N x D

if nargout == 2
  N = size(mu, 1);
  dp = zeros(N, D, 3);

  error = -cplus(mu, -x) ./ sqrt(var);
  sq_error = cplus(mu, -x) .^ 2 ./ var;

  % derivative for mu
  dlogp = (1 ./ sqrt(var)) .* ((nu + 1) .* error) ./ (nu + sq_error);
  dp(:, :, 1) = p .* dlogp;

  % derivative for var
  dlogpdprec = sqrt(var) - ((nu + 1) .* -cplus(mu, -x) .* error) ./ (nu + sq_error);
  dp(:, :, 2) = - .5 * (p ./ var .^ 1.5) .* dlogpdprec;

  % derivative for nu (df)
  dlogp = Pcache(1:size(nu, 1), :) - log(1 + (1 ./ nu) .* sq_error) + ((nu + 1) .* sq_error) ./ (nu .^ 2 + nu .* sq_error);
  %dlogptest = psi(nu / 2 + .5) - psi(nu / 2) - (1 ./ nu) - log(1 + (1 ./ nu) .* sq_error) + ((nu + 1) .* sq_error) ./ (nu .^ 2 + nu .* sq_error);
  %assert(max2(abs(dlogptest-dlogp)) <= 1e-12);

  dp(:, :, 3) = .5 * p .* dlogp;
end
