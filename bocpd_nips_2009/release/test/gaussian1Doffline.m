

function [post_params, dparams] = gaussian1Doffline(theta_prior, X)

T = length(X);

post_params = zeros(T + 1, 4);
dparams = zeros(4, 4, T + 1);

mu_prior = theta_prior(1);
kappa_prior = theta_prior(2);
alpha_prior = theta_prior(3);
beta_prior = theta_prior(4);

for ii = 1:(T+1)
  timePoints = ii - 1;
  % => when timePoints = 0, selects (T+1):T = [], = 1 then T:T => 1 point.
  Xcurrent = X((T + 1 - timePoints):T);
  meanX = mean(Xcurrent);
  SS = sum((Xcurrent - meanX) .^ 2);

  mu = (kappa_prior * mu_prior + timePoints * meanX) / (kappa_prior + timePoints);
  kappa = kappa_prior + timePoints;
  alpha = alpha_prior + timePoints / 2;
  beta = beta_prior + SS / 2 + (kappa_prior * timePoints * (meanX - mu_prior) .^ 2) / (2 * (kappa_prior + timePoints));
  post_params(ii, :) = [mu kappa alpha beta];

  % mu derivatives
  dmu_dmu = kappa_prior / (kappa_prior + timePoints);
  dmu_dkappa = timePoints * (mu_prior - meanX) / (kappa_prior + timePoints) .^ 2;

  % kappa derivatives
  dkappa_dkappa = 1;

  % alpha derivative
  dalpha_dalpha = 1;

  % beta derivatives
  dbeta_dmu = (kappa_prior * timePoints * (mu_prior - meanX)) / (kappa_prior + timePoints);
  dbeta_dkappa = .5 * ((timePoints * (meanX - mu_prior)) / (kappa_prior + timePoints)) .^ 2;
  dbeta_dbeta = 1;

  % assign them all in place
  dparams(1, 1, ii) = dmu_dmu;
  dparams(1, 2, ii) = dmu_dkappa;
  dparams(2, 2, ii) = dkappa_dkappa;
  dparams(3, 3, ii) = dalpha_dalpha;
  dparams(4, 1, ii) = dbeta_dmu;
  dparams(4, 2, ii) = dbeta_dkappa;
  dparams(4, 4, ii) = dbeta_dbeta;
end
