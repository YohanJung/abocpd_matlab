
% TODO document interface to this function
function [pred, dpred] = IFM_predict(post_params, xnew, dpost_params)

% Valid calls:
% pred = gaussian1D_predict(post_params, xnew)
% or
% [pred, dpred] = gaussian1D_predict(post_params, xnew, dpost_params)

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

predIF = zeros(N, D);
for ii = 1:D
  predIF(:, ii) = studentpdf(xnew(:, ii), mus(:, ii), predictive_variance(:, ii), df(:, ii)); % N x 1. [P/x]
end
pred = prod(predIF, 2);

if nargout == 1
  return;
end
% else: get the derivatives too

dp_dtheta = zeros(N, D, 4);
for ii = 1:D
  [tmp, dtpdf] = studentpdf(xnew(ii), mus(:, ii), predictive_variance(:, ii), df(:, ii)); % N x 1. [P/x]

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
