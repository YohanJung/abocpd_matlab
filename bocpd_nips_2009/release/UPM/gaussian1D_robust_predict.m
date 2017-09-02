
% TODO document interface to this function
function [pred, dpred] = gaussian1D_robust_predict(post_params, xnew, dpost_params)

pJunk = 0.1;
extension = 10;
global isJunk;

% Valid calls:
% pred = gaussian1D_predict(post_params, xnew)
% or
% [pred, dpred] = gaussian1D_predict(post_params, xnew, dpost_params)

N = size(post_params, 1); % 1 x 1

mus = post_params(:, 1); % N x 1. [x]
kappas = post_params(:, 2); % N x 1. [points]
alphas = post_params(:, 3); % N x 1. [points]
betas = post_params(:, 4); % N x 1. [x^2]

% TODO verify this is correct by citing reference with posterior predictive
% However, probably correct since we get the same lml under random
% permutations of the data => coherence.
% N x 1. [x^2]
predictive_variance = betas .* (kappas + 1) ./ (alphas .* kappas);
df = 2 * alphas; % N x 1. [points]

isJunk = tcdf((xnew - mus) ./ sqrt(predictive_variance), df) <= pJunk;

if nargout == 1
  pred = studentpdf(xnew, mus, predictive_variance, df); % N x 1. [P/x]
  
  pred(isJunk) = pJunk / extension;
  % Warning: this asssumes that xnew is in [CI_lower - extension, CI_lower]
  % => all data falls inside extension
  
  % TODO assert in [-6,4]
  %pred = .5 * pred + .5 * (1/10); %studentpdf(xnew, 0, 30, 1);
  
  return;
end
% else: get the derivatives too

[pred, dtpdf] = studentpdf(xnew, mus, predictive_variance, df); % N x 1. [P/x]

dmu_dtheta = permute(dpost_params(1, :, :), [3 2 1]); % N x 4
dkappa_dtheta = permute(dpost_params(2, :, :), [3 2 1]); % N x 4
dalpha_dtheta = permute(dpost_params(3, :, :), [3 2 1]); % N x 4
dbeta_dtheta = permute(dpost_params(4, :, :), [3 2 1]); % N x 4

dnu_dtheta = 2 * dalpha_dtheta; % N x 4

% TODO use rmult and eliminate the for loop
dpv_dtheta = zeros(N, 4);
for ii = 1:4
  QRpart = (dbeta_dtheta(:, ii) .* alphas - betas .* dalpha_dtheta(:, ii)) ...
    ./ alphas .^ 2; % N x 1
  dpv_dtheta(:, ii) = -(betas ./ (alphas .* kappas .^ 2)) .* ...
    dkappa_dtheta(:, ii) + (1 + 1 ./ kappas) .* QRpart; % N x 1
end

% TODO use rmult and eliminate the for loop
dp_dtheta = zeros(N, 4);
for ii = 1:4
  % dp/dtheta_i = dp/dmu * dmu/dtheta_i + dp/dsigma2 * dsigma2/dtheta_i +
  % dp/dnu + dnu/dtheta_i
  % N x 1
  dp_dtheta(:, ii) = dtpdf(:, 1) .* dmu_dtheta(:, ii) + ...
    dtpdf(:, 2) .* dpv_dtheta(:, ii) + ...
    dtpdf(:, 3) .* dnu_dtheta(:, ii);
end
dpred = dp_dtheta;
