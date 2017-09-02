
function [X changePoints] = IFMrnd(T, D, model_params, hazard_params)

changePoints = false(T, 1);
X = zeros(T, D);

assert(all(size(model_params) == [1 4 * D]));

if isscalar(hazard_params)
  assert(hazard_params >= 0);
  assert(hazard_params <= 1);
  % Turns logistic into constant hazard
  hazard_params = [hazard_params 0 Inf];
end
% else TODO assert proper hazard params for logistic_h

mu_prior = model_params(1:D); % 1 x D. [x]
kappa_prior = model_params((D + 1):2 * D); % 1 x D. [points]
alpha_prior = model_params((2 * D + 1):3 * D); % 1 x D. [points]
beta_prior = model_params((3 * D + 1):4 * D); % 1 x D. [x^2]

assert(all(kappa_prior > 0));
assert(all(alpha_prior > 0));
assert(all(beta_prior > 0));

runlength = 0;
for t = 1:T
  % TODO double check for off by one errors here
  hazard_rate = logistic_h(runlength, hazard_params);
  if rand() < hazard_rate || t == 1
    % Generate new Gaussian parameters from the prior.
    curr_prec = gamrnd(alpha_prior, 1 ./ beta_prior);
    std_mean = 1 ./ sqrt(kappa_prior .* curr_prec);
    curr_mean = normrnd(mu_prior, std_mean);

    changePoints(t) = true;
    runlength = 0;
  end

  X(t, :) = normrnd(curr_mean, 1 ./ sqrt(curr_prec));
  inc runlength;
end
