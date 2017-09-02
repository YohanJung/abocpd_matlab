

function [hazard_params, model_params, nlml] = learn_bocpd(X, useLogistic)

max_minimize_iter = 30;

if nargin == 2 && useLogistic
  model_f = 'gaussian1D';
  hazard_f = 'logistic_h';
  num_hazard_params = 3;

  hazard_init = [logit(.01) 0 0]';
  model_init = [0 log(.1) log(.1) log(.1)]';
  conversion = [2 0 0 0 1 1 1];
else
  model_f = 'gaussian1D';
  hazard_f = 'constant_h';
  num_hazard_params = 1;

  hazard_init = logit(.01)';
  model_init = [0 log(.1) log(.1) log(.1)]';
  conversion = [2 0 1 1 1];
end

theta = [hazard_init; model_init];

[theta, nlml] = rt_minimize(theta, @bocpd_dwrap1D, -max_minimize_iter, ...
  X, model_f, hazard_f, conversion, num_hazard_params);

hazard_params = theta(1:num_hazard_params);
model_params = theta((num_hazard_params + 1):end);

hazard_params(1) = logistic(hazard_params(1));
model_params(2:end) = exp(model_params(2:end));

function [nlml, dnlml] = ...
  bocpd_dwrap1D(theta, X, model_f, hazard_f, conversion, num_hazard_params)

% Warning: this code assumes: theta_h are in logit scale, theta_m(1) is in
% linear, and theta_m(2:end) are in log scale!

theta(conversion == 2) = logistic(theta(conversion == 2));
theta(conversion == 1) = exp(theta(conversion == 1));

% Seperate theta into hazard and model hypers
theta_h = theta(1:num_hazard_params);
theta_m = theta(num_hazard_params+1:end);

[nlml, dnlml_h, dnlml_m] = ...
  bocpd_deriv(theta_h', theta_m', X, hazard_f, model_f);

% Put back into one vector for minimize
dnlml = [dnlml_h; dnlml_m];

% Correct for the distortion
dnlml(conversion == 2) = dnlml(conversion == 2) .* theta(conversion == 2) .* (1 - theta(conversion == 2));
dnlml(conversion == 1) = dnlml(conversion == 1) .* theta(conversion == 1);
