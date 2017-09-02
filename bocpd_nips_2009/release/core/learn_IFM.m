
function [hazard_params, model_params, nlml] = learn_IFM(X, useLogistic, init)

D = size(X, 2);

max_minimize_iter = 30;

if nargin > 1 && useLogistic
  model_f = 'IFMfast';
  hazard_f = 'logistic_h';
  num_hazard_params = 3;

  hazard_init = [logit(.01) 0 0]';
  model_init = [zeros(1, D) log(ones(1, D)) log(ones(1, D)) log(ones(1, D))]';
  conversion = [2 0 0 zeros(1, D) ones(1, 3 * D)];
else
  model_f = 'IFMfast';
  hazard_f = 'constant_h';
  num_hazard_params = 1;

  hazard_init = logit(.01);
  model_init = [zeros(1, D) log(ones(1, D)) log(ones(1, D)) log(ones(1, D))]';
  conversion = [2 zeros(1, D) ones(1, 3 * D)];
end

if nargin == 3 && ~isempty(init)
  theta = init;
  theta(conversion == 2) = logit(theta(conversion == 2));
  theta(conversion == 1) =   log(theta(conversion == 1));
else
  theta = [hazard_init; model_init];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[theta, nlml] = rt_minimize(theta, @bocpd_dwrap_sp, -max_minimize_iter, ...
  X, model_f, hazard_f, conversion, num_hazard_params);

theta(conversion == 2) = logistic(theta(conversion == 2));
theta(conversion == 1) = exp(theta(conversion == 1));

hazard_params = theta(1:num_hazard_params);
model_params = theta((num_hazard_params + 1):end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [nlml, dnlml] = ...
  bocpd_dwrap_sp(theta, X, model_f, hazard_f, conversion, num_hazard_params)

theta(conversion == 2) = logistic(theta(conversion == 2));
theta(conversion == 1) = exp(theta(conversion == 1));

% Seperate theta into hazard and model hypers
theta_h = theta(1:num_hazard_params);
theta_m = theta(num_hazard_params+1:end);

[nlml, dnlml_h, dnlml_m] = ...
  bocpd_deriv_sparse(theta_h', theta_m', X, hazard_f, model_f, .001);

% Put back into one vector for minimize
dnlml = [dnlml_h; dnlml_m];

% Correct for the distortion
dnlml(conversion == 2) = dnlml(conversion == 2) .* theta(conversion == 2) .* (1 - theta(conversion == 2));
dnlml(conversion == 1) = dnlml(conversion == 1) .* theta(conversion == 1);
