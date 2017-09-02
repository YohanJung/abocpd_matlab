function [hazard_params, model_params, nlml, R, S] = learnFC(X, theta_init)

num_hazard_params = 1;
D = size(X, 2);

theta = theta_init;
theta(1) = logit(theta(1));
theta(2) =   log(theta(2));
theta(3) = log(theta(3)-D);

[theta, nlml] = minimize(theta, @bocpd_dwrap_fc, -20, X, 'FCsimple', 'constant_h', num_hazard_params);

theta(1) = logistic(theta(1));
theta(2) = exp(theta(2));
theta(3) = exp(theta(3)) + D;

hazard_params = theta(1:num_hazard_params);
model_params = theta((num_hazard_params + 1):end);

%[R, S, nlml] = bocpd(X, 'FCsimple', model_params', 'constant_h', hazard_params');
[R S nlml] = bocpd_sparse(hazard_params', model_params', X, 'constant_h', 'FCsimple', 0.001); 

function [nlml, dnlml] = bocpd_dwrap_fc(theta, X, model_f, hazard_f, num_hazard_params)

D = size(X,2);
theta(1) = logistic(theta(1));
theta(2) = exp(theta(2));
theta(3) = exp(theta(3)) + D;

disp(sprintf('theta = %d, %d, %d', theta(1),theta(2), theta(3)));

% Seperate theta into hazard and model hypers
theta_h = theta(1:num_hazard_params);
theta_m = theta(num_hazard_params+1:end);

[nlml, dnlml_h, dnlml_m] = bocpd_deriv_sparse(theta_h', theta_m', X, hazard_f, model_f, 0.001);

% Put back into one vector for minimize
dnlml = [dnlml_h; dnlml_m];

disp(sprintf('dnlml = %d, %d, %d', dnlml(1),dnlml(2), dnlml(3)));

% Correct for the distortion
dnlml(1) = dnlml(1) .* theta(1) .* (1 - theta(1));
dnlml(2) = dnlml(2) .* theta(2);
dnlml(3) = dnlml(3) .* (theta(3) - D);

disp(sprintf('nlml = %d', nlml));
