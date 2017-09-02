

function [theta_new, dtheta] = gaussian1D_wrap(theta_try, theta_m, post_params, dmessage, X, idx)

post_params(idx, :) = theta_try;

[post_params, dtheta_d0] = gaussian1D('update_posteriors', theta_m, post_params, dmessage, X);

theta_new = post_params(idx + 1, :);
dtheta = dtheta_d0(:, :, idx + 1) ./ dmessage(:, :, idx);
