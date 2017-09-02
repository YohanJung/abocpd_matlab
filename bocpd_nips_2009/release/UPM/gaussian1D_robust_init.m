

function [post_params, dmessage] = gaussian1D_robust_init(T, D, theta_m)

post_params = theta_m;

assert(D == 1);
dmessage = eye(4);
