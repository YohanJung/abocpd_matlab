

function [post_params, dmessage] = IFM_init(T, D, theta_m)

post_params = theta_m;

dmessage = repmat(eye(4,4), [1 1 T D]);
dmessage = permute(dmessage, [3 1 2 4]);
