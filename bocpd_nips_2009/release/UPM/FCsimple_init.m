function [post_params, dmessage] = FCsimple_init(T, D, theta_m)

%First do some assertions about the hypers
assert(length(theta_m) == 2);
assert(theta_m(1) > 0);
assert(theta_m(2) > D);

post_params = zeros(D*D + D + 3, 1);
post_params(2) = theta_m(1);
post_params(3) = theta_m(2);

dmessage = [];
