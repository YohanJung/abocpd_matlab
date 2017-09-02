

function [post_params, dmessage] = IFMfast_init(T, D, theta_m)

post_params = theta_m;

dmessage = repmat(eye(4,4), [1 1 T D]);
dmessage = permute(dmessage, [3 1 2 4]);

% TODO replace with min(T, 5000);
setupGcache(theta_m, 5000, D);

function setupGcache(theta_prior, N, D)

global Gcache;
global Pcache;

alpha_prior = theta_prior((2 * D + 1):3 * D);

nu = cplus(repmat((0:N)', 1, D), 2 * alpha_prior);

Gcache = exp(gammaln(nu / 2 + 0.5) - gammaln(nu / 2));
Pcache = psi(nu / 2 + .5) - psi(nu / 2) - (1 ./ nu);
