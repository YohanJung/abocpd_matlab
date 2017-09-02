function [post_params, dmessage] = FCsimple_update(theta_m, post_params, xt, dmessage, maxRunConsidered)
% Update in the case where mu0 = 0 and lambda0 = eye(D).
% Sufficient statistics in this case include just enough info to construct
% the posteriors parameters. (N, kappa_N, nu_N, sum, sum_outer) 

% dmessage is not altered (always empty) -- only do derivatives thru
% dpredprobs

% work with column vector observations here
xt = xt'; % D x 1
D = length(xt);
assert(size(post_params,1) == (D*D + D + 3));

%Update sufficient statistics post_params. size of post_params is (3+D+D^2) * T
T = size(post_params, 2);
% update numpoints, kappa, nu
post_params(1:3, 2:T+1) = post_params(1:3,:) + 1; % 3 x T
% update the sum of the data: rplus(post_params(4:D+3, 1:T), xt)
post_params(4:D+3, 2:T+1) = bsxfun(@plus, post_params(4:D+3, 1:T), xt);
% update sum of outer products of data points:
% outer_xt = xt * xt';
% rplus(post_params(D+4:end, 1:T), outer_xt(:))
post_params(D+4:end, 2:T+1) = bsxfun(@plus, post_params(D+4:end, 1:T), reshape(xt*xt',D*D,1));
% the base case, numpoints, kappa0, nu0
post_params(1:3,1) = [0; theta_m(1); theta_m(2)];
% set empirical sums to zero
post_params(4:end,1) = zeros(D * D + D, 1);

if nargin > 4
    %crop post_params
    post_params = post_params(:, 1:maxRunConsidered);
end

