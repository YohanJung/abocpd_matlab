

function [pp, dtheta] = gaussian1D_wrap2(theta_try, post_params, dmessage, X, idx)

post_params(idx, :) = theta_try;

[predprobs, dpp_d0] = gaussian1D('eval_predictives', X, post_params, dmessage);

dtheta = dpp_d0(idx, :) * inv(dmessage(:, :, idx));
pp = predprobs(idx);

% for ii = 1:4
%   dtheta(ii) = dpp_d0(ii, idx) ./ dmessage(:, :, idx);
% end
