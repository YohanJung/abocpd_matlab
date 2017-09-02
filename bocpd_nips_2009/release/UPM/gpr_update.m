

function post_params = gpr_update(theta_prior, post_params, xt, dpost, maxRunConsidered)

if isempty(post_params)
  T = 1;
else
  T = post_params(1, 1) + 1;
end

post_params = [T xt; post_params];

if exist('maxRunConsidered', 'var')
  post_params = post_params(1:maxRunConsidered - 1, :);
end
