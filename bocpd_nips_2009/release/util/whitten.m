

function Y = whitten(x, N)

% Is this scale invariant?
min_lambda = 1e-10;

T = size(x, 1);

if nargin == 1
  N = T;
end

x_bar = mean(x(1:N, :));
sigma = cov(x(1:N, :));

% Get the eigenvectors and corresponding eigenvalues for this data.
[U, lambda] = eig(sigma);

% Avoid numerical problems.
lambda = diag(lambda);
lambda(lambda <= min_lambda) = min_lambda;
lambda_inv = diag(1 ./ sqrt(lambda));

% Whiten the data by applying the following linear transform. and plot the
% result.
% TODO replace ones() with rplus
% TODO get rid of all the transposes
% Y = (U * lambda ^ (-.5) * U' * (x - ones(T, 1) * x_bar)')';
Y = (U * lambda_inv * U' * (x - ones(T, 1) * x_bar)')';
