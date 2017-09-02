

function [mu, scale, df] = blr(X, Y, xstar)

[N K] = size(X);
E = size(Y, 2);
assert(size(Y, 1) == N);
assert(size(xstar, 2) == K);
N2 = size(xstar, 1);

assert(E == 1 || N2 == 1);

% eqn 14.4
beta_hat = pinv(X) * Y; % K x E
VBinv = X' * X; % K x K

residual = Y - X * beta_hat; % N x E
s2 = residual' * residual / (N - K); % E x E

mu = xstar * beta_hat; % N2 x E
scale = s2 * (eye(N2) + xstar * (VBinv \ xstar')); % (E x E) x (N2 x N2)
% => only works if E = 1 OR N2 = 1. TODO make work in general case
% Use kron() for this?
df = (N - K); % 1 x 1
