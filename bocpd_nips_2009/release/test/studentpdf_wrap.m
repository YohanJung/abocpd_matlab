

function [p, dp] = studentpdf_wrap(theta, x, theta_all, idx)

theta_all(idx, 1) = theta(1);
theta_all(idx, 2) = theta(2);
theta_all(idx, 3) = theta(3);

[p, dp] = studentpdf(x, theta_all(:, 1), theta_all(:, 2), theta_all(:, 3));
p = p(idx);
dp = dp(idx, :);
