function [p, dp] = studentpdf(x, mu, var, nu)
%
% p = studentpdf(x, mu, var, nu)
%
% Can be made equivalent to MATLAB's tpdf() by:
% tpdf((y - mu) ./ sqrt(var), nu) ./ sqrt(var)
% Equations found in p. 577 of Gelman

% TODO assert size consistency

% This form is taken from Kevin Murphy's lecture notes.
c = exp(gammaln(nu / 2 + 0.5) - gammaln(nu / 2)) .* (nu .* pi .* var) .^ (-0.5);
p = c .* (1 + (1 ./ (nu .* var)) .* (x - mu) .^ 2) .^ (-(nu + 1) / 2);

if nargout == 2
  N = length(mu);
  dp = zeros(N, 3);

  error = (x - mu) ./ sqrt(var);
  sq_error = (x - mu) .^ 2 ./ var;

  % derivative for mu
  dlogp = (1 ./ sqrt(var)) .* ((nu + 1) .* error) ./ (nu + sq_error);
  dp(:, 1) = p .* dlogp;

  % derivative for var
  dlogpdprec = sqrt(var) - ((nu + 1) .* (x - mu) .* error) ./ (nu + sq_error);
  dp(:, 2) = - .5 * (p ./ var .^ 1.5) .* dlogpdprec;

  % derivative for nu (df)
  dlogp = psi(nu / 2 + .5) - psi(nu / 2) - (1 ./ nu) - log(1 + (1 ./ nu) .* sq_error) + ((nu + 1) .* sq_error) ./ (nu .^ 2 + nu .* sq_error);
  dp(:, 3) = .5 * p .* dlogp;
end
