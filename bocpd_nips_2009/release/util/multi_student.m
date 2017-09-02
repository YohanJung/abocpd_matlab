function [logp, dmu, dlambda, dnu] = multi_student(x, mu, sigma, nu)

D = size(x, 1);
L = chol(sigma, 'lower');
x0 = x - mu;

% Gelman p. 576
% remember: det sigma = prod of diag of L (lower chol)
logp = gammaln((nu + D) / 2) - gammaln(nu / 2) - ((D / 2) * log(nu * pi)) ...
  - sum(log(diag(L))) - (((nu + D) / 2) * log(1 + (x0' * (L' \ (L \ x0))) / nu));
% More readable version:
% logp = gammaln((nu + D) / 2) - gammaln(nu / 2) - ((D / 2) * log(nu * pi)) ...
%  - 0.5*log(det(sigma)) - (nu + D) / 2 * log(1 + (xi-mu)'*inv(sigma)*(x-mu)/nu);

if nargout == 4
    %precompute some quantities for the derivatives
    invSigma = L' \ (L \ eye(D));
    chol_sol =  L' \ (L \ x0);
    quad = 1 + ((x0' * chol_sol) / nu);
    %compute derivatives w.r.t. parameters
    dmu =  - ( - ((2/nu) .* chol_sol) .* ((nu + D) / (2 * quad)) );
    dlambda = - ( 0.5.*invSigma + (((nu + D) / (2 * quad)) .* ( - (invSigma * (x0 * x0') * invSigma) ./ nu)) );
    dlambda = dlambda + dlambda'; %correct for the fact that lambda is constrained to be symmetric
    dlambda = dlambda./2;
%     for d=1:D
%         dlambda(d,d) = dlambda(d,d)/2;
%     end
    dnu = 0.5*digamma((nu+D)/2) - 0.5*digamma(nu/2) - ( D/(2*nu) + ((nu + D)/(2*quad) * (- ((x0' * chol_sol) / (nu*nu)) )) + (0.5*log(quad)) );
end

