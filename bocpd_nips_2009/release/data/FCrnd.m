function [X changePoints] = FCrnd(T, D, kappa0, nu0, hazard_rate)

changePoints = false(T, 1);
X = zeros(D, T);
assert(nu0 > D); %nu0 should be greater than D here. 
assert(kappa0 > 0); 

for t = 1:T
  if rand() < hazard_rate || t == 1
    % Generate new Gaussian parameters from the prior.
%     L = randwishart(nu0, D)'; % lower triangular matrix = chol(precision, 'lower')
%     Sigma = L' \ (L \ eye(D)); % invert the matrix => get the covariance
    Sigma = iwishrnd(eye(D), nu0);
    mu = randnorm(1, zeros(D,1), [], Sigma ./ kappa0); % p. 87 of Gelman
    changePoints(t) = true;
  end
  X(:, t) = randnorm(1, mu, [], Sigma); 
end
