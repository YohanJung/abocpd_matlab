% Ryan Turner rt324@cam.ac.uk

function Y = logit(X)

Y = log(X ./ (1 - X));
