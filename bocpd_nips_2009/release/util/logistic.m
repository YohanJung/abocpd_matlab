% Ryan Turner rt324@cam.ac.uk

function Y = logistic(X)

Y = 1 ./ (1 + exp(-X));
