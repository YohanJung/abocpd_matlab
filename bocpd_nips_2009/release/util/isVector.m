% Ryan Turner rt324@cam.ac.uk
% Checks that the variable is a vector. A scalar is considered as a vector.
% An empty variable is considered as a vector.  Note a vector along the
% third dimension is not supported here unlike in checkvar.  Also, this
% function does not require X to be numeric.

function valid = isVector(X)

[r c] = size(X);
valid = (r <= 1 || c <= 1);
