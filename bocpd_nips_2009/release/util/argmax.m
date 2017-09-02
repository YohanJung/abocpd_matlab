% Ryan Turner rt324@cam.ac.uk

function idx = argmax(X, dim)

if nargin == 2
  [temp, idx] = max(X, dim);
else
  [temp, idx] = max(X);
end
