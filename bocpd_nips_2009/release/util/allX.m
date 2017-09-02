% Ryan Turner (rt324@cam.ac.uk)
% Similar to all2 but it works with higher dimensional object too.  It is the
% "real" all function as in all the elements of a matrix.

function y = allX(X)

D = ndims(X);

if D <= 2
  y = all(all(X));
elseif D == 3
  y = all(all(all(X)));
elseif D == 4
  y = all(all(all(all(X))));
else
  % TODO put in for loop
  error('allX not supported for D > 4 yet');
end
