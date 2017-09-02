% Ryan Turner (rt324@cam.ac.uk)
% Checks to see if a matrix is kosher meaning all the values are finite (no
% -Inf, +Inf, NaN) and it is real (no complex numbers).  If check_pos is 1 then
% it also checks if all the elements are more than zero.

function kosher = isKosher(X, check_pos)

if nargin == 2 && check_pos == 1
  kosher = allX(isfinite(X) & isreal(X) & (X > 0));
else
  kosher = allX(isfinite(X) & isreal(X));
end
