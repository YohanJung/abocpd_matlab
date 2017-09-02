% function Z = rmult(X, Y)
%
% row multiplication: Z = X * Y row-wise
% Y must have one column 

function Z = rmult(X, Y)

[N M] = size(X);
[K L] = size(Y);

if N ~= K || L ~=1
  disp('Error in RMULT');
  return;
end

Z = bsxfun(@times, X, Y);
