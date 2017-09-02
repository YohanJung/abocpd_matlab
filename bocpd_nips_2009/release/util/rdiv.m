% function Z = rdiv(X, Y)
%
% row division: Z = X / Y row-wise
% Y must have one column 

function Z = rdiv(X, Y)

[N M] = size(X);
[K L] = size(Y);
if N ~= K || L ~=1
  disp('Error in RDIV');
  return;
end

Z = bsxfun(@rdivide, X, Y);
