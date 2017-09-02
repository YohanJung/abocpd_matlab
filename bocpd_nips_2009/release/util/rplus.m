% function Z = rdiv(X, Y)
%
% row addition: Z = X + Y row-wise
% Y must have one column 

function Z = rplus(X, Y)

[N M] = size(X);
[K L] = size(Y);

if N ~= K || L ~=1
  disp('Error in RPLUS');
  return;
end

Z = bsxfun(@plus, X, Y);
