% function Z = requal(X, Y)
%
% row addition: Z = (X == Y) row-wise
% Y must have one column 

function Z = requal(X, Y)

[N M] = size(X);
[K L] = size(Y);

if N ~= K || L ~=1
  disp('Error in Requal');
  return;
end

Z = bsxfun(@eq, X, Y);
