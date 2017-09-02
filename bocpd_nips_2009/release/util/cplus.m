% function Z = cplus(X, Y)
%
% col addition: Z = X + Y col-wise
% Y must have one row

function Z = cplus(X, Y)

[N M] = size(X);
[K L] = size(Y);

if M ~= L || K ~=1
  disp('Error in CPLUS');
  return;
end

Z = bsxfun(@plus, X, Y);
