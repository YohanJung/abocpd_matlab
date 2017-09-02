% function Z = cdiv(X, Y)
%
% col division: Z = X / Y col-wise
% Y must have one row

function Z = cdiv(X, Y)

[N M] = size(X);
[K L] = size(Y);

if M ~= L || K ~=1
  disp('Error in CDIV');
  return;
end

Z = bsxfun(@rdivide, X, Y);
