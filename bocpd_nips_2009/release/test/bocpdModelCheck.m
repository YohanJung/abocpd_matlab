

function bocpdModelCheck(S, X)

[T D] = size(X);

thold = .5;
alert = convertToAlert(S, thold);

changePoints = find(alert);
change_point_num = length(changePoints);

assert(changePoints(1) == 1);
changePoints = [changePoints; T + 1];

regimeDuration = zeros(change_point_num, 1);
meanX = zeros(change_point_num, D);
covX = zeros(D, D, change_point_num);
Xwhite = zeros(T, D);
pnorm = NaN(change_point_num, 1);
corrX = NaN(change_point_num, D);
for ii = 1:change_point_num
  Xcurr = X(changePoints(ii):changePoints(ii + 1) - 1, :);
  regimeDuration(ii) = size(Xcurr, 1);
  meanX(ii, :) = mean(Xcurr);
  covX(:, :, ii) = cov(Xcurr);

  Xwhite(changePoints(ii):changePoints(ii + 1) - 1, :) = whitten(Xcurr);

  if regimeDuration(ii) > 4
    [h, pnorm(ii)] = lillietest(sum(Xcurr, 2), .05);

    for jj = 1:D
      [R, P] = corrcoef(Xcurr(1:end-1, jj), Xcurr(2:end, jj));
      corrX(ii, jj) = P(1, 2);
      % TODO check for trend
    end
  end
end

keyboard;

