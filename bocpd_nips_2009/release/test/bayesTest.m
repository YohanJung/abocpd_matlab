

function endVal = bayesTest(X, model_f, testNum, theta_m)

T = length(X);
hyper_count = length(theta_m);
endVal = zeros(hyper_count, testNum);

for testIter = 1:testNum
  ordering = randperm(T);

  post_params = feval(model_f, 'update_posteriors', theta_m);
  for t = 1:T
    post_params = feval(model_f, 'update_posteriors', theta_m, post_params, X(ordering(t)));
  end
  all_data_params = post_params(end, :);
  endVal(:, testIter) = all_data_params(:);
end
