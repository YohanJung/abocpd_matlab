

function lml = bayesTest2(X, model_f, testNum, theta_m)

[T,D] = size(X);
lml = zeros(testNum, 1);

for testIter = 1:testNum
  ordering = randperm(T);
  
  %post_params = theta_m;
  post_params = feval([model_f, '_init'], T + 1, D, theta_m);
  for t = 1:T
    predprobs = feval([model_f '_predict'], post_params, X(ordering(t), :));
    lml(testIter) = lml(testIter) + log(predprobs(end));
    post_params = feval([model_f '_update'], theta_m, post_params, X(ordering(t), :));
  end
end
