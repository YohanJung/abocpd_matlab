

function pred = gpr_predict(post_params, xnew)
covfunc = {'covSEiso'};
%covfunc = {'covSum', {'covSEiso', 'covNoise'}};
loghyper = [1.8895   -0.6947   -1.3123]';

if isempty(post_params)
  t = 1;
  post_params(1, 2) = xnew;    
else
  t = post_params(1, 1) + 1;
  post_params(1, 2) = xnew;    
end
    
[mu, sigma] = gpr1step2(loghyper, covfunc, post_params(:, 1), post_params(:, 2) , t);

pred = normpdf(xnew, mu, sqrt(sigma));

function [mu, sigma] = ...
    gpr1step2(loghyper, covfunc, train_input, train_output, t) 
    
    hyp.mean = 0;
    hyp.cov = loghyper(1:2);
    hyp.lik = loghyper(3);

    mean = {'meanConst'};
    cov = covfunc;
    lik = {'likGauss'};
    inf = @infExact;
%     train_input = post_params(2:t,1);
%     train_output = post_params(2:t,2);

[mu, sigma] = gp(hyp,inf,mean,cov,lik,train_input,train_output,t);




