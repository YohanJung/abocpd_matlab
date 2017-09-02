

function pred = blr_predict(post_params, xnew)

x0 = [-1; 0; 1];
y0 = [-1; 1; -1];

if isempty(post_params)
  t = 1;
else
  t = post_params(1, 1) + 1;
end

max_run = size(post_params, 1);

mu = zeros(max_run + 1, 1);
sigma = zeros(max_run + 1, 1);
df = zeros(max_run + 1, 1);
for ii = 0:max_run
  [mu(ii + 1), sigma(ii + 1), df(ii + 1)] = ...
    blr([x0; post_params(1:ii, 1)], [y0; post_params(1:ii, 2)], t);
end

pred = studentpdf(xnew, mu, sigma, df);
