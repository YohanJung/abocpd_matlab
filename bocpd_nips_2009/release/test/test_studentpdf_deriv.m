
rand('state', 4);
randn('state', 4);

runs = 3000;

% Might need to decrease the step size further for super smaller variances ~
% .0001.  Should be optimizing wrt log variance to be super stable.
step_size = 1e-6;
tol = 1e-7;

for ii = 1:runs
  idx = 5;
  x = rand(10, 1);
  theta_all = rand(10, 3) * 5;
  [d dy dh] = jf_checkgrad({'studentpdf_wrap', x, theta_all, idx}, theta_all(idx, :), step_size);
  if sum(d) >= tol
    disp('Error');
    keyboard;
  end
end

disp('Done Testing.');
