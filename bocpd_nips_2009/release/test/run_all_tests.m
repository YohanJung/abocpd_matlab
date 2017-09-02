

% Run all unit tests function

function run_all_tests()

rand('state', 20);
randn('state', 24);

% --- checks on model modules:
% try random hypers => no symetry
% random X from synthetic data
% repeat with many random draws
% Bayes consistency check on post_param updates
  % => random permutation and assert same post_params at end

% Bayes consistency check on predictive
  % => sum of one step ahead predictive should be the same under random
  % permutations

% do Bayes consistency check on dpredictive/dmodel
  % function invariant to permutation => derivative invariant to permutation

% assert derivatives, post_params, and likelihood equivalent to offline version

% --- checks on bocpd_deriv general system. If modules pass their tests the
% whole system should pass these tests if it is correct.
% Check grad on dR/dhazard and dR/dmodel
%  - even in fast versions that don't ordinarily return dR

% check grad on dZ/dhazard and dZ/dmodel

% check grad on dnlml/dhazard and dnlml/dmodel

% check that the nlml is the same as bocpd (not deriv)

% test sum(dR) over runlength index should always be 0 to machine prec

% test extreme inputs:
  % hazard -> 0 and hazard -> 1
  % mu = 0 and mu -> -Inf and mu -> Inf
  % positive params -> 0 and positive params -> Inf
  % Test at random. each hyper can be in three states (small limit, normal,
  % large limit). Try different combos
% check the nlml and derivatives with isKosher
% check grad at extreme values too
% note: there exist some combos that will not work because we are not doing log
% scale stuff (rescaling instead).

% if hazard = 1 => runlength should always be zero
% if hazard = 0 => runelgnth should always be max

% Test illegal values make sure they result in assertion error:
%  [0,1] quanitites, positive quantities, NaNs, row dims, ...

% assert R is upper triangular & sums to one on cols, assert size

% be sure tests include non-constant hazard function

% test if hazard = 1 then nlml = nlml for const gaussian
% test if hazard = 0 then nlml = nlml for prior predictive on each point
% test if hazard = 0/1 & prior var on prec -> and prior var on mean -> 0 then
% nlml = nlml for gaussian with known parameters

% --- testing certain modules
%  IFM: assert that IFM derivatives and likelihood equivalent to doining
%  gaussian1D on each channel indep.

% test IFMfast == IFM results. IFMfast does caching

% hazard functions:
% check grad
% assert in [0,1]

% --- sparse version:
  % set epsilon -> 0 => results should be almost identical to non-pruned code

% --- other
% test_student_pdf derivatives

%[d dy dh] = jf_checkgrad({'bocpd_dwrap', X, 'gaussian1D', 'constant_h', 1}, [-4,0,0,0,0]', 1e-4);

model_params = rand(1, 12) + 1;
X = IFMrnd(1000, 3, model_params, .01);

diary on;

setting = zeros(1, 12);
error_d = zeros(1, 2);
time_running = 0;
ii = 0;
tic;
while time_running <= 60*60*8
  ii = ii + 1;
  setting(ii, :) = randn(1, 12);
  setting(ii, 4:end) = exp(setting(ii, 4:end));
  [error_d(ii, :) dy dh] = jf_checkgrad({@check_bocpd_deriv, X}, setting(ii, :), 1e-4);
  disp(setting(ii, :));
  disp(error_d(ii, :));
  if any(error_d(ii, :) > 1e-7)
    disp('Error!');
  end
  time_running = toc;
end

save checkgrad_anal

function [R, dR_m] = check_bocpd_deriv(theta_m, X)

[nlml, dnlml_h, dnlml_m, Z, dZ_h, dZ_m, R, dR_h, dR_m] = ...
  bocpd_deriv(.01, theta_m, X, 'constant_h', 'IFMfast');
