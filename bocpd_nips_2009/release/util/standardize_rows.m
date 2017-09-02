% Ryan Turner rt324@cam.ac.uk

function Y = standardize_rows(X)

% What to do if row sums to zero?  Then we get NaN.  Change that?

row_means = nanmean(X, 2);
row_stds = nanstd(X, 0, 2);
Y = rplus(X, -row_means);
Y = rdiv(Y, row_stds);
