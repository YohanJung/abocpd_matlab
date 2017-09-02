% Ryan Turner rt324@cam.ac.uk

function Y = standardize_cols(X)

% What to do if col sums to zero?  Then we get NaN.  Change that?

col_means = nanmean(X, 1);
col_stds = nanstd(X, 0, 1);
Y = cplus(X, -col_means);
Y = cdiv(Y, col_stds);
