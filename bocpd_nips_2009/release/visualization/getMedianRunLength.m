% Ryan Turner (rt324@cam.ac.uk)
% Yunus Saatci (ys267@cam.ac.uk)

function [Mrun, MchangeTime] = getMedianRunLength(S)

T = size(S, 2);
% TODO assert is upper triangular
% TODO assert cols sum to 1

% Warning: if we interpret the current point as being the first point in a new
% regime as run length = 0 then we need to subtract one from this.  currently
% use run = 1.

cdf = cumsum(S);
secondHalf = cdf >= .5;

Mrun = zeros(1, T);
for ii = 1:T
  Mrun(ii) = find(secondHalf(:, ii), 1, 'first');
end

MchangeTime = (1:T) - Mrun + 1;
