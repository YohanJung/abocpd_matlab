% Ryan Turner (rt324@cam.ac.uk)
% Yunus Saatci (ys267@cam.ac.uk)

function [Mrun, MchangeTime] = getMAPRunLength(S)

T = size(S, 1);
assert(size(S, 2) == T);
% TODO assert is upper triangular
% TODO assert cols sum to 1

% Warning: if we interpret the current point as being the first point in a new
% regime as run length = 0 then we need to subtract one from this.  currently
% use run = 1.
Mrun = argmax(S);
MchangeTime = (1:T) - Mrun + 1;
