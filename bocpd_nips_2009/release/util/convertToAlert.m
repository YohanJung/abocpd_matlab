
function alert = convertToAlert(Rs, thold)

[max_run T] = size(Rs);
last_alarm = Inf;
alert = false(T, 1);
for ii = 1:T
  [alert(ii), last_alarm] = convertToAlertSingle(Rs(:, ii), last_alarm, thold);
end

function [alert, last_alarm] = convertToAlertSingle(Rs, last_alarm, thold)

% Rs(1) = 1 => current observation is certainly first after change point
% last_alarm = 1 => alerted 1 time step ago => Rs(2) refers to time step with
% at which we last alarmed.
% we are interested in P(change point after last alarm) => if last_alarm = 10
% then we want to know P(change point between 0 and 9 time steps ago) => we use
% sum(Rs(1:last_alarm)) = sum(Rs(1:10)).
% Need to be very careful here, it is easy to make off by one error,

if last_alarm > length(Rs)
  last_alarm = length(Rs); % 1 x 1. [bins]
end

changePointProb = sum(Rs(1:last_alarm)); % 1 x 1. [P]

if changePointProb >= thold
  alert = true;
  % At this iteration time since last alarm is zero. but it will be 1 next time
  % this is called.
  last_alarm = 1; % 1 x 1. [bins]
else
  alert = false;
  last_alarm = last_alarm + 1; % 1 x 1. [bins]
end
