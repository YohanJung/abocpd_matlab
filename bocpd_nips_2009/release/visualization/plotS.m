% Ryan Turner (rt324@cam.ac.uk)
% Yunus Saatci (ys267@cam.ac.uk)

function [h1, h2] = plotS(S, X, changePoints, timeindex)

alertThold = .5;

if nargin <= 3
  timeindex = 1:size(X, 1);
end

% Plot the data
figure;
h1 = subplot(2, 1, 1);
plot(timeindex', X);

alert = convertToAlert(S, alertThold);
hold on;
plot(find(alert), 0, 'rx');

if nargin == 3 && ~isempty(changePoints)
  CP = find(changePoints);
  hold on;
  plot(CP, zeros(size(CP)), 'kx', 'MarkerSize', 12, 'LineWidth', 2);
  hold off;
end
axis tight;
grid on;

% Plot the inferred S
h2 = subplot(2, 1, 2);
colormap gray;
imagesc(timeindex, 1:size(S, 1), cumsum(S));
Mrun = getMedianRunLength(S);
hold on;
plot(timeindex, Mrun, 'r-');
axis tight;
hold off;
