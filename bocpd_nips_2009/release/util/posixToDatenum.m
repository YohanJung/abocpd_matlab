

function dateval = posixToDatenum(posix)

secondsInDay = 86400; % 60 * 60 * 24
offset = 719529; % datenum('January 1, 1970  12:00:00 AM')

dateval = double(posix) ./ secondsInDay + offset;
