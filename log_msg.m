function log_msg(fmt, varargin)
%LOG_MSG Print a timestamped message to the MATLAB terminal.
%   LOG_MSG(FMT, ...) works like fprintf but prepends [yyyy-mm-dd HH:MM:SS].
%
%   Example:
%       log_msg('Starting iteration %d of %d', i, N);
%       % Output: [2026-02-08 09:15:02] Starting iteration 3 of 10
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    msg = sprintf(fmt, varargin{:});
    fprintf('[%s] %s\n', timestamp, msg);
end