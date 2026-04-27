function [u_hat, v_hat, H_hat] = ls_dc_estimator(y_act, N, Nact, fc, d_elem, K_users, layout)
%LS_DC_ESTIMATOR Least-Squares + Direct-Calculation angle estimator.
%   [u_hat, v_hat, H_hat] = LS_DC_ESTIMATOR(y_act, N, Nact, fc, d_elem, K_users, layout)
%
%   EXTENSION vs original:
%     layout : (optional, default 'center') active element placement layout.
%              One of: 'center' | 'corners' | 'cross' | 'edges' | 'diagonal' | 'random'
%
%   GENERALIZED PHASE-DIFFERENCE SOLVER
%   ------------------------------------
%   The original code assumed a 2x2 center block and extracted u,v from
%   two scalar phase differences. This version works for ANY layout by
%   solving an over-determined linear system:
%
%       For every pair of active elements (m, n):
%           angle(g_m) - angle(g_n) = u*(col_m - col_n) + v*(row_m - row_n) + 2*pi*k
%
%   Stacking all Nact*(Nact-1)/2 pairs gives:
%       A * [u; v] = phi_vec    (over-determined, solved via LS)
%
%   where A(:,1) = delta_col, A(:,2) = delta_row, phi_vec = wrapped phase diffs.
%
%   For the original 2x2 center layout this reduces to exactly the same
%   answer as the original scalar formulas (since only 2 pairs are
%   linearly independent), so backward-compatibility is preserved.
%
%   WHY THIS MATTERS (the research contribution)
%   -----------------------------------------------
%   Different layouts produce different A matrices with different condition
%   numbers, directly affecting estimation variance. Corner layouts with
%   maximum baseline separation yield well-conditioned A and thus lower
%   NMSE. This function lets us measure that effect.

if nargin < 7 || isempty(layout)
    layout = 'center';
end

c0 = 3e8;
%lambda = c0 / fc;   % reserved for potential future use

Nv = sqrt(N); Nh = Nv;
if abs(Nv - round(Nv)) > 1e-12
    error('ls_dc_estimator: N must correspond to a square RIS.');
end
Nv = round(Nv);
Nh = Nv;

%% Active element indices via centralised helper
act_lin_idx = get_active_indices(Nv, Nh, Nact, layout);

% Convert linear indices to (row, col) for the phase-difference matrix
[row_act, col_act] = ind2sub([Nv, Nh], act_lin_idx);   % 1-based

numMCS = size(y_act, 1);

u_hat = zeros(numMCS, 1);
v_hat = zeros(numMCS, 1);
H_hat = zeros(numMCS, N);

s = sqrt(K_users);   % pilot magnitude

%% Pre-build the geometry matrix A (same for every MC trial)
%   All pairs (i,j), i<j
[ii, jj] = find(tril(ones(Nact), -1));   % lower triangle -> all pairs
nPairs   = numel(ii);

A = zeros(nPairs, 2);
for p = 1:nPairs
    A(p, 1) = col_act(ii(p)) - col_act(jj(p));   % delta_col  (u direction)
    A(p, 2) = row_act(ii(p)) - row_act(jj(p));   % delta_row  (v direction)
end

% Remove zero-difference pairs (elements in the same row AND column,
% which can happen with some layouts after trimming)
nonzero = any(A ~= 0, 2);
A  = A(nonzero, :);
ii = ii(nonzero);
jj = jj(nonzero);

%% Per-trial estimation
for i = 1:numMCS
    y_i = y_act(i,:).';       % Nact x 1

    % LS estimate of active channels
    g_act_hat = y_i / s;

    %% Generalized phase-difference LS
    phi_vec = zeros(size(A,1), 1);
    for p = 1:size(A,1)
        raw_diff  = angle(g_act_hat(ii(p))) - angle(g_act_hat(jj(p)));
        phi_vec(p) = atan2(sin(raw_diff), cos(raw_diff));  % wrap to [-pi,pi]
    end

    % Weighted LS via ridge-regularized normal equations (no rank warnings)
    mag_weight = sqrt(abs(g_act_hat(ii)) .* abs(g_act_hat(jj)));
    uv = solve_phase_ls(A, phi_vec, mag_weight);

    u_est = uv(1);
    v_est = uv(2);

    u_hat(i) = u_est;
    v_hat(i) = v_est;

    %% Full steering vector and gain estimation
    a = steering_vector_UPA(Nv, Nh, u_est, v_est);

    a_act    = a(act_lin_idx);
    alpha_hat = (a_act' * g_act_hat) / (a_act' * a_act);

    H_hat(i,:) = (alpha_hat * a).';
end

end
