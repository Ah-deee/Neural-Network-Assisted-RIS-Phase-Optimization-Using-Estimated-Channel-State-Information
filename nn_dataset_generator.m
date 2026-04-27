function [X, Y, SNR_labels, u_mean, u_std, v_mean, v_std, X_mean, X_std, loss_weights] = ...
    nn_dataset_generator(D, SNR_dB, fc, d_elem, beta0, d0, d_k_set, alpha_PL, K_users, Nact, N, layout)
%NN_DATASET_GENERATOR Generate dataset with NMSE-aware loss weighting.
%
%   ADDED PARAMETER:
%     layout : (optional, default 'center') active element placement layout.
%              Passed directly to GET_ACTIVE_INDICES. Must match the layout
%              used during evaluation (generate_channel, ls_dc_estimator).
%
%   All other inputs/outputs unchanged from nn_dataset_generator_v4.

%% Defaults
if nargin < 11 || isempty(N);      N      = 16;      end
if nargin < 12 || isempty(layout); layout = 'center'; end

Nv = sqrt(N);
if abs(Nv - round(Nv)) > 1e-12
    error('nn_dataset_generator: N must be a perfect square.');
end
Nv = round(Nv);  Nh = Nv;

%% SNR-weighted sampling
if numel(SNR_dB) == 1
    SNR_vec = repmat(SNR_dB, D, 1);
else
    snr_levels = SNR_dB(:);
    weights    = exp(snr_levels / 20);
    weights    = weights / sum(weights);
    cum_w      = cumsum(weights);
    r          = rand(D, 1);
    idx_snr    = sum(r > cum_w', 2) + 1;
    idx_snr    = min(idx_snr, numel(snr_levels));
    SNR_vec    = snr_levels(idx_snr);
end
SNR_labels = SNR_vec;

%% Active-element indices via centralised helper
act_lin_idx = get_active_indices(Nv, Nh, Nact, layout);

%% Physical constants
c0     = 3e8;
lambda = c0 / fc;
k0     = 2*pi / lambda;
s      = sqrt(K_users);

%% Pre-build phase-difference geometry (same for every sample)
[row_act, col_act] = ind2sub([Nv, Nh], act_lin_idx);
[ii, jj] = find(tril(ones(Nact), -1));
nPairs   = numel(ii);
A_geom   = zeros(nPairs, 2);
for p = 1:nPairs
    A_geom(p,1) = col_act(ii(p)) - col_act(jj(p));
    A_geom(p,2) = row_act(ii(p)) - row_act(jj(p));
end
nonzero = any(A_geom ~= 0, 2);
A_geom  = A_geom(nonzero,:);
ii      = ii(nonzero);
jj      = jj(nonzero);

%% Pre-allocate  (2*Nact + 5 features)
nFeat      = 2*Nact + 5;
X_raw      = zeros(D, nFeat);
Y_raw      = zeros(D, 2);
loss_weights = zeros(D, 1);

n_feat = log2(N / 16);   % size feature

%% Generate samples
for i = 1:D
    SNR_lin = 10^(SNR_vec(i) / 10);

    dk     = d_k_set(randi(numel(d_k_set)));
    beta_k = beta0 * (dk / d0)^(-alpha_PL);
    theta  = -pi/2 + pi*rand();
    phi_a  = -pi/2 + pi*rand();

    u_true = k0 * d_elem * cos(theta) * cos(phi_a);
    v_true = k0 * d_elem * cos(theta) * sin(phi_a);

    a     = steering_vector_UPA(Nv, Nh, u_true, v_true);
    g     = sqrt(beta_k) * a;
    g_act = g(act_lin_idx);

    noise_var = (norm(g_act)^2 * s^2) / SNR_lin;
    n  = sqrt(noise_var/2) * (randn(size(g_act)) + 1j*randn(size(g_act)));
    y  = g_act*s + n;

    %% Generalized LS-DC estimate
    g_hat = y / s;

    % Weighted LS on all pairs
    phi_vec = zeros(size(A_geom,1), 1);
    for p = 1:size(A_geom,1)
        raw_diff   = angle(g_hat(ii(p))) - angle(g_hat(jj(p)));
        phi_vec(p) = atan2(sin(raw_diff), cos(raw_diff));
    end
    mag_weight = sqrt(abs(g_hat(ii)) .* abs(g_hat(jj)));
    uv   = solve_phase_ls(A_geom, phi_vec, mag_weight);
    u_LS = uv(1);
    v_LS = uv(2);

    % Residual target
    delta_u = atan2(sin(u_true - u_LS), cos(u_true - u_LS));
    delta_v = atan2(sin(v_true - v_LS), cos(v_true - v_LS));

    %% Build feature vector
    py      = norm(y);
    if py < 1e-30; py = 1e-30; end
    y_phase = y / py;

    % SNR estimate from LS-DC fit residual
    a_LS     = steering_vector_UPA(Nv, Nh, u_LS, v_LS);
    a_LS_act = a_LS(act_lin_idx);
    alpha_LS = (a_LS_act' * g_hat) / (a_LS_act' * a_LS_act);
    y_fit    = alpha_LS * a_LS_act * s;
    residual = y - y_fit;
    sig_pow    = norm(y_fit)^2;
    res_pow    = norm(residual)^2;
    if res_pow < 1e-30; res_pow = 1e-30; end
    SNR_est_dB = 10*log10(sig_pow / res_pow);
    SNR_est_dB = max(-10, min(50, SNR_est_dB));

    X_raw(i,:) = [u_LS, v_LS, real(y_phase).', imag(y_phase).', ...
                  log10(py), n_feat, SNR_est_dB];
    Y_raw(i,:) = [delta_u, delta_v];

    % Per-sample loss weight: higher for high-SNR and large N
    loss_weights(i) = exp(SNR_vec(i) / 30) * (N / 16);
end

%% Normalize loss weights
loss_weights = loss_weights / mean(loss_weights);

%% Normalise inputs
X_mean = mean(X_raw, 1);
X_std  = std(X_raw,  0, 1);
X_std(X_std < 1e-10) = 1;
X = (X_raw - X_mean) ./ X_std;

%% Normalise outputs
u_mean = mean(Y_raw(:,1));  u_std = std(Y_raw(:,1));
v_mean = mean(Y_raw(:,2));  v_std = std(Y_raw(:,2));
if u_std < 1e-10; u_std = 1; end
if v_std < 1e-10; v_std = 1; end

Y = zeros(size(Y_raw));
Y(:,1) = (Y_raw(:,1) - u_mean) / u_std;
Y(:,2) = (Y_raw(:,2) - v_mean) / v_std;

fprintf('Dataset ready [layout=%s, N=%d]: %d samples.\n', layout, N, D);
fprintf('Loss weight stats: min=%.2f  max=%.2f  ratio=%.1f:1\n', ...
    min(loss_weights), max(loss_weights), max(loss_weights)/min(loss_weights));

end
