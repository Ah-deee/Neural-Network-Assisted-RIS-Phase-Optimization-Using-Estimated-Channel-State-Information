function uv_hat = nn_predict_angles(net, X_raw, u_mean, u_std, v_mean, v_std, X_mean, X_std, N, Nact, layout)
%NN_PREDICT_ANGLES  Predict spatial frequencies via LS-DC + MLP-NN residual.
%
%   uv_hat = NN_PREDICT_ANGLES(net, X_raw, u_mean, u_std, v_mean, v_std,
%                               X_mean, X_std, N, Nact, layout)
%
%   ADDED PARAMETER:
%     layout : (optional, default 'center') active element placement layout.
%              Must match the layout used during training (nn_dataset_generator)
%              and during channel generation (generate_channel).
%
%   Pipeline (must exactly match nn_dataset_generator):
%     1. Compute generalized LS-DC estimate (u_LS, v_LS) using weighted
%        phase-difference LS across all active element pairs.
%     2. Build feature vector: [u_LS, v_LS, Re(y/||y||)', Im(y/||y||)',
%                               log10(||y||), log2(N/16), SNR_est_dB]
%     3. Z-score normalise with (X_mean, X_std) from training.
%     4. Run network -> normalised [delta_u, delta_v].
%     5. De-normalise and add back to LS-DC:
%           u_hat = u_LS + delta_u,  v_hat = v_LS + delta_v

if nargin < 11 || isempty(layout); layout = 'center'; end
if nargin < 10 || isempty(Nact);   Nact   = 4;        end
if nargin < 9  || isempty(N);      N      = 16;       end

[D, twoNact] = size(X_raw);
if twoNact ~= 2*Nact
    error('nn_predict_angles: X_raw must have 2*Nact=%d columns, got %d.', 2*Nact, twoNact);
end

%% Reconstruct complex y
Re_y = X_raw(:, 1:Nact);
Im_y = X_raw(:, Nact+1:end);

Nv = round(sqrt(N));
Nh = Nv;

%% Active-element indices via centralised helper
act_lin_idx = get_active_indices(Nv, Nh, Nact, layout);
[row_act, col_act] = ind2sub([Nv, Nh], act_lin_idx);

%% Pre-build geometry matrix for generalized LS-DC
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

%% Step 1: Generalized LS-DC estimate for every sample
u_LS = zeros(D, 1);
v_LS = zeros(D, 1);

for i = 1:D
    g_hat = Re_y(i,:).' + 1j*Im_y(i,:).';

    phi_vec = zeros(size(A_geom,1), 1);
    for p = 1:size(A_geom,1)
        raw_diff   = angle(g_hat(ii(p))) - angle(g_hat(jj(p)));
        phi_vec(p) = atan2(sin(raw_diff), cos(raw_diff));
    end
    mag_weight = sqrt(abs(g_hat(ii)) .* abs(g_hat(jj)));
    uv = solve_phase_ls(A_geom, phi_vec, mag_weight);
    u_LS(i) = uv(1);
    v_LS(i) = uv(2);
end

%% Step 2: Build feature vector
py       = sqrt(sum(Re_y.^2 + Im_y.^2, 2));
py(py < 1e-30) = 1e-30;

Re_phase = Re_y ./ py;
Im_phase = Im_y ./ py;
x_power  = log10(py);
n_feat   = log2(N / 16) * ones(D, 1);

% Per-sample SNR estimate from LS-DC fit residual
SNR_est_dB = zeros(D, 1);
for i = 1:D
    y_i = Re_y(i,:).' + 1j*Im_y(i,:).';
    a_LS = steering_vector_UPA(Nv, Nh, u_LS(i), v_LS(i));
    a_LS_act = a_LS(act_lin_idx);

    alpha_LS = (a_LS_act' * y_i) / (a_LS_act' * a_LS_act);
    y_fit    = alpha_LS * a_LS_act;
    residual = y_i - y_fit;

    sig_pow = norm(y_fit)^2;
    res_pow = norm(residual)^2;
    if res_pow < 1e-30; res_pow = 1e-30; end
    SNR_est_dB(i) = 10*log10(sig_pow / res_pow);
    SNR_est_dB(i) = max(-10, min(50, SNR_est_dB(i)));
end

X_feat = [u_LS, v_LS, Re_phase, Im_phase, x_power, n_feat, SNR_est_dB];

%% Step 3: Z-score normalise
X_norm = (X_feat - X_mean) ./ X_std;

%% Step 4: Network inference
delta_norm = predict(net, X_norm);

%% Step 5: De-normalise and add LS-DC correction
delta_u = delta_norm(:,1) * u_std + u_mean;
delta_v = delta_norm(:,2) * v_std + v_mean;

uv_hat       = zeros(D, 2);
uv_hat(:, 1) = u_LS + delta_u;
uv_hat(:, 2) = v_LS + delta_v;

end
