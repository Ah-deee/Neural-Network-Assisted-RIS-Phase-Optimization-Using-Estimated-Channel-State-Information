function results = ris_beamforming_eval(...
    N, Nact, fc, d_elem, beta0, d0, d_k_set, alpha_PL, K_users, SNR_dB_range, ...
    numMCS, net, X_mean, X_std, u_mean_nn, u_std_nn, v_mean_nn, v_std_nn, noise_pow)
%RIS_BEAMFORMING_EVAL  Contribution 2: RIS phase-shift optimisation and
%   achievable-rate comparison across three CSI scenarios.
%
%   Scenarios evaluated
%   -------------------
%   1. Perfect CSI      : phase shifts set from true channel (upper bound).
%   2. NN-estimated CSI : phase shifts derived from the MLP-NN estimator.
%   3. LS-DC CSI        : phase shifts derived from the baseline LS-DC.
%
%   Beamforming rule (closed-form, single-user)
%   -------------------------------------------
%       theta_n = -angle(g_n)            for RIS-AP channel g
%   This aligns all N reflected paths coherently at the receiver.
%
%   Achievable rate
%   ---------------
%       R = log2(1 + |sum_n g_n * exp(j*theta_n)|^2 / noise_pow)   [bits/s/Hz]
%
%   Parameters
%   ----------
%   noise_pow  : receiver noise power (linear). Default 1.
%   net, X_mean, X_std, u_mean_nn, u_std_nn, v_mean_nn, v_std_nn :
%                parameters from a pre-trained MLP-NN (see train_mlp_nn /
%                nn_predict_angles).  Pass [] to skip NN scenario.
%
%   Output
%   ------
%   results : struct with fields rate_perfect, rate_nn, rate_lsdc (nSNR x 1).

if nargin < 13 || isempty(noise_pow); noise_pow = 1; end

Nv = round(sqrt(N));
nSNR = numel(SNR_dB_range);
have_nn = ~isempty(net);

rate_perfect = zeros(nSNR,1);
rate_nn      = zeros(nSNR,1);
rate_lsdc    = zeros(nSNR,1);

fprintf('\n=== Contribution 2: Beamforming Rate Evaluation ===\n');

for si = 1:nSNR
    snr = SNR_dB_range(si);

    [H_full, y_act, u_true, v_true] = generate_channel(numMCS, N, Nact, ...
        fc, d_elem, beta0, d0, d_k_set, alpha_PL, K_users, snr);

    % ------------------------------------------------------------------
    % 1. Perfect CSI
    % ------------------------------------------------------------------
    rate_perfect(si) = mean(compute_rate(H_full, H_full, noise_pow));

    % ------------------------------------------------------------------
    % 2. LS-DC CSI
    % ------------------------------------------------------------------
    [u_lsdc, v_lsdc, H_lsdc] = ls_dc_estimator(y_act, N, Nact, fc, d_elem, K_users);
    rate_lsdc(si) = mean(compute_rate(H_full, H_lsdc, noise_pow));

    % ------------------------------------------------------------------
    % 3. NN-estimated CSI
    % ------------------------------------------------------------------
    if have_nn
        X_raw = [real(y_act), imag(y_act)];
        uv_nn = nn_predict_angles(net, X_raw, u_mean_nn, u_std_nn, ...
            v_mean_nn, v_std_nn, X_mean, X_std, N, Nact);
        H_nn = interpolate_passive_elements(uv_nn(:,1), uv_nn(:,2), ...
            N, Nact, fc, d_elem, y_act, K_users);
        rate_nn(si) = mean(compute_rate(H_full, H_nn, noise_pow));
    end

    if have_nn
        fprintf('  SNR=%+3d dB | Perfect=%.2f  NN=%.2f  LS-DC=%.2f  bits/s/Hz\n', ...
            snr, rate_perfect(si), rate_nn(si), rate_lsdc(si));
    else
        fprintf('  SNR=%+3d dB | Perfect=%.2f  LS-DC=%.2f  bits/s/Hz\n', ...
            snr, rate_perfect(si), rate_lsdc(si));
    end
end

%% Plot
figure('Name','Contribution 2 - Achievable Rate vs SNR');
hold on; grid on;
plot(SNR_dB_range, rate_perfect, 'k-o', 'LineWidth',2, 'MarkerSize',8, ...
    'DisplayName','Perfect CSI (upper bound)');
if have_nn
    plot(SNR_dB_range, rate_nn, 'b-s', 'LineWidth',2, 'MarkerSize',8, ...
        'DisplayName','NN-estimated CSI (proposed)');
end
plot(SNR_dB_range, rate_lsdc, 'r--^', 'LineWidth',2, 'MarkerSize',8, ...
    'DisplayName','LS-DC CSI (baseline)');
xlabel('SNR (dB)'); ylabel('Achievable Rate (bits/s/Hz)');
title(sprintf('Contribution 2: RIS Beamforming Rate  |  N=%d, Nact=%d', N, Nact));
legend('Location','northwest'); set(gca,'FontSize',12);
hold off;

%% Rate gap analysis
fprintf('\n--- Rate Gap (Perfect CSI vs Estimators) ---\n');
if have_nn
    fprintf('  Mean gap  NN   : %.3f bits/s/Hz\n', mean(rate_perfect - rate_nn));
end
fprintf('  Mean gap LS-DC : %.3f bits/s/Hz\n', mean(rate_perfect - rate_lsdc));

results.SNR_range    = SNR_dB_range;
results.rate_perfect = rate_perfect;
results.rate_nn      = rate_nn;
results.rate_lsdc    = rate_lsdc;
end

% =========================================================================
%  LOCAL HELPER
% =========================================================================

function R = compute_rate(H_true, H_hat, noise_pow)
%COMPUTE_RATE  Achievable rate for each Monte-Carlo sample.
%
%   For each trial i:
%     theta_n = -angle(H_hat(i,n))          (phase compensation)
%     received_signal_power = |H_true(i,:) * exp(j*theta)|^2
%     R(i) = log2(1 + received_signal_power / noise_pow)

numMCS = size(H_true, 1);
R = zeros(numMCS, 1);

for i = 1:numMCS
    g_true = H_true(i,:).';   % N x 1 true channel
    g_hat  = H_hat(i,:).';    % N x 1 estimated channel

    % Optimal phase shifts based on estimated channel
    theta = -angle(g_hat);    % N x 1

    % Received signal power using TRUE channel
    eff_gain = abs(g_true.' * exp(1j*theta))^2;  % coherent combining

    R(i) = log2(1 + eff_gain / noise_pow);
end
end
