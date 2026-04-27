%% MAIN_CONTRIBUTIONS.M
%   Master script that runs all four novel contributions on top of the
%   baseline RIS channel estimation framework.
%
%   Contributions
%   -------------
%   C1 : Optimal Active Element Layout Selection
%   C2 : RIS Phase Shift Optimisation for Beamforming Gain

%
%   Usage
%   -----
%   1. Make sure all .m files from the baseline framework (generate_channel,
%      ls_dc_estimator, steering_vector_UPA, interpolate_passive_elements,
%      compute_nmse, nn_predict_angles, train_mlp_nn, nn_dataset_generator,
%      prepare_weighted_training_data_v5, gpu_check, log_msg) are on the path.
%   2. Run this script. Set RUN_C1..RUN_C4 flags to selectively run
%      individual contributions.
%   3. Results are saved to 'contribution_results.mat'.

clear; clc; close all;
addpath(fileparts(mfilename('fullpath')));

%% ========= System Parameters (shared with baseline paper) ========= %%
fc        = 30e9;          % Carrier frequency (30 GHz)
d_elem    = 0.5 * 3e8/fc; % Element spacing (lambda/2)
beta0     = 1;             % Reference path-loss at d0
d0        = 1;             % Reference distance (m)
d_k_set   = [30, 50, 80]; % User-RIS distances (m)
alpha_PL  = 2.8;           % Path-loss exponent
K_users   = 4;             % Number of UEs (pilot magnitude = sqrt(K))

N    = 64;   % Total RIS elements (8x8 panel)
Nact = 4;    % Number of active sensing elements (2x2)

SNR_dB_range = [-10, -5, 0, 5, 10, 15, 20, 25, 30];
numMCS       = 500;        % Monte-Carlo samples per SNR point

%% ========= Contribution Flags ========= %%
RUN_C1 = true;
RUN_C2 = true;

NET_FILE = 'trained_nn.mat';
have_nn  = false;
net = []; X_mean=[]; X_std=[]; u_mean_nn=0; u_std_nn=1; v_mean_nn=0; v_std_nn=1;

if exist(NET_FILE,'file')
    log_msg('Loading pre-trained NN from %s', NET_FILE);
    tmp = load(NET_FILE);
    net = tmp.net; X_mean = tmp.X_mean; X_std = tmp.X_std;
    u_mean_nn = tmp.u_mean; u_std_nn = tmp.u_std;
    v_mean_nn = tmp.v_mean; v_std_nn = tmp.v_std;
    have_nn = true;
    log_msg('NN loaded successfully.');
else
    log_msg('No pre-trained NN found. Training a quick baseline NN ...');
    D_train = 15000;
    [X, Y, SNR_labels, u_mean_nn, u_std_nn, v_mean_nn, v_std_nn, X_mean, X_std, ~] = ...
        nn_dataset_generator(D_train, SNR_dB_range, fc, d_elem, beta0, d0, ...
        d_k_set, alpha_PL, K_users, Nact, N);
    [X_tr, Y_tr, X_val, Y_val] = prepare_weighted_training_data_v5(...
        X, Y, SNR_labels, N, 0.85);
    inputDim = size(X_tr, 2);
    layers = train_mlp_nn(inputDim, 128, 3, 'tanh');
    opts = trainingOptions('adam', 'MaxEpochs',60, 'MiniBatchSize',512, ...
        'InitialLearnRate',1e-3, 'ValidationData',{X_val,Y_val}, ...
        'ValidationFrequency',50, 'Verbose',false, 'Plots','none');
    net = trainNetwork(X_tr, Y_tr, layers, opts);
    save(NET_FILE, 'net','X_mean','X_std','u_mean_nn','u_std_nn','v_mean_nn','v_std_nn');
    have_nn = true;
    log_msg('Baseline NN trained and saved.');
end

%% ========= CONTRIBUTION 1: Optimal Layout Selection ========= %%
if RUN_C1
    log_msg('--- Running Contribution 1: Layout Selection ---');
    [C1_best_layout, C1_best_nmse, C1_results] = optimal_layout_selection(...
        N, Nact, fc, d_elem, beta0, d0, d_k_set, alpha_PL, K_users, ...
        SNR_dB_range, numMCS);
    log_msg('C1 Best Layout: %s (mean NMSE=%.2f dB)', C1_best_layout, C1_best_nmse);
end

%% ========= CONTRIBUTION 2: Beamforming Rate ========= %%
if RUN_C2
    log_msg('--- Running Contribution 2: Beamforming Rate ---');
    noise_pow = 1;   % normalised noise power
    if have_nn
        C2_results = ris_beamforming_eval(N, Nact, fc, d_elem, beta0, d0, ...
            d_k_set, alpha_PL, K_users, SNR_dB_range, numMCS, ...
            net, X_mean, X_std, u_mean_nn, u_std_nn, v_mean_nn, v_std_nn, noise_pow);
    else
        C2_results = ris_beamforming_eval(N, Nact, fc, d_elem, beta0, d0, ...
            d_k_set, alpha_PL, K_users, SNR_dB_range, numMCS, ...
            [], [], [], [], [], [], [], noise_pow);
    end
end


%% ========= Summary Table ========= %%
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════╗\n');
fprintf('║               RESULTS SUMMARY                           ║\n');
fprintf('╠══════════════════════════════════════════════════════════╣\n');
if RUN_C1
    fprintf('║  C1 Optimal Layout : %-35s║\n', ...
        sprintf('%s  (%.1f dB mean NMSE)', C1_best_layout, C1_best_nmse));
end
if RUN_C2
    gap_nn   = mean(C2_results.rate_perfect - C2_results.rate_nn);
    gap_lsdc = mean(C2_results.rate_perfect - C2_results.rate_lsdc);
    fprintf('║  C2 Rate gap NN vs Perfect  : %.3f bits/s/Hz          ║\n', gap_nn);
    fprintf('║  C2 Rate gap LSDC vs Perfect: %.3f bits/s/Hz          ║\n', gap_lsdc);
end
if RUN_C3
    fprintf('║  C3 NN advantage over LSDC (L=4, 20dB): %.1f dB       ║\n', ...
        C3_results.nmse_lsdc_L(end) - C3_results.nmse_nn_L(end));
end
if RUN_C4
    aoa_gain = mean(C4_results.nmse_fixed_aoa - C4_results.nmse_adaptive_aoa);
    fprintf('║  C4 Adaptive vs Fixed (mean AoA gain): %.1f dB         ║\n', aoa_gain);
end
fprintf('╚══════════════════════════════════════════════════════════╝\n');

%% Save
save('contribution_results.mat', ...
    'SNR_dB_range','N','Nact','numMCS', ...
    'C1_results','C2_results','C3_results','C4_results','-v7.3');
log_msg('All results saved to contribution_results.mat');
