%% main_simulation_v5_GPU.m

% **v5 GPU-OPTIMIZED VERSION**
%   - Integrates v5 aggressive weighting for Figure 4 (9x replication at 35dB for N=64)
%   - GPU-optimized parameters: batch=512, LR=2e-3, reduced samples for 4-5x speedup
%
% ROBUSTNESS FEATURES:
%   - SIM_MODE: 'fast' (quick validation) or 'slow' 


clear; clc; close all;

%% =====================================================================
%% USER CONTROLS
%% =====================================================================
SIM_MODE      = 'slow';  
FORCE_RESTART = false;     % set true to ignore all checkpoints and start fresh

%% =====================================================================
%% MODE-DEPENDENT PARAMETERS (GPU-OPTIMIZED)
%% =====================================================================
if strcmpi(SIM_MODE, 'fast')
    numMCS       = 500;
    datasetSizes = [2000 5000 10000];
    trainRatios  = [0.4 0.8];
    D_3bcd       = 5000;
    D_train_NN   = 20000;   % GPU-optimized: reduced from 15k (fast mode)
    maxEpochs    = 80;      % GPU-optimized: reduced with higher LR
else
    numMCS       = 10000;   % GPU-optimized: reduced from 20k (still valid)
    datasetSizes = 2e4:2e4:2e5;
    trainRatios  = [0.2 0.4 0.6 0.8];
    D_3bcd       = 8e4;
    D_train_NN   = 50000;   % GPU-optimized: reduced from 80k
    maxEpochs    = 120;     % GPU-optimized: reduced from 200
end

%% Global configuration and RNG
rng(2022,'twister'); % fixed seed for reproducibility

addpath(genpath('functions'));

useGPU = gpu_check();
execEnv = 'cpu';
if useGPU; execEnv = 'gpu'; end

baseDir = fileparts(mfilename('fullpath'));
figDir  = fullfile(baseDir, 'figures');
ckptDir = fullfile(baseDir, 'checkpoints');

if ~exist(figDir,'dir'),  mkdir(figDir);  end
if ~exist(ckptDir,'dir'), mkdir(ckptDir); end

if FORCE_RESTART && exist(ckptDir,'dir')
    log_msg('FORCE_RESTART enabled -- clearing all checkpoints');
    delete(fullfile(ckptDir, '*.mat'));
end

errCount = 0;  % global error counter

log_msg('=== SIMULATION START (%s mode) ===', upper(SIM_MODE));
log_msg('numMCS=%d, maxEpochs=%d, D_train_NN=%d', numMCS, maxEpochs, D_train_NN);

%% System parameters (Table I in the paper)
fc        = 30e9;        % carrier frequency [Hz]
c0        = 3e8;         % speed of light [m/s]
lambda    = c0/fc;
d_elem    = 0.5*lambda;  % inter-element spacing

K_users   = 4;           % number of users
Nact      = 4;           % active elements (2x2 block)

SNR_dB_vec = 0:5:35;     % SNR range [dB] -- used for Fig 3b-d and Fig 4

% Path-loss parameters
beta0_dB  = -20;
beta0     = 10^(beta0_dB/10);
d0        = 1;               % [m]
d_k_set   = [5 10 15 20 25]; % [m]
alpha_PL  = 2.2;

%% NN architecture parameters

numHiddenUnits  = 128;   % 128 units per layer (was 8)
numHiddenLayers = 4;     % 4 hidden layers (was 3)
actFcnHidden    = 'tanh';
% NOTE: nn_dataset_generator now produces 2*Nact+1 input features
% (phase vector + log-power).  train_mlp_nn takes inputDim from
% size(XTrain,2) so no manual change is needed here -- it adapts
% automatically once you retrain with the fixed generator.

% NOTE: make_train_opts is defined as a local function at the bottom of
% this file.

%% ======================================================================
%% FIGURE 3(a): MSE vs Dataset Size
%% ======================================================================
log_msg('========== FIGURE 3(a): MSE vs Dataset Size ==========');
SNR_3a = 20; % dB -- fixed SNR used in the paper for this figure

% Load or initialize checkpoint
ckptFile_3a = fullfile(ckptDir, 'fig3a_progress.mat');
if ~FORCE_RESTART && exist(ckptFile_3a,'file')
    load(ckptFile_3a, 'MSE_3a');
    log_msg('Loaded Fig3a checkpoint');
else
    MSE_3a = nan(numel(trainRatios), numel(datasetSizes));
end

for itRatio = 1:numel(trainRatios)
    trainRatio = trainRatios(itRatio);
    for iD = 1:numel(datasetSizes)
        D = datasetSizes(iD);

        if ~isnan(MSE_3a(itRatio, iD))
            log_msg('[Fig3a] Ratio %.2f, D=%d -- SKIPPED (checkpoint)', trainRatio, D);
            continue;
        end

        log_msg('[Fig3a] Ratio %d/%d (%.2f), Dataset %d/%d (D=%d)', ...
            itRatio, numel(trainRatios), trainRatio, iD, numel(datasetSizes), D);

        try
            log_msg('  -> Generating dataset...');
            % FIX: capture ALL normalization outputs
            % FIX: pass N=16 explicitly (Fig3 fixes N=16 per the paper)
            [X, Y, ~, u_mean, u_std, v_mean, v_std, X_mean, X_std] = ...
                nn_dataset_generator(D, SNR_3a, fc, d_elem, ...
                beta0, d0, d_k_set, alpha_PL, K_users, Nact, 16);

            numTrain = round(trainRatio * D);
            idx      = randperm(D);
            trainIdx = idx(1:numTrain);
            valIdx   = idx(numTrain+1:end);

            XTrain = X(trainIdx, :);  YTrain = Y(trainIdx, :);
            XVal   = X(valIdx,   :);  YVal   = Y(valIdx,   :);

            layers = train_mlp_nn(size(XTrain,2), numHiddenUnits, ...
                                  numHiddenLayers, actFcnHidden);

            opts = make_train_opts(maxEpochs, 512, 1e-3, 30, 0.5, ...
                                   {XVal, YVal}, execEnv);

            log_msg('  -> Training network (%d-unit x %d layers)...', ...
                numHiddenUnits, numHiddenLayers);
            net = trainNetwork(XTrain, YTrain, layers, opts);

            
            u_pred = YPred(:,1)*u_std + u_mean;
            v_pred = YPred(:,2)*v_std + v_mean;
            u_val  = YVal(:,1) *u_std + u_mean;
            v_val  = YVal(:,2) *v_std + v_mean;
            MSE_3a(itRatio, iD) = mean((u_val-u_pred).^2 + (v_val-v_pred).^2);

            log_msg('  -> Angle MSE = %.6f', MSE_3a(itRatio, iD));
            save(ckptFile_3a, 'MSE_3a');

        catch ME
            errCount = errCount + 1;
            log_msg('ERROR in [Fig3a] Ratio=%.2f, D=%d: %s', trainRatio, D, ME.message);
            for kk = 1:numel(ME.stack)
                log_msg('  >> In %s (line %d)', ME.stack(kk).name, ME.stack(kk).line);
            end
            save(ckptFile_3a, 'MSE_3a');
        end
    end
end

% Plot Figure 3(a)
try
    log_msg('Plotting Figure 3(a)...');
    figure;
    markers = {'o-','s-','^-','d-'};
    hold on;
    for itRatio = 1:numel(trainRatios)
        valid = ~isnan(MSE_3a(itRatio,:));
        if any(valid)
            semilogy(datasetSizes(valid), MSE_3a(itRatio,valid), ...
                markers{min(itRatio,4)}, 'LineWidth', 1.4);
        end
    end
    grid on;
    xlabel('|{\itD}|', 'FontName','Times New Roman');
    ylabel('MSE of angle parameters [rad^2]', 'FontName','Times New Roman');
    legendStrs = arrayfun(@(r) sprintf('%.0f%% training', r*100), ...
        trainRatios, 'UniformOutput', false);
    legend(legendStrs{:}, 'Location','best');
    set(gca,'FontName','Times New Roman','FontSize',11,'fontweight','bold');
    print(fullfile(figDir,'fig3a.png'),'-dpng','-r300');
    log_msg('Figure 3(a) saved.');
catch ME
    log_msg('ERROR plotting Fig3a: %s', ME.message);
end

%% ======================================================================
%% FIGURE 3(b): MSE vs SNR -- per-SNR training
%% ======================================================================
log_msg('========== FIGURE 3(b): MSE vs SNR (per-SNR training) ==========');
SNR_grid = SNR_dB_vec;

ckptFile_3b = fullfile(ckptDir, 'fig3b_progress.mat');
if ~FORCE_RESTART && exist(ckptFile_3b,'file')
    load(ckptFile_3b, 'MSE_3b');
    log_msg('Loaded Fig3b checkpoint');
else
    MSE_3b = nan(numel(trainRatios), numel(SNR_grid));
end

for itRatio = 1:numel(trainRatios)
    trainRatio = trainRatios(itRatio);
    for iS = 1:numel(SNR_grid)
        snr_dB = SNR_grid(iS);

        if ~isnan(MSE_3b(itRatio, iS))
            log_msg('[Fig3b] Ratio %.2f, SNR=%ddB -- SKIPPED (checkpoint)', trainRatio, snr_dB);
            continue;
        end

        log_msg('[Fig3b] Ratio %d/%d (%.2f), SNR %d/%d (%ddB)', ...
            itRatio, numel(trainRatios), trainRatio, iS, numel(SNR_grid), snr_dB);

        try
            log_msg('  -> Generating dataset...');
            % FIX: capture normalization params
            % FIX: pass N=16 explicitly (Fig3 fixes N=16 per the paper)
            [X, Y, ~, u_mean, u_std, v_mean, v_std, X_mean, X_std] = ...
                nn_dataset_generator(D_3bcd, snr_dB, fc, d_elem, ...
                beta0, d0, d_k_set, alpha_PL, K_users, Nact, 16);

            numTrain = round(trainRatio * D_3bcd);
            idx      = randperm(D_3bcd);
            trainIdx = idx(1:numTrain);
            valIdx   = idx(numTrain+1:end);

            XTrain = X(trainIdx, :);  YTrain = Y(trainIdx, :);
            XVal   = X(valIdx,   :);  YVal   = Y(valIdx,   :);

            layers = train_mlp_nn(size(XTrain,2), numHiddenUnits, ...
                                  numHiddenLayers, actFcnHidden);

            opts = make_train_opts(maxEpochs, 512, 1e-3, 30, 0.5, ...
                                   {XVal, YVal}, execEnv);

            log_msg('  -> Training network...');
            net = trainNetwork(XTrain, YTrain, layers, opts);

            YPred = predict(net, XVal);
            % Denormalize to angle domain
            u_pred = YPred(:,1)*u_std + u_mean;
            v_pred = YPred(:,2)*v_std + v_mean;
            u_val  = YVal(:,1) *u_std + u_mean;
            v_val  = YVal(:,2) *v_std + v_mean;
            MSE_3b(itRatio, iS) = mean((u_val-u_pred).^2 + (v_val-v_pred).^2);

            log_msg('  -> Angle MSE = %.6f', MSE_3b(itRatio, iS));
            save(ckptFile_3b, 'MSE_3b');

        catch ME
            errCount = errCount + 1;
            log_msg('ERROR in [Fig3b] Ratio=%.2f, SNR=%ddB: %s', trainRatio, snr_dB, ME.message);
            for kk = 1:numel(ME.stack)
                log_msg('  >> In %s (line %d)', ME.stack(kk).name, ME.stack(kk).line);
            end
            save(ckptFile_3b, 'MSE_3b');
        end
    end
end

% Plot Figure 3(b)
try
    log_msg('Plotting Figure 3(b)...');
    figure;
    markers = {'o-','s-','^-','d-'};
    hold on;
    for itRatio = 1:numel(trainRatios)
        valid = ~isnan(MSE_3b(itRatio,:));
        if any(valid)
            semilogy(SNR_grid(valid), MSE_3b(itRatio,valid), ...
                markers{min(itRatio,4)}, 'LineWidth', 1.4);
        end
    end
    grid on;
    xlabel('SNR [dB]', 'FontName','Times New Roman');
    ylabel('MSE of angle parameters [rad^2]', 'FontName','Times New Roman');
    legendStrs = arrayfun(@(r) sprintf('%.0f%%', r*100), trainRatios, 'UniformOutput', false);
    legend(legendStrs{:}, 'Location','southwest');
    set(gca,'FontName','Times New Roman','FontSize',11,'fontweight','bold');
    print(fullfile(figDir,'fig3b.png'),'-dpng','-r300');
    log_msg('Figure 3(b) saved.');
catch ME
    log_msg('ERROR plotting Fig3b: %s', ME.message);
end

%% ======================================================================
%% FIGURE 3(c): MSE vs SNR -- all-SNR training
%% ======================================================================
log_msg('========== FIGURE 3(c): MSE vs SNR (all-SNR training) ==========');

ckptFile_3c = fullfile(ckptDir, 'fig3c_progress.mat');
if ~FORCE_RESTART && exist(ckptFile_3c,'file')
    load(ckptFile_3c, 'MSE_3c');
    log_msg('Loaded Fig3c checkpoint');
else
    MSE_3c = nan(numel(trainRatios), numel(SNR_grid));
end

for itRatio = 1:numel(trainRatios)
    trainRatio = trainRatios(itRatio);

    log_msg('[Fig3c] Ratio %d/%d (%.2f): Generating all-SNR dataset...', ...
        itRatio, numel(trainRatios), trainRatio);

    try
        % FIX: capture all normalization outputs
        % FIX: pass N=16 explicitly (Fig3 fixes N=16 per the paper)
        [X_all, Y_all, SNR_labels_all, u_mean_c, u_std_c, v_mean_c, v_std_c, ...
         X_mean_c, X_std_c] = nn_dataset_generator(D_3bcd, SNR_grid, fc, d_elem, ...
            beta0, d0, d_k_set, alpha_PL, K_users, Nact, 16);

        numTrain = round(trainRatio * D_3bcd);
        idx      = randperm(D_3bcd);
        trainIdx = idx(1:numTrain);
        valIdx   = idx(numTrain+1:end);

        XTrain = X_all(trainIdx, :);  YTrain = Y_all(trainIdx, :);
        XVal   = X_all(valIdx,   :);  YVal   = Y_all(valIdx,   :);

        layers = train_mlp_nn(size(XTrain,2), numHiddenUnits, ...
                              numHiddenLayers, actFcnHidden);

        opts = make_train_opts(maxEpochs, 512, 1e-3, 30, 0.5, ...
                               {XVal, YVal}, execEnv);

        log_msg('  -> Training network on mixed-SNR data...');
        net = trainNetwork(XTrain, YTrain, layers, opts);

        % Evaluate per-SNR on validation set
        for iS = 1:numel(SNR_grid)
            snr_dB = SNR_grid(iS);

            if ~isnan(MSE_3c(itRatio, iS))
                log_msg('[Fig3c] Ratio %.2f, SNR=%ddB -- SKIPPED (checkpoint)', ...
                    trainRatio, snr_dB);
                continue;
            end

            log_msg('[Fig3c] Evaluating at SNR = %ddB...', snr_dB);

            val_at_snr = valIdx(abs(SNR_labels_all(valIdx) - snr_dB) < 0.1);

            if ~isempty(val_at_snr)
                YPred_snr = predict(net, X_all(val_at_snr, :));
                % Denormalize to angle domain
                u_pred = YPred_snr(:,1)*u_std_c + u_mean_c;
                v_pred = YPred_snr(:,2)*v_std_c + v_mean_c;
                u_val  = Y_all(val_at_snr,1)*u_std_c + u_mean_c;
                v_val  = Y_all(val_at_snr,2)*v_std_c + v_mean_c;
                MSE_3c(itRatio, iS) = mean((u_val-u_pred).^2 + (v_val-v_pred).^2);
                log_msg('  -> Angle MSE = %.6f', MSE_3c(itRatio, iS));
            else
                log_msg('  -> WARNING: No validation samples at this SNR');
            end
        end

        save(ckptFile_3c, 'MSE_3c');

    catch ME
        errCount = errCount + 1;
        log_msg('ERROR in [Fig3c] Ratio=%.2f: %s', trainRatio, ME.message);
        for kk = 1:numel(ME.stack)
            log_msg('  >> In %s (line %d)', ME.stack(kk).name, ME.stack(kk).line);
        end
        save(ckptFile_3c, 'MSE_3c');
    end
end

% Plot Figure 3(c)
try
    log_msg('Plotting Figure 3(c)...');
    figure;
    markers = {'o-','s-','^-','d-'};
    hold on;
    for itRatio = 1:numel(trainRatios)
        valid = ~isnan(MSE_3c(itRatio,:));
        if any(valid)
            semilogy(SNR_grid(valid), MSE_3c(itRatio,valid), ...
                markers{min(itRatio,4)}, 'LineWidth', 1.4);
        end
    end
    grid on;
    xlabel('SNR [dB]', 'FontName','Times New Roman');
    ylabel('MSE of angle parameters [rad^2]', 'FontName','Times New Roman');
    legendStrs = arrayfun(@(r) sprintf('%.0f%%', r*100), trainRatios, 'UniformOutput', false);
    legend(legendStrs{:}, 'Location','southwest');
    set(gca,'FontName','Times New Roman','FontSize',11,'fontweight','bold');
    print(fullfile(figDir,'fig3c.png'),'-dpng','-r300');
    log_msg('Figure 3(c) saved.');
catch ME
    log_msg('ERROR plotting Fig3c: %s', ME.message);
end

%% ======================================================================
%% FIGURE 3(d): MSE vs SNR -- per-SNR vs all-SNR comparison
%% ======================================================================
log_msg('========== FIGURE 3(d): Comparison ==========');

try
    log_msg('Plotting Figure 3(d)...');
    figure;
    markers = {'o-','s-','^-','d-'};
    hold on;
    for itRatio = 1:numel(trainRatios)
        valid_3b = ~isnan(MSE_3b(itRatio,:));
        valid_3c = ~isnan(MSE_3c(itRatio,:));
        if any(valid_3b)
            semilogy(SNR_grid(valid_3b), MSE_3b(itRatio,valid_3b), ...
                markers{min(itRatio,4)}, 'LineWidth', 1.4);
        end
        if any(valid_3c)
            semilogy(SNR_grid(valid_3c), MSE_3c(itRatio,valid_3c), ...
                [markers{min(itRatio,4)}(1) '--'], 'LineWidth', 1.4);
        end
    end
    grid on;
    xlabel('SNR [dB]', 'FontName','Times New Roman');
    ylabel('MSE of angle parameters [rad^2]', 'FontName','Times New Roman');
    legEntries = {};
    for itRatio = 1:numel(trainRatios)
        legEntries{end+1} = sprintf('%.0f%% per-SNR', trainRatios(itRatio)*100);
        legEntries{end+1} = sprintf('%.0f%% all-SNR', trainRatios(itRatio)*100);
    end
    legend(legEntries{:}, 'Location','southwest');
    set(gca,'FontName','Times New Roman','FontSize',11,'fontweight','bold');
    print(fullfile(figDir,'fig3d.png'),'-dpng','-r300');
    log_msg('Figure 3(d) saved.');
catch ME
    log_msg('ERROR plotting Fig3d: %s', ME.message);
end

%% ======================================================================
%% FIGURE 4(a-c): NMSE vs SNR, LS-DC vs MLP-NN for different N
%% ======================================================================
log_msg('========== FIGURE 4: NMSE vs SNR ==========');

N_vec = [16 36 64];  % Square UPAs

ckptFile_4 = fullfile(ckptDir, 'fig4_progress.mat');
if ~FORCE_RESTART && exist(ckptFile_4,'file')
    load(ckptFile_4, 'NMSE_LS_dB', 'NMSE_NN_dB');
    log_msg('Loaded Fig4 checkpoint');
else
    NMSE_LS_dB = nan(numel(N_vec), numel(SNR_dB_vec));
    NMSE_NN_dB = nan(numel(N_vec), numel(SNR_dB_vec));
end

% Train ONE MLP-NN per N value using ALL SNRs
for iN = 1:numel(N_vec)
    N = N_vec(iN);

    netFile_fig4 = fullfile(ckptDir, sprintf('fig4_net_N%d_allSNR.mat', N));

    if ~exist(netFile_fig4, 'file')
        try
            log_msg('[Fig4] Training MLP-NN for N=%d with v5 AGGRESSIVE WEIGHTING...', N);

            % Generate base dataset
            log_msg('  -> Generating NN training data (%d samples, all SNRs)...', D_train_NN);
            [Xnn, Ynn, SNR_train, u_mean, u_std, v_mean, v_std, X_mean, X_std] = ...
                nn_dataset_generator(D_train_NN, SNR_dB_vec, fc, d_elem, ...
                beta0, d0, d_k_set, alpha_PL, K_users, Nact, N);

            % v5 WEIGHTED REPLICATION (9x at 35dB for N=64)
            log_msg('  -> Applying v5 weighted training (high-SNR emphasis)...');
            [Xnn_tr, Ynn_tr, Xnn_val, Ynn_val] = prepare_weighted_training_data_v5(...
                Xnn, Ynn, SNR_train, N, 0.85);

            log_msg('  -> Training samples after v5 weighting: %d (%.1fx)', ...
                size(Xnn_tr,1), size(Xnn_tr,1)/D_train_NN);

            % Build network
            layersNN = train_mlp_nn(size(Xnn,2), numHiddenUnits, ...
                                    numHiddenLayers, actFcnHidden);

            % GPU-OPTIMIZED: batch 512, LR 2e-3, drop every 30 epochs
            optsNN = make_train_opts(maxEpochs, 512, 2e-3, 30, 0.5, ...
                {Xnn_val, Ynn_val}, execEnv);

            log_msg('  -> Training MLP-NN (128x4, %d epochs, batch=512, LR=2e-3)...', maxEpochs);
            netNN = trainNetwork(Xnn_tr, Ynn_tr, layersNN, optsNN);

            % Save network WITH all normalization params (needed at inference)
            save(netFile_fig4, 'netNN', ...
                'u_mean','u_std','v_mean','v_std','X_mean','X_std');
            log_msg('  -> Network saved to %s', netFile_fig4);

        catch ME
            errCount = errCount + 1;
            log_msg('ERROR training network for Fig4, N=%d: %s', N, ME.message);
            for kk = 1:numel(ME.stack)
                log_msg('  >> In %s (line %d)', ME.stack(kk).name, ME.stack(kk).line);
            end
            log_msg('Skipping evaluation for N=%d.', N);
            continue;
        end
    else
        log_msg('[Fig4] Loading pre-trained network for N=%d', N);
        load(netFile_fig4, 'netNN', ...
            'u_mean','u_std','v_mean','v_std','X_mean','X_std');
    end

    % Evaluate at each SNR point
    for iS = 1:numel(SNR_dB_vec)
        snr_dB = SNR_dB_vec(iS);

        if ~isnan(NMSE_LS_dB(iN, iS))
            log_msg('[Fig4] N=%d, SNR=%ddB -- SKIPPED (checkpoint)', N, snr_dB);
            continue;
        end

        log_msg('[Fig4] N=%d (%d/%d), SNR=%ddB (%d/%d)', ...
            N, iN, numel(N_vec), snr_dB, iS, numel(SNR_dB_vec));

        try
            log_msg('  -> Generating %d MC channels...', numMCS);
            [H_full, y_act, u_true, v_true] = generate_channel(numMCS, N, Nact, ...
                fc, d_elem, beta0, d0, d_k_set, alpha_PL, K_users, snr_dB);

            % ----- LS-DC estimator -----
            log_msg('  -> Running LS-DC estimator...');
            [~, ~, H_hat_LS] = ls_dc_estimator(y_act, N, Nact, fc, d_elem, K_users);
            nmse_ls = compute_nmse(H_full, H_hat_LS);

            % ----- NN estimator -----
            log_msg('  -> Running NN inference + interpolation...');
            % Build raw [Re Im] matrix -- nn_predict_angles applies
            % the per-sample L2 normalisation + log-power feature
            % construction + global z-score step internally.
            % Do NOT pre-normalise here.
            X_test  = [real(y_act) imag(y_act)];
            uv_hat  = nn_predict_angles(netNN, X_test, ...
                u_mean, u_std, v_mean, v_std, X_mean, X_std, N, Nact);

            H_hat_NN = interpolate_passive_elements(uv_hat(:,1), uv_hat(:,2), ...
                N, Nact, fc, d_elem, y_act, K_users);
            nmse_nn = compute_nmse(H_full, H_hat_NN);

            NMSE_LS_dB(iN, iS) = 10*log10(nmse_ls);
            NMSE_NN_dB(iN, iS) = 10*log10(nmse_nn);

            log_msg('  -> LS-DC NMSE = %.2f dB, NN NMSE = %.2f dB', ...
                NMSE_LS_dB(iN,iS), NMSE_NN_dB(iN,iS));

            save(ckptFile_4, 'NMSE_LS_dB', 'NMSE_NN_dB');

        catch ME
            errCount = errCount + 1;
            log_msg('ERROR in [Fig4] N=%d, SNR=%ddB: %s', N, snr_dB, ME.message);
            for kk = 1:numel(ME.stack)
                log_msg('  >> In %s (line %d)', ME.stack(kk).name, ME.stack(kk).line);
            end
            save(ckptFile_4, 'NMSE_LS_dB', 'NMSE_NN_dB');
        end
    end
end

% Plot Figure 4 -- combined panel
try
    log_msg('Plotting Figure 4 (combined)...');
    figure; hold on;
    colors     = {'b','r','m'};
    lineStyles = {'-','--','-.'};
    for iN = 1:numel(N_vec)
        validLS = ~isnan(NMSE_LS_dB(iN,:));
        validNN = ~isnan(NMSE_NN_dB(iN,:));
        if any(validLS)
            plot(SNR_dB_vec(validLS), NMSE_LS_dB(iN,validLS), ...
                ['b o' lineStyles{iN}], 'LineWidth', 1.4, 'Color', 'b');
        end
        if any(validNN)
            plot(SNR_dB_vec(validNN), NMSE_NN_dB(iN,validNN), ...
                ['r ^' lineStyles{iN}], 'LineWidth', 1.4, 'Color', 'r');
        end
    end
    grid on;
    xlabel('SNR [dB]', 'FontName','Times New Roman');
    ylabel('NMSE [dB]', 'FontName','Times New Roman');
    legEntries = {};
    for iN = 1:numel(N_vec)
        legEntries{end+1} = sprintf('LS-DC + interpolation, N=%d', N_vec(iN));
        legEntries{end+1} = sprintf('MLP-NN + interpolation, N=%d', N_vec(iN));
    end
    legend(legEntries{:}, 'Location','southwest');
    set(gca,'FontName','Times New Roman','FontSize',11,'fontweight','bold');
    print(fullfile(figDir,'fig4_all.png'),'-dpng','-r300');
    log_msg('Figure 4 (combined) saved.');
catch ME
    log_msg('ERROR plotting Fig4 combined: %s', ME.message);
end

% Plot separate panels for each N
panelLabels = {'a','b','c'};
for iN = 1:numel(N_vec)
    try
        N = N_vec(iN);
        figure;
        validLS = ~isnan(NMSE_LS_dB(iN,:));
        validNN = ~isnan(NMSE_NN_dB(iN,:));
        if any(validLS)
            plot(SNR_dB_vec(validLS), NMSE_LS_dB(iN,validLS), ...
                'bo-', 'LineWidth', 1.4); hold on;
        end
        if any(validNN)
            plot(SNR_dB_vec(validNN), NMSE_NN_dB(iN,validNN), ...
                'r^-', 'LineWidth', 1.4);
        end
        grid on;
        xlabel('SNR [dB]', 'FontName','Times New Roman');
        ylabel('NMSE [dB]', 'FontName','Times New Roman');
        title(sprintf('(%s) N = %d, N_{act} = %d', panelLabels{iN}, N, Nact));
        legend('LS-DC + interpolation','MLP-NN + interpolation', ...
            'Location','southwest');
        set(gca,'FontName','Times New Roman','FontSize',11,'fontweight','bold');
        print(fullfile(figDir, sprintf('fig4%s.png', panelLabels{iN})), ...
            '-dpng','-r300');
        log_msg('Figure 4(%s) saved.', panelLabels{iN});
    catch ME
        log_msg('ERROR plotting Fig4(%s): %s', panelLabels{iN}, ME.message);
    end
end

%% ======================================================================
%% SUMMARY
%% ======================================================================
log_msg('=== SIMULATION COMPLETE ===');
log_msg('Mode: %s', upper(SIM_MODE));
log_msg('Errors encountered: %d', errCount);
if errCount > 0
    log_msg('Check NaN entries in result matrices for failed iterations.');
    log_msg('Fix the bug, then re-run -- completed iterations will be skipped.');
end
log_msg('Figures saved in: %s', figDir);
log_msg('Checkpoints saved in: %s', ckptDir);

disp(' ');
disp('All simulations completed. Figures saved in the "figures" folder.');



function opts = make_train_opts(maxEpochs, batchSz, lr, dropPer, dropFac, ...
                                valData, execEnv)
%MAKE_TRAIN_OPTS  This helper is called
%   by every training block so the schedule is always applied consistently.
    opts = trainingOptions('adam', ...
        'MaxEpochs',           maxEpochs, ...
        'MiniBatchSize',       batchSz, ...
        'InitialLearnRate',    lr, ...
        'LearnRateSchedule',   'piecewise', ...
        'LearnRateDropPeriod', dropPer, ...
        'LearnRateDropFactor', dropFac, ...
        'Shuffle',             'every-epoch', ...
        'ValidationData',      valData, ...
        'ValidationFrequency', 50, ...
        'Verbose',             false, ...
        'Plots',               'none', ...
        'ExecutionEnvironment', execEnv);
end
