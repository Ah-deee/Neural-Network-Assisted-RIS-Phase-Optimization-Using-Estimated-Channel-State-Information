%% fig5_placement_comparison.m
% FIGURE 5: Active Element Placement Optimization
%
% Compares six active-element layouts on a 4x4 (N=16) RIS:
%   center | corners | cross | edges | diagonal | random
%
% TWO ESTIMATORS evaluated vs SNR:
%   1. LS-DC  : generalized two-stage phase-unwrapping LS (solve_phase_ls)
%   2. MLP-NN : trained per layout, predicts residual correction on LS-DC
%

% RESEARCH FINDING:
%   For LS-DC, center is optimal at full angle range because baseline=1
%   avoids wrapping ambiguity. The NN can mitigate wrapping via raw phase
%   features, so corners/diagonal may improve under NN estimation.
%
% FIX (v2): 'random' layout is now averaged over numRandomTrials seeds so
%   the curve reflects the *expected* performance of random placement, not
%   one lucky fixed draw.

clear; clc; close all;

%% ===================================================================
%% USER CONTROLS
%% ===================================================================
SIM_MODE        = 'slow';
FORCE_RESTART   = true;
numRandomTrials = 20;   % <-- number of random seeds to average over

if strcmpi(SIM_MODE,'fast')
    numMCS     = 500;
    D_train_NN = 15000;
    maxEpochs  = 80;
else
    numMCS     = 5000;
    D_train_NN = 40000;
    maxEpochs  = 120;
end

rng(0,'twister');   % master seed (no longer drives the random layout)
addpath(genpath('functions'));

useGPU  = gpu_check();
execEnv = 'cpu';
if useGPU; execEnv = 'gpu'; end

baseDir = fileparts(mfilename('fullpath'));
figDir  = fullfile(baseDir,'figures');
ckptDir = fullfile(baseDir,'checkpoints');
if ~exist(figDir,'dir'),  mkdir(figDir);  end
if ~exist(ckptDir,'dir'), mkdir(ckptDir); end

log_msg('=== FIG5: PLACEMENT COMPARISON (%s mode) ===', upper(SIM_MODE));

%% ===================================================================
%% SYSTEM PARAMETERS
%% ===================================================================
fc        = 30e9;
c0        = 3e8;
lambda    = c0/fc;
d_elem    = 0.5*lambda;
K_users   = 4;
Nact      = 4;
N         = 16;
SNR_dB_vec = 0:5:35;
beta0_dB  = -20;
beta0     = 10^(beta0_dB/10);
d0        = 1;
d_k_set   = [5 10 15 20 25];
alpha_PL  = 2.2;
numHiddenUnits  = 128;
numHiddenLayers = 4;

%% ===================================================================
%% LAYOUTS
%% ===================================================================
layouts      = {'center','corners','cross','edges','diagonal','random'};
layoutLabels = {'Center (baseline)','Corners','Cross','Edges','Diagonal','Random (avg)'};
nLayouts     = numel(layouts);

%% ===================================================================
%% CHECKPOINT
%% ===================================================================
ckptFile = fullfile(ckptDir,'fig5_progress.mat');
if ~FORCE_RESTART && exist(ckptFile,'file')
    load(ckptFile,'NMSE_LS_dB','NMSE_NN_dB');
    log_msg('Loaded Fig5 checkpoint.');
else
    NMSE_LS_dB = nan(nLayouts, numel(SNR_dB_vec));
    NMSE_NN_dB = nan(nLayouts, numel(SNR_dB_vec));
end

%% ===================================================================
%% MAIN LOOP
%% ===================================================================
for iL = 1:nLayouts
    layout = layouts{iL};
    log_msg('---------- Layout %d/%d: %s ----------', iL, nLayouts, layout);

    isRandom = strcmpi(layout, 'random');

    % ----------------------------------------------------------------
    % RANDOM LAYOUT: average over numRandomTrials independent seeds
    % ----------------------------------------------------------------
    if isRandom
        log_msg('[random] Averaging over %d random seeds...', numRandomTrials);

        NMSE_rand_LS_trials = nan(numRandomTrials, numel(SNR_dB_vec));
        NMSE_rand_NN_trials = nan(numRandomTrials, numel(SNR_dB_vec));

        for trial = 1:numRandomTrials
            rng(trial, 'twister');   % each trial gets its own reproducible seed
            log_msg('[random] Trial %d/%d', trial, numRandomTrials);

            % Train one NN per random trial (layout geometry changes with seed)
            netFile_rand = fullfile(ckptDir, ...
                sprintf('fig5_net_random_trial%d_N%d.mat', trial, N));

            if ~exist(netFile_rand, 'file')
                try
                    [Xnn, Ynn, ~, u_mean, u_std, v_mean, v_std, X_mean, X_std] = ...
                        nn_dataset_generator(D_train_NN, SNR_dB_vec, fc, d_elem, ...
                            beta0, d0, d_k_set, alpha_PL, K_users, Nact, N, layout);

                    numTrain = round(0.85 * D_train_NN);
                    idx      = randperm(D_train_NN);
                    Xnn_tr   = Xnn(idx(1:numTrain),:);
                    Ynn_tr   = Ynn(idx(1:numTrain),:);
                    Xnn_val  = Xnn(idx(numTrain+1:end),:);
                    Ynn_val  = Ynn(idx(numTrain+1:end),:);

                    layersNN = train_mlp_nn(size(Xnn_tr,2), numHiddenUnits, ...
                                            numHiddenLayers, 'tanh');
                    opts = make_train_opts(maxEpochs, 256, 1e-3, 30, 0.5, ...
                                           {Xnn_val, Ynn_val}, execEnv);
                    netNN_rand = trainNetwork(Xnn_tr, Ynn_tr, layersNN, opts);

                    save(netFile_rand, 'netNN_rand', ...
                        'u_mean','u_std','v_mean','v_std','X_mean','X_std');
                    log_msg('[random] Trial %d network saved.', trial);
                catch ME
                    log_msg('ERROR training NN [random trial %d]: %s', trial, ME.message);
                    continue;
                end
            else
                load(netFile_rand, 'netNN_rand', ...
                    'u_mean','u_std','v_mean','v_std','X_mean','X_std');
            end

            % Evaluate each SNR for this trial
            for iS = 1:numel(SNR_dB_vec)
                snr_dB = SNR_dB_vec(iS);
                try
                    [H_full, y_act, ~, ~] = generate_channel(numMCS, N, Nact, ...
                        fc, d_elem, beta0, d0, d_k_set, alpha_PL, K_users, snr_dB, layout);

                    % LS-DC
                    [~, ~, H_hat_LS] = ls_dc_estimator(y_act, N, Nact, ...
                        fc, d_elem, K_users, layout);
                    NMSE_rand_LS_trials(trial, iS) = ...
                        10*log10(compute_nmse(H_full, H_hat_LS));

                    % MLP-NN
                    X_test = [real(y_act), imag(y_act)];
                    uv_hat = nn_predict_angles(netNN_rand, X_test, ...
                        u_mean, u_std, v_mean, v_std, X_mean, X_std, N, Nact, layout);
                    H_hat_NN = interpolate_passive_elements(uv_hat(:,1), uv_hat(:,2), ...
                        N, Nact, fc, d_elem, y_act, K_users, layout);
                    NMSE_rand_NN_trials(trial, iS) = ...
                        10*log10(compute_nmse(H_full, H_hat_NN));

                catch ME
                    log_msg('ERROR [random trial %d] SNR=%ddB: %s', trial, snr_dB, ME.message);
                end
            end

            clear netNN_rand;
        end % trial loop

        % Average across trials (ignore any NaN trials)
        NMSE_LS_dB(iL,:) = mean(NMSE_rand_LS_trials, 1, 'omitnan');
        NMSE_NN_dB(iL,:) = mean(NMSE_rand_NN_trials, 1, 'omitnan');
        save(ckptFile, 'NMSE_LS_dB', 'NMSE_NN_dB');
        log_msg('[random] Averaged LS-DC = %.2f dB  NN = %.2f dB  (at max SNR)', ...
            NMSE_LS_dB(iL,end), NMSE_NN_dB(iL,end));
        continue;   % skip the deterministic layout block below
    end

    % ----------------------------------------------------------------
    % DETERMINISTIC LAYOUTS (center, corners, cross, edges, diagonal)
    % ----------------------------------------------------------------
    netFile = fullfile(ckptDir, sprintf('fig5_net_%s_N%d.mat', layout, N));

    %% --- Train MLP-NN ---
    if ~exist(netFile,'file')
        try
            log_msg('[%s] Generating %d training samples...', layout, D_train_NN);
            [Xnn, Ynn, ~, u_mean, u_std, v_mean, v_std, X_mean, X_std] = ...
                nn_dataset_generator(D_train_NN, SNR_dB_vec, fc, d_elem, ...
                    beta0, d0, d_k_set, alpha_PL, K_users, Nact, N, layout);

            % Simple 85/15 random split -- stable, balanced across SNRs
            numTrain = round(0.85 * D_train_NN);
            idx      = randperm(D_train_NN);
            Xnn_tr   = Xnn(idx(1:numTrain),:);
            Ynn_tr   = Ynn(idx(1:numTrain),:);
            Xnn_val  = Xnn(idx(numTrain+1:end),:);
            Ynn_val  = Ynn(idx(numTrain+1:end),:);

            log_msg('[%s] Training MLP-NN (%dx%d, %d epochs, batch=256, LR=1e-3)...', ...
                layout, numHiddenUnits, numHiddenLayers, maxEpochs);

            layersNN = train_mlp_nn(size(Xnn_tr,2), numHiddenUnits, ...
                                    numHiddenLayers, 'tanh');

            opts = make_train_opts(maxEpochs, 256, 1e-3, 30, 0.5, ...
                                   {Xnn_val, Ynn_val}, execEnv);

            netNN = trainNetwork(Xnn_tr, Ynn_tr, layersNN, opts);

            save(netFile,'netNN','u_mean','u_std','v_mean','v_std','X_mean','X_std');
            log_msg('[%s] Network saved.', layout);

        catch ME
            log_msg('ERROR training NN [%s]: %s', layout, ME.message);
            for kk = 1:numel(ME.stack)
                log_msg('  >> %s (line %d)', ME.stack(kk).name, ME.stack(kk).line);
            end
        end
    else
        log_msg('[%s] Loading pre-trained network...', layout);
        load(netFile,'netNN','u_mean','u_std','v_mean','v_std','X_mean','X_std');
    end

    %% --- Evaluate ---
    for iS = 1:numel(SNR_dB_vec)
        snr_dB = SNR_dB_vec(iS);

        if ~isnan(NMSE_LS_dB(iL,iS)) && ~isnan(NMSE_NN_dB(iL,iS))
            log_msg('[%s] SNR=%ddB -- SKIPPED', layout, snr_dB);
            continue;
        end

        log_msg('[%s] SNR = %d dB (%d/%d)...', layout, snr_dB, iS, numel(SNR_dB_vec));

        try
            [H_full, y_act, ~, ~] = generate_channel(numMCS, N, Nact, ...
                fc, d_elem, beta0, d0, d_k_set, alpha_PL, K_users, snr_dB, layout);

            % LS-DC
            if isnan(NMSE_LS_dB(iL,iS))
                [~, ~, H_hat_LS] = ls_dc_estimator(y_act, N, Nact, ...
                    fc, d_elem, K_users, layout);
                NMSE_LS_dB(iL,iS) = 10*log10(compute_nmse(H_full, H_hat_LS));
            end

            % MLP-NN (only if network exists)
            if isnan(NMSE_NN_dB(iL,iS)) && exist('netNN','var')
                X_test   = [real(y_act), imag(y_act)];
                uv_hat   = nn_predict_angles(netNN, X_test, ...
                    u_mean, u_std, v_mean, v_std, X_mean, X_std, N, Nact, layout);
                H_hat_NN = interpolate_passive_elements(uv_hat(:,1), uv_hat(:,2), ...
                    N, Nact, fc, d_elem, y_act, K_users, layout);
                NMSE_NN_dB(iL,iS) = 10*log10(compute_nmse(H_full, H_hat_NN));
            end

            log_msg('  LS-DC = %.2f dB   NN = %.2f dB', ...
                NMSE_LS_dB(iL,iS), NMSE_NN_dB(iL,iS));

            save(ckptFile,'NMSE_LS_dB','NMSE_NN_dB');

        catch ME
            log_msg('ERROR [%s] SNR=%ddB: %s', layout, snr_dB, ME.message);
            save(ckptFile,'NMSE_LS_dB','NMSE_NN_dB');
        end
    end

    clear netNN;  % always clear before next layout
end

%% ===================================================================
%% PLOTTING
%% ===================================================================
colors  = lines(nLayouts);
mks     = {'o','s','^','d','p','h'};

%% Fig5a: LS-DC
try
    figure('Position',[50 100 700 500]); hold on;
    hL = gobjects(nLayouts,1);
    for iL = 1:nLayouts
        v = ~isnan(NMSE_LS_dB(iL,:));
        if any(v)
            hL(iL) = plot(SNR_dB_vec(v), NMSE_LS_dB(iL,v), ...
                [mks{iL} '-'],'Color',colors(iL,:),'LineWidth',1.6, ...
                'MarkerSize',7,'DisplayName',layoutLabels{iL});
        end
    end
    grid on;
    xlabel('SNR [dB]','FontName','Times New Roman','FontSize',12);
    ylabel('NMSE [dB]','FontName','Times New Roman','FontSize',12);
    title('Active Element Placement: LS-DC (N=16)','FontName','Times New Roman','FontSize',11);
    valid_h = arrayfun(@(h) isgraphics(h,'line'), hL);
    legend(hL(valid_h),'Location','northeast','FontName','Times New Roman','FontSize',9);
    set(gca,'FontName','Times New Roman','FontSize',11,'fontweight','bold');
    xlim([SNR_dB_vec(1)-1, SNR_dB_vec(end)+1]);
    print(fullfile(figDir,'fig5a_placement_LSDC.png'),'-dpng','-r300');
    log_msg('Fig5a saved.');
catch ME; log_msg('ERROR Fig5a: %s', ME.message); end

%% Fig5b: MLP-NN
try
    figure('Position',[50 100 700 500]); hold on;
    hL = gobjects(nLayouts,1);
    for iL = 1:nLayouts
        v = ~isnan(NMSE_NN_dB(iL,:));
        if any(v)
            hL(iL) = plot(SNR_dB_vec(v), NMSE_NN_dB(iL,v), ...
                [mks{iL} '--'],'Color',colors(iL,:),'LineWidth',1.6, ...
                'MarkerSize',7,'DisplayName',layoutLabels{iL});
        end
    end
    grid on;
    xlabel('SNR [dB]','FontName','Times New Roman','FontSize',12);
    ylabel('NMSE [dB]','FontName','Times New Roman','FontSize',12);
    title('Active Element Placement: MLP-NN (N=16)','FontName','Times New Roman','FontSize',11);
    valid_h = arrayfun(@(h) isgraphics(h,'line'), hL);
    legend(hL(valid_h),'Location','northeast','FontName','Times New Roman','FontSize',9);
    set(gca,'FontName','Times New Roman','FontSize',11,'fontweight','bold');
    xlim([SNR_dB_vec(1)-1, SNR_dB_vec(end)+1]);
    print(fullfile(figDir,'fig5b_placement_NN.png'),'-dpng','-r300');
    log_msg('Fig5b saved.');
catch ME; log_msg('ERROR Fig5b: %s', ME.message); end

%% Fig5c: Gain bar chart
try
    [~,iSnrTop] = max(SNR_dB_vec);
    gain_ls = NMSE_LS_dB(1,iSnrTop) - NMSE_LS_dB(:,iSnrTop);
    gain_nn = NMSE_NN_dB(1,iSnrTop) - NMSE_NN_dB(:,iSnrTop);

    figure('Position',[50 100 800 420]);
    b1 = bar((1:nLayouts)-0.2, gain_ls, 0.35, 'FaceColor',[0.2 0.5 0.8]);
    hold on;
    b2 = bar((1:nLayouts)+0.2, gain_nn, 0.35, 'FaceColor',[0.9 0.4 0.2]);
    yline(0,'k--','LineWidth',1.2);
    grid on;
    set(gca,'XTick',1:nLayouts,'XTickLabel',layoutLabels, ...
        'FontName','Times New Roman','FontSize',10,'fontweight','bold');
    xtickangle(15);
    xlabel('Layout','FontName','Times New Roman','FontSize',12);
    ylabel('NMSE gain over Center [dB]','FontName','Times New Roman','FontSize',12);
    title(sprintf('Placement gain at SNR=%d dB, N=%d', SNR_dB_vec(iSnrTop), N), ...
        'FontName','Times New Roman','FontSize',11);
    legend([b1,b2],{'LS-DC','MLP-NN'},'Location','best', ...
        'FontName','Times New Roman','FontSize',10);
    print(fullfile(figDir,'fig5c_placement_gain_bar.png'),'-dpng','-r300');
    log_msg('Fig5c saved.');
catch ME; log_msg('ERROR Fig5c: %s', ME.message); end

%% Fig5d: Geometry
try
    Nv_d = round(sqrt(N));
    figure('Position',[50 100 1100 340]);
    tl = tiledlayout(1,nLayouts,'TileSpacing','compact','Padding','compact');
    for iL = 1:nLayouts
        nexttile;
        % For the random layout, show seed=1 as a representative example
        if strcmpi(layouts{iL},'random'); rng(1,'twister'); end
        idx = get_active_indices(Nv_d, Nv_d, Nact, layouts{iL});
        [rv,cv] = ind2sub([Nv_d,Nv_d], idx);
        [ac,ar] = meshgrid(1:Nv_d, 1:Nv_d);
        scatter(ac(:), Nv_d+1-ar(:), 160, [0.85 0.85 0.85],'o','filled');
        hold on;
        scatter(cv, Nv_d+1-rv, 200, colors(iL,:),'s','filled', ...
            'MarkerEdgeColor','k','LineWidth',1.1);
        axis equal tight;
        set(gca,'XTick',1:Nv_d,'YTick',1:Nv_d,'XTickLabel',{},'YTickLabel',{}, ...
            'XGrid','on','YGrid','on','GridColor',[0.7 0.7 0.7],'fontweight','bold');
        xlim([0.4 Nv_d+0.6]); ylim([0.4 Nv_d+0.6]);
        title(layoutLabels{iL},'FontName','Times New Roman','FontSize',10);
        box on;
    end
    title(tl, sprintf('Active element layouts (N=%d, N_{act}=%d)', N, Nact), ...
        'FontName','Times New Roman','FontSize',11);
    print(fullfile(figDir,'fig5d_sensor_geometry.png'),'-dpng','-r300');
    log_msg('Fig5d saved.');
catch ME; log_msg('ERROR Fig5d: %s', ME.message); end

%% Summary
[~,iSnr35] = max(SNR_dB_vec);
fprintf('\n%-20s  %10s  %10s\n','Layout','LS-DC [dB]','NN [dB]');
fprintf('%s\n',repmat('-',1,44));
for iL = 1:nLayouts
    fprintf('%-20s  %10.2f  %10.2f\n', layoutLabels{iL}, ...
        NMSE_LS_dB(iL,iSnr35), NMSE_NN_dB(iL,iSnr35));
end
log_msg('=== FIG5 COMPLETE ===');

%% -------------------------------------------------------------------
function opts = make_train_opts(maxEpochs, batchSz, lr, dropPer, dropFac, valData, execEnv)
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
        'ExecutionEnvironment',execEnv);
end
