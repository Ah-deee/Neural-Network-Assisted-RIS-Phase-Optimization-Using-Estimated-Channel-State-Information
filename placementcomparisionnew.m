%% fig5_placement_comparison.m  (rebuilt from scratch)
% FIGURE 5: Active Element Placement Optimization
%
% Compares six active-element layouts on a 4x4 (N=16) RIS:
%   center | corners | cross | edges | diagonal | random (averaged)
%
% TWO ESTIMATORS evaluated vs SNR:
%   1. LS-DC  : two-stage weighted phase-unwrapping LS  (solve_phase_ls)
%   2. MLP-NN : per-layout MLP that predicts residual correction on LS-DC
%
% KEY FIX vs original:
%   get_active_indices('random',...) always uses rng(42) internally, so
%   changing the caller's seed had no effect.  We now bypass it for the
%   random layout and generate Nact indices directly here, then pass them
%   to wrapper functions (channel_for_indices / lsdc_for_indices / etc.)
%   that accept an explicit act_lin_idx argument instead of a layout string.

clear; clc; close all;

%% ===================================================================
%% USER CONTROLS
%% ===================================================================
SIM_MODE        = 'slow';   % 'fast' | 'slow'
FORCE_RESTART   = true;
NUM_RAND_TRIALS = 1;       % independent random seeds to average for 'random'

if strcmpi(SIM_MODE, 'fast')
    numMCS     = 500;
    D_train_NN = 15000;
    maxEpochs  = 80;
else
    numMCS     = 5000;
    D_train_NN = 40000;
    maxEpochs  = 120;
end

addpath(genpath('functions'));

useGPU  = gpu_check();
execEnv = 'cpu';
if useGPU; execEnv = 'gpu'; end

baseDir = fileparts(mfilename('fullpath'));
figDir  = fullfile(baseDir, 'figures');
ckptDir = fullfile(baseDir, 'checkpoints');
if ~exist(figDir,  'dir'), mkdir(figDir);  end
if ~exist(ckptDir, 'dir'), mkdir(ckptDir); end

log_msg('=== FIG5: PLACEMENT COMPARISON (%s mode) ===', upper(SIM_MODE));

%% ===================================================================
%% SYSTEM PARAMETERS
%% ===================================================================
fc         = 30e9;
c0         = 3e8;
lambda     = c0 / fc;
d_elem     = 0.5 * lambda;
K_users    = 4;
Nact       = 4;
N          = 16;
Nv         = round(sqrt(N));
SNR_dB_vec = 0:5:35;
nSNR       = numel(SNR_dB_vec);

beta0_dB   = -20;
beta0      = 10^(beta0_dB / 10);
d0         = 1;
d_k_set    = [5 10 15 20 25];
alpha_PL   = 2.2;

numHiddenUnits  = 128;
numHiddenLayers = 4;

%% ===================================================================
%% LAYOUTS
%% ===================================================================
det_layouts      = {'center','corners','cross','edges','diagonal'};
det_labels       = {'Center (baseline)','Corners','Cross','Edges','Diagonal'};
all_layout_names = [det_layouts, {'random'}];
all_layout_labels= [det_labels,  {'Random (avg)'}];
nLayouts         = numel(all_layout_names);

%% ===================================================================
%% RESULT ARRAYS  +  CHECKPOINT
%% ===================================================================
ckptFile = fullfile(ckptDir, 'fig5_progress.mat');
if ~FORCE_RESTART && exist(ckptFile, 'file')
    load(ckptFile, 'NMSE_LS_dB', 'NMSE_NN_dB');
    log_msg('Loaded Fig5 checkpoint.');
else
    NMSE_LS_dB = nan(nLayouts, nSNR);
    NMSE_NN_dB = nan(nLayouts, nSNR);
end

%% ===================================================================
%% PART 1 : DETERMINISTIC LAYOUTS
%% ===================================================================
for iL = 1:numel(det_layouts)
    layout = det_layouts{iL};
    log_msg('---------- Layout %d/%d : %s ----------', iL, nLayouts, layout);

    %% --- Train / load NN ---
    netFile = fullfile(ckptDir, sprintf('fig5_net_%s_N%d.mat', layout, N));

    if ~exist(netFile, 'file')
        try
            log_msg('[%s] Generating %d training samples...', layout, D_train_NN);
            rng(100 + iL, 'twister');
            [Xnn, Ynn, ~, u_mean, u_std, v_mean, v_std, X_mean, X_std] = ...
                nn_dataset_generator(D_train_NN, SNR_dB_vec, fc, d_elem, ...
                    beta0, d0, d_k_set, alpha_PL, K_users, Nact, N, layout);

            numTrain = round(0.85 * D_train_NN);
            idx_perm = randperm(D_train_NN);
            Xtr = Xnn(idx_perm(1:numTrain), :);
            Ytr = Ynn(idx_perm(1:numTrain), :);
            Xvl = Xnn(idx_perm(numTrain+1:end), :);
            Yvl = Ynn(idx_perm(numTrain+1:end), :);

            log_msg('[%s] Training MLP (%dx%d hidden, %d epochs)...', ...
                layout, numHiddenUnits, numHiddenLayers, maxEpochs);
            layersNN = train_mlp_nn(size(Xtr,2), numHiddenUnits, numHiddenLayers, 'tanh');
            opts     = local_train_opts(maxEpochs, 256, 1e-3, 30, 0.5, {Xvl,Yvl}, execEnv);
            netNN    = trainNetwork(Xtr, Ytr, layersNN, opts);

            save(netFile, 'netNN', 'u_mean','u_std','v_mean','v_std','X_mean','X_std');
            log_msg('[%s] Network saved.', layout);
        catch ME
            log_msg('ERROR training NN [%s]: %s', layout, ME.message);
            for k = 1:numel(ME.stack)
                log_msg('  >> %s (line %d)', ME.stack(k).name, ME.stack(k).line);
            end
        end
    else
        log_msg('[%s] Loading pre-trained network...', layout);
        load(netFile, 'netNN', 'u_mean','u_std','v_mean','v_std','X_mean','X_std');
    end

    %% --- Evaluate ---
    for iS = 1:nSNR
        snr_dB = SNR_dB_vec(iS);
        if ~isnan(NMSE_LS_dB(iL,iS)) && ~isnan(NMSE_NN_dB(iL,iS))
            log_msg('[%s] SNR=%ddB -- SKIPPED', layout, snr_dB); continue;
        end
        log_msg('[%s] SNR=%ddB (%d/%d)...', layout, snr_dB, iS, nSNR);
        try
            rng(200 + iS, 'twister');
            [H_full, y_act_obs, ~, ~] = generate_channel(numMCS, N, Nact, ...
                fc, d_elem, beta0, d0, d_k_set, alpha_PL, K_users, snr_dB, layout);

            if isnan(NMSE_LS_dB(iL,iS))
                [~, ~, H_hat_LS]   = ls_dc_estimator(y_act_obs, N, Nact, fc, d_elem, K_users, layout);
                NMSE_LS_dB(iL,iS) = 10*log10(compute_nmse(H_full, H_hat_LS));
            end

            if isnan(NMSE_NN_dB(iL,iS)) && exist('netNN','var')
                X_test   = [real(y_act_obs), imag(y_act_obs)];
                uv_hat   = nn_predict_angles(netNN, X_test, u_mean, u_std, ...
                    v_mean, v_std, X_mean, X_std, N, Nact, layout);
                H_hat_NN = interpolate_passive_elements(uv_hat(:,1), uv_hat(:,2), ...
                    N, Nact, fc, d_elem, y_act_obs, K_users, layout);
                NMSE_NN_dB(iL,iS) = 10*log10(compute_nmse(H_full, H_hat_NN));
            end

            log_msg('  LS-DC=%.2f dB   NN=%.2f dB', NMSE_LS_dB(iL,iS), NMSE_NN_dB(iL,iS));
            save(ckptFile, 'NMSE_LS_dB', 'NMSE_NN_dB');
        catch ME
            log_msg('ERROR [%s] SNR=%ddB: %s', layout, snr_dB, ME.message);
            save(ckptFile, 'NMSE_LS_dB', 'NMSE_NN_dB');
        end
    end

    clear netNN;
end

%% ===================================================================
%% PART 2 : RANDOM LAYOUT  (averaged over NUM_RAND_TRIALS seeds)
%% ===================================================================
% CRITICAL: We do NOT call get_active_indices('random',...) here because
% that function hardcodes rng(42) internally and always returns the same
% layout regardless of the caller's seed.  Instead we generate indices
% directly and pass them to index-aware local wrappers.
% ===================================================================
iL_rand = nLayouts;
log_msg('---------- Layout %d/%d : random (avg over %d trials) ----------', ...
    iL_rand, nLayouts, NUM_RAND_TRIALS);

NMSE_rand_LS = nan(NUM_RAND_TRIALS, nSNR);
NMSE_rand_NN = nan(NUM_RAND_TRIALS, nSNR);

for trial = 1:NUM_RAND_TRIALS
    log_msg('[random] Trial %d / %d', trial, NUM_RAND_TRIALS);

    % Unique reproducible layout for this trial
    rng(trial * 13 + 7, 'twister');
    perm_trial   = randperm(Nv * Nv, Nact);
    act_idx_rand = sort(perm_trial(:));   % Nact x 1 linear indices

    %% --- Train / load NN for this trial ---
    netFile_r = fullfile(ckptDir, ...
        sprintf('fig5_net_random_trial%02d_N%d.mat', trial, N));

    if ~exist(netFile_r, 'file')
        try
            log_msg('[random trial %d] Generating training data...', trial);
            rng(trial * 13 + 8, 'twister');
            [Xnn, Ynn, u_mean, u_std, v_mean, v_std, X_mean, X_std] = ...
                dataset_for_indices(D_train_NN, SNR_dB_vec, fc, d_elem, ...
                    beta0, d0, d_k_set, alpha_PL, K_users, Nact, N, act_idx_rand);

            numTrain = round(0.85 * D_train_NN);
            idx_perm = randperm(D_train_NN);
            Xtr = Xnn(idx_perm(1:numTrain), :);
            Ytr = Ynn(idx_perm(1:numTrain), :);
            Xvl = Xnn(idx_perm(numTrain+1:end), :);
            Yvl = Ynn(idx_perm(numTrain+1:end), :);

            log_msg('[random trial %d] Training MLP...', trial);
            layersNN = train_mlp_nn(size(Xtr,2), numHiddenUnits, numHiddenLayers, 'tanh');
            opts     = local_train_opts(maxEpochs, 256, 1e-3, 30, 0.5, {Xvl,Yvl}, execEnv);
            netNN_r  = trainNetwork(Xtr, Ytr, layersNN, opts);

            save(netFile_r, 'netNN_r', 'act_idx_rand', ...
                'u_mean','u_std','v_mean','v_std','X_mean','X_std');
            log_msg('[random trial %d] Network saved.', trial);
        catch ME
            log_msg('ERROR training NN [random trial %d]: %s', trial, ME.message);
            continue;
        end
    else
        load(netFile_r, 'netNN_r', 'act_idx_rand', ...
            'u_mean','u_std','v_mean','v_std','X_mean','X_std');
    end

    %% --- Evaluate each SNR for this trial ---
    for iS = 1:nSNR
        snr_dB = SNR_dB_vec(iS);
        try
            rng(trial * 1000 + iS, 'twister');
            [H_full, y_act_obs] = channel_for_indices(numMCS, N, act_idx_rand, ...
                fc, d_elem, beta0, d0, d_k_set, alpha_PL, K_users, snr_dB);

            [u_ls, v_ls]     = lsdc_for_indices(y_act_obs, N, act_idx_rand, K_users);
            H_hat_LS         = interp_for_indices(u_ls, v_ls, N, act_idx_rand, y_act_obs, K_users);
            NMSE_rand_LS(trial,iS) = 10*log10(compute_nmse(H_full, H_hat_LS));

            X_test   = [real(y_act_obs), imag(y_act_obs)];
            uv_hat   = nn_pred_for_indices(netNN_r, X_test, u_mean, u_std, ...
                v_mean, v_std, X_mean, X_std, N, Nact, act_idx_rand);
            H_hat_NN = interp_for_indices(uv_hat(:,1), uv_hat(:,2), ...
                N, act_idx_rand, y_act_obs, K_users);
            NMSE_rand_NN(trial,iS) = 10*log10(compute_nmse(H_full, H_hat_NN));

            log_msg('  trial%02d SNR=%ddB  LS=%.2f  NN=%.2f', ...
                trial, snr_dB, NMSE_rand_LS(trial,iS), NMSE_rand_NN(trial,iS));
        catch ME
            log_msg('ERROR [random trial %d] SNR=%ddB: %s', trial, snr_dB, ME.message);
        end
    end

    clear netNN_r;

    % Update running average after each completed trial
    NMSE_LS_dB(iL_rand,:) = mean(NMSE_rand_LS, 1, 'omitnan');
    NMSE_NN_dB(iL_rand,:) = mean(NMSE_rand_NN, 1, 'omitnan');
    save(ckptFile, 'NMSE_LS_dB', 'NMSE_NN_dB', 'NMSE_rand_LS', 'NMSE_rand_NN');
end

NMSE_LS_dB(iL_rand,:) = mean(NMSE_rand_LS, 1, 'omitnan');
NMSE_NN_dB(iL_rand,:) = mean(NMSE_rand_NN, 1, 'omitnan');
save(ckptFile, 'NMSE_LS_dB', 'NMSE_NN_dB', 'NMSE_rand_LS', 'NMSE_rand_NN');
log_msg('[random] Final avg  LS=%.2f dB  NN=%.2f dB  (SNR=%ddB)', ...
    NMSE_LS_dB(iL_rand,end), NMSE_NN_dB(iL_rand,end), SNR_dB_vec(end));

%% ===================================================================
%% PLOTTING
%% ===================================================================
colors = lines(nLayouts);
mks    = {'o','s','^','d','p','h'};

%% Fig5a : LS-DC
try
    figure('Position',[50 100 700 500]); hold on;
    hL = gobjects(nLayouts,1);
    for iL = 1:nLayouts
        v = ~isnan(NMSE_LS_dB(iL,:));
        if any(v)
            hL(iL) = plot(SNR_dB_vec(v), NMSE_LS_dB(iL,v), [mks{iL} '-'], ...
                'Color',colors(iL,:),'LineWidth',1.6,'MarkerSize',7, ...
                'DisplayName',all_layout_labels{iL});
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

%% Fig5b : MLP-NN
try
    figure('Position',[50 100 700 500]); hold on;
    hL = gobjects(nLayouts,1);
    for iL = 1:nLayouts
        v = ~isnan(NMSE_NN_dB(iL,:));
        if any(v)
            hL(iL) = plot(SNR_dB_vec(v), NMSE_NN_dB(iL,v), [mks{iL} '--'], ...
                'Color',colors(iL,:),'LineWidth',1.6,'MarkerSize',7, ...
                'DisplayName',all_layout_labels{iL});
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

%% Fig5c : Gain bar chart
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
    set(gca,'XTick',1:nLayouts,'XTickLabel',all_layout_labels, ...
        'FontName','Times New Roman','FontSize',10,'fontweight','bold');
    xtickangle(15);
    xlabel('Layout','FontName','Times New Roman','FontSize',12);
    ylabel('NMSE gain over Center [dB]','FontName','Times New Roman','FontSize',12);
    title(sprintf('Placement gain at SNR=%d dB, N=%d', SNR_dB_vec(iSnrTop),N), ...
        'FontName','Times New Roman','FontSize',11);
    legend([b1,b2],{'LS-DC','MLP-NN'},'Location','best', ...
        'FontName','Times New Roman','FontSize',10);
    print(fullfile(figDir,'fig5c_placement_gain_bar.png'),'-dpng','-r300');
    log_msg('Fig5c saved.');
catch ME; log_msg('ERROR Fig5c: %s', ME.message); end

%% Fig5d : Geometry panel
try
    figure('Position',[50 100 1100 340]);
    tl = tiledlayout(1,nLayouts,'TileSpacing','compact','Padding','compact');
    for iL = 1:nLayouts
        nexttile;
        if iL < nLayouts
            idx = get_active_indices(Nv, Nv, Nact, all_layout_names{iL});
        else
            rng(1*13+7,'twister');
            perm_show = randperm(Nv*Nv, Nact);
            idx = sort(perm_show(:));
        end
        [rv,cv] = ind2sub([Nv,Nv], idx);
        [ac,ar] = meshgrid(1:Nv, 1:Nv);
        scatter(ac(:), Nv+1-ar(:), 160, [0.85 0.85 0.85],'o','filled');
        hold on;
        scatter(cv, Nv+1-rv, 200, colors(iL,:),'s','filled', ...
            'MarkerEdgeColor','k','LineWidth',1.1);
        axis equal tight;
        set(gca,'XTick',1:Nv,'YTick',1:Nv,'XTickLabel',{},'YTickLabel',{}, ...
            'XGrid','on','YGrid','on','GridColor',[0.7 0.7 0.7],'fontweight','bold');
        xlim([0.4 Nv+0.6]); ylim([0.4 Nv+0.6]);
        title(all_layout_labels{iL},'FontName','Times New Roman','FontSize',10);
        box on;
    end
    title(tl, sprintf('Active element layouts (N=%d, N_{act}=%d)',N,Nact), ...
        'FontName','Times New Roman','FontSize',11);
    print(fullfile(figDir,'fig5d_sensor_geometry.png'),'-dpng','-r300');
    log_msg('Fig5d saved.');
catch ME; log_msg('ERROR Fig5d: %s', ME.message); end

%% Summary table
[~,iSnr35] = max(SNR_dB_vec);
fprintf('\n%-22s  %10s  %10s\n','Layout','LS-DC [dB]','NN [dB]');
fprintf('%s\n',repmat('-',1,46));
for iL = 1:nLayouts
    fprintf('%-22s  %10.2f  %10.2f\n', all_layout_labels{iL}, ...
        NMSE_LS_dB(iL,iSnr35), NMSE_NN_dB(iL,iSnr35));
end
log_msg('=== FIG5 COMPLETE ===');


%% ===================================================================
%% LOCAL HELPER FUNCTIONS
%% ===================================================================

% -----------------------------------------------------------------------
function opts = local_train_opts(maxEpochs, batchSz, lr, dropPer, dropFac, valData, execEnv)
    opts = trainingOptions('adam', ...
        'MaxEpochs',            maxEpochs, ...
        'MiniBatchSize',        batchSz, ...
        'InitialLearnRate',     lr, ...
        'LearnRateSchedule',    'piecewise', ...
        'LearnRateDropPeriod',  dropPer, ...
        'LearnRateDropFactor',  dropFac, ...
        'Shuffle',              'every-epoch', ...
        'ValidationData',       valData, ...
        'ValidationFrequency',  50, ...
        'Verbose',              false, ...
        'Plots',                'none', ...
        'ExecutionEnvironment', execEnv);
end

% -----------------------------------------------------------------------
function [H_full, y_act] = channel_for_indices(numMCS, N, act_idx, ...
    fc, d_elem, beta0, d0, d_k_set, alpha_PL, K_users, SNR_dB)
% Like generate_channel but accepts an explicit act_idx vector, bypassing
% get_active_indices entirely so the caller controls the random layout.
    c0      = 3e8;
    lambda  = c0/fc;
    k0      = 2*pi/lambda;
    Nv      = round(sqrt(N));
    s       = sqrt(K_users);
    SNR_lin = 10^(SNR_dB/10);
    Nact_l  = numel(act_idx);
    H_full  = zeros(numMCS, N);
    y_act   = zeros(numMCS, Nact_l);
    for i = 1:numMCS
        dk     = d_k_set(randi(numel(d_k_set)));
        beta_k = beta0*(dk/d0)^(-alpha_PL);
        theta  = -pi/2 + pi*rand();
        phi_a  = -pi/2 + pi*rand();
        u = k0*d_elem*cos(theta)*cos(phi_a);
        v = k0*d_elem*cos(theta)*sin(phi_a);
        a = steering_vector_UPA(Nv, Nv, u, v);
        g = sqrt(beta_k)*a;
        H_full(i,:) = g.';
        g_act = g(act_idx);
        noise_var = (norm(g_act)^2*s^2)/SNR_lin;
        n = sqrt(noise_var/2)*(randn(size(g_act))+1j*randn(size(g_act)));
        y_act(i,:) = (g_act*s + n).';
    end
end

% -----------------------------------------------------------------------
function [u_hat, v_hat] = lsdc_for_indices(y_act, N, act_idx, K_users)
% Generalised phase-difference LS-DC with explicit act_idx.
    Nv     = round(sqrt(N));
    s      = sqrt(K_users);
    numMCS = size(y_act,1);
    Nact_l = numel(act_idx);
    [row_act, col_act] = ind2sub([Nv,Nv], act_idx);
    [ii,jj] = find(tril(ones(Nact_l),-1));
    A = zeros(numel(ii),2);
    for p = 1:numel(ii)
        A(p,1) = col_act(ii(p)) - col_act(jj(p));
        A(p,2) = row_act(ii(p)) - row_act(jj(p));
    end
    nz = any(A~=0,2);  A = A(nz,:);  ii = ii(nz);  jj = jj(nz);
    u_hat = zeros(numMCS,1);
    v_hat = zeros(numMCS,1);
    for i = 1:numMCS
        g_hat = y_act(i,:).'/s;
        phi_vec = zeros(size(A,1),1);
        for p = 1:size(A,1)
            d = angle(g_hat(ii(p))) - angle(g_hat(jj(p)));
            phi_vec(p) = atan2(sin(d),cos(d));
        end
        mw = sqrt(abs(g_hat(ii)).*abs(g_hat(jj)));
        uv = solve_phase_ls(A, phi_vec, mw);
        u_hat(i) = uv(1);
        v_hat(i) = uv(2);
    end
end

% -----------------------------------------------------------------------
function H_hat = interp_for_indices(u_hat, v_hat, N, act_idx, y_act, K_users)
% Full channel reconstruction with explicit act_idx.
    Nv     = round(sqrt(N));
    s      = sqrt(K_users);
    numMCS = numel(u_hat);
    H_hat  = zeros(numMCS,N);
    for i = 1:numMCS
        a     = steering_vector_UPA(Nv, Nv, u_hat(i), v_hat(i));
        a_act = a(act_idx);
        y_i   = y_act(i,:).';
        alpha  = (a_act'*y_i)/(s*(a_act'*a_act));
        H_hat(i,:) = (alpha*a).';
    end
end

% -----------------------------------------------------------------------
function uv_hat = nn_pred_for_indices(net, X_raw, u_mean, u_std, ...
    v_mean, v_std, X_mean, X_std, N, Nact_l, act_idx)
% Mirrors nn_predict_angles feature pipeline but uses explicit act_idx.
    D      = size(X_raw,1);
    Nv     = round(sqrt(N));
    Re_y   = X_raw(:,1:Nact_l);
    Im_y   = X_raw(:,Nact_l+1:end);
    [row_act,col_act] = ind2sub([Nv,Nv], act_idx);
    [ii,jj] = find(tril(ones(Nact_l),-1));
    A = zeros(numel(ii),2);
    for p = 1:numel(ii)
        A(p,1) = col_act(ii(p)) - col_act(jj(p));
        A(p,2) = row_act(ii(p)) - row_act(jj(p));
    end
    nz = any(A~=0,2);  A = A(nz,:);  ii = ii(nz);  jj = jj(nz);
    u_LS = zeros(D,1);  v_LS = zeros(D,1);  SNR_est = zeros(D,1);
    for i = 1:D
        g_hat = Re_y(i,:).'+1j*Im_y(i,:).';
        phi_vec = zeros(size(A,1),1);
        for p = 1:size(A,1)
            d = angle(g_hat(ii(p)))-angle(g_hat(jj(p)));
            phi_vec(p) = atan2(sin(d),cos(d));
        end
        mw = sqrt(abs(g_hat(ii)).*abs(g_hat(jj)));
        uv = solve_phase_ls(A, phi_vec, mw);
        u_LS(i) = uv(1);  v_LS(i) = uv(2);
        a_LS  = steering_vector_UPA(Nv,Nv,u_LS(i),v_LS(i));
        a_act = a_LS(act_idx);
        alpha_LS = (a_act'*g_hat)/(a_act'*a_act);
        y_fit = alpha_LS*a_act;
        res_pow = max(norm(g_hat-y_fit)^2,1e-30);
        SNR_est(i) = max(-10,min(50,10*log10(norm(y_fit)^2/res_pow)));
    end
    py = sqrt(sum(Re_y.^2+Im_y.^2,2));  py(py<1e-30) = 1e-30;
    n_feat = log2(N/16)*ones(D,1);
    X_feat = [u_LS, v_LS, Re_y./py, Im_y./py, log10(py), n_feat, SNR_est];
    X_norm = (X_feat-X_mean)./X_std;
    delta_norm = predict(net, X_norm);
    uv_hat = zeros(D,2);
    uv_hat(:,1) = u_LS + delta_norm(:,1)*u_std + u_mean;
    uv_hat(:,2) = v_LS + delta_norm(:,2)*v_std + v_mean;
end

% -----------------------------------------------------------------------
function [X, Y, u_mean, u_std, v_mean, v_std, X_mean, X_std] = ...
    dataset_for_indices(D, SNR_dB_vec, fc, d_elem, beta0, d0, ...
    d_k_set, alpha_PL, K_users, Nact_l, N, act_idx)
% Mirrors nn_dataset_generator feature pipeline but uses explicit act_idx.
    c0     = 3e8;  lambda = c0/fc;  k0 = 2*pi/lambda;
    Nv     = round(sqrt(N));
    s      = sqrt(K_users);
    n_feat_val = log2(N/16);
    snr_levels = SNR_dB_vec(:);
    weights = exp(snr_levels/20);  weights = weights/sum(weights);
    cum_w   = cumsum(weights);
    r       = rand(D,1);
    idx_snr = sum(r>cum_w',2)+1;
    idx_snr = min(idx_snr,numel(snr_levels));
    SNR_vec = snr_levels(idx_snr);
    [row_act,col_act] = ind2sub([Nv,Nv], act_idx);
    [ii,jj] = find(tril(ones(Nact_l),-1));
    A = zeros(numel(ii),2);
    for p = 1:numel(ii)
        A(p,1) = col_act(ii(p))-col_act(jj(p));
        A(p,2) = row_act(ii(p))-row_act(jj(p));
    end
    nz = any(A~=0,2);  A = A(nz,:);  ii = ii(nz);  jj = jj(nz);
    nFeat = 2*Nact_l+5;
    X_raw = zeros(D,nFeat);  Y_raw = zeros(D,2);
    for i = 1:D
        SNR_lin = 10^(SNR_vec(i)/10);
        dk     = d_k_set(randi(numel(d_k_set)));
        beta_k = beta0*(dk/d0)^(-alpha_PL);
        theta  = -pi/2+pi*rand();  phi_a = -pi/2+pi*rand();
        u_true = k0*d_elem*cos(theta)*cos(phi_a);
        v_true = k0*d_elem*cos(theta)*sin(phi_a);
        a      = steering_vector_UPA(Nv,Nv,u_true,v_true);
        g      = sqrt(beta_k)*a;
        g_act  = g(act_idx);
        noise_var = (norm(g_act)^2*s^2)/SNR_lin;
        n_n   = sqrt(noise_var/2)*(randn(size(g_act))+1j*randn(size(g_act)));
        y = g_act*s+n_n;
        g_hat = y/s;
        phi_vec = zeros(size(A,1),1);
        for p = 1:size(A,1)
            d = angle(g_hat(ii(p)))-angle(g_hat(jj(p)));
            phi_vec(p) = atan2(sin(d),cos(d));
        end
        mw = sqrt(abs(g_hat(ii)).*abs(g_hat(jj)));
        uv = solve_phase_ls(A, phi_vec, mw);
        u_LS = uv(1);  v_LS = uv(2);
        delta_u = atan2(sin(u_true-u_LS),cos(u_true-u_LS));
        delta_v = atan2(sin(v_true-v_LS),cos(v_true-v_LS));
        py = norm(y);  if py<1e-30; py=1e-30; end
        y_phase = y/py;
        a_LS   = steering_vector_UPA(Nv,Nv,u_LS,v_LS);
        a_act  = a_LS(act_idx);
        alpha_LS = (a_act'*g_hat)/(a_act'*a_act);
        y_fit  = alpha_LS*a_act*s;
        res_pow = max(norm(y-y_fit)^2,1e-30);
        SNR_est = max(-10,min(50,10*log10(norm(y_fit)^2/res_pow)));
        X_raw(i,:) = [u_LS,v_LS,real(y_phase).',imag(y_phase).',log10(py),n_feat_val,SNR_est];
        Y_raw(i,:) = [delta_u,delta_v];
    end
    X_mean = mean(X_raw,1);  X_std = std(X_raw,0,1);
    X_std(X_std<1e-10) = 1;
    X = (X_raw-X_mean)./X_std;
    u_mean = mean(Y_raw(:,1));  u_std = std(Y_raw(:,1));
    v_mean = mean(Y_raw(:,2));  v_std = std(Y_raw(:,2));
    if u_std<1e-10; u_std=1; end
    if v_std<1e-10; v_std=1; end
    Y = zeros(size(Y_raw));
    Y(:,1) = (Y_raw(:,1)-u_mean)/u_std;
    Y(:,2) = (Y_raw(:,2)-v_mean)/v_std;
    fprintf('Random-trial dataset: %d samples, N=%d.\n', D, N);
end
