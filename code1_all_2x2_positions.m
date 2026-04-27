%% code1_all_2x2_positions.m 
%
% PURPOSE:
%   Exhaustively tests ALL 9 possible 2x2 active element block positions
%   on a 4x4 RIS panel and finds which gives the best NMSE.

%
% FIGURES PRODUCED:
%   code1_geometry.png        -- 4x4 grid showing each block position
%   code1_lsdc_all.png        -- NMSE vs SNR, all 9 positions, LS-DC
%   code1_nn_all.png          -- NMSE vs SNR, all 9 positions, MLP-NN
%   code1_heatmap.png         -- NMSE heatmap at SNR=20dB (both estimators)
%   code1_best_vs_center.png  -- Best position vs center block comparison
%   code1_phase_shifts.png    -- Optimal phase shift patterns per position
%
% DEPENDENCIES:
%   steering_vector_UPA.m, solve_phase_ls.m, compute_nmse.m,
%   train_mlp_nn.m, log_msg.m, gpu_check.m

clear; clc; close all;

%% ================================================================
%% USER CONTROLS
%% ================================================================
SIM_MODE      = 'fast';   % 'fast' for testing, 'slow' for final results
FORCE_RESTART = false;    % false = resume from checkpoint (do NOT change to true)

if strcmpi(SIM_MODE,'fast')
    numMCS     = 500;
    D_train    = 12000;
    maxEpochs  = 60;
else
    numMCS     = 5000;
    D_train    = 40000;
    maxEpochs  = 150;
end

rng(2022,'twister');

%% ================================================================
%% SYSTEM PARAMETERS  (identical to Table I of paper)
%% ================================================================
fc        = 30e9;
c0        = 3e8;
lambda    = c0/fc;
d_elem    = 0.5*lambda;
k0        = 2*pi/lambda;
K_users   = 4;
s         = sqrt(K_users);
N         = 16;
Nv        = 4;  Nh = 4;
Nact      = 4;                % 2x2 block = 4 active elements
SNR_dB_vec = 0:5:35;
beta0     = 10^(-20/10);
d0        = 1;
d_k_set   = [5 10 15 20 25];
alpha_PL  = 2.2;
numHU     = 128;
numHL     = 4;

useGPU  = gpu_check();
execEnv = 'cpu';
if useGPU; execEnv = 'gpu'; end

%% Anchor ALL file paths to the script's own directory so checkpoints
%% are always saved in the same place regardless of MATLAB's cwd.
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir); scriptDir = pwd; end
figDir  = fullfile(scriptDir, 'figures');
ckptDir = fullfile(scriptDir, 'checkpoints');
if ~exist(figDir,  'dir'); mkdir(figDir);  end
if ~exist(ckptDir, 'dir'); mkdir(ckptDir); end

log_msg('=== CODE 1: ALL 2x2 BLOCK POSITIONS (N=%d) ===', N);
log_msg('Script dir  : %s', scriptDir);
log_msg('Checkpoints : %s', ckptDir);
log_msg('Figures     : %s', figDir);

%% ================================================================
%% ENUMERATE ALL 9 VALID 2x2 BLOCK POSITIONS
%% ================================================================
% On a 4x4 panel, a 2x2 block's top-left corner (r0,c0) ranges:
%   r0 in {1,2,3},  c0 in {1,2,3}  -> 9 positions total

pos_r0    = zeros(1,9);
pos_c0    = zeros(1,9);
pos_idx   = cell(1,9);
pos_label = cell(1,9);

ip = 0;
for r0 = 1:(Nv-1)
    for c0 = 1:(Nh-1)
        ip = ip + 1;
        rows = [r0, r0+1];
        cols = [c0, c0+1];
        [CC, RR] = meshgrid(cols, rows);
        lin = sort(sub2ind([Nv,Nh], RR(:), CC(:)));

        pos_r0(ip)    = r0;
        pos_c0(ip)    = c0;
        pos_idx{ip}   = lin;
        pos_label{ip} = sprintf('r%dc%d', r0, c0);
    end
end
nPos = ip;  % should be 9

% Center block = top-left at (2,2) -> rows 2-3, cols 2-3
iCenter = find(pos_r0==2 & pos_c0==2, 1);

log_msg('Total positions: %d  |  Center index: %d (%s)',...
    nPos, iCenter, pos_label{iCenter});

%% Show which positions already have saved NNs (for resume awareness)
log_msg('--- Checking existing checkpoints ---');
for ip_chk = 1:nPos
    nf = fullfile(ckptDir, sprintf('code1_net_pos%d.mat', ip_chk));
    if exist(nf,'file')
        d = dir(nf);
        log_msg('  pos%d (%s): NET EXISTS  (%.1f KB, saved %s)', ...
            ip_chk, pos_label{ip_chk}, d.bytes/1024, d.date);
    else
        log_msg('  pos%d (%s): not yet trained', ip_chk, pos_label{ip_chk});
    end
end
log_msg('-------------------------------------');

%% ================================================================
%% FIGURE 0: GEOMETRY — FIXED (shows proper 4x4 grid)
%% ================================================================
try
    col_geo = lines(nPos);
    figure('Position',[50 50 1050 280]);
    tl = tiledlayout(1, nPos, 'TileSpacing','compact', 'Padding','compact');

    for ip = 1:nPos
        nexttile;
        hold on;

        % Draw all 16 passive elements as grey circles
        for r = 1:Nv
            for c = 1:Nh
                plot(c, Nv+1-r, 'o', ...
                    'MarkerSize',     14, ...
                    'MarkerFaceColor',[0.82 0.82 0.82], ...
                    'MarkerEdgeColor',[0.55 0.55 0.55], ...
                    'LineWidth',      0.8);
            end
        end

        % Draw 4 active elements as filled coloured squares
        [rr, cc] = ind2sub([Nv, Nh], pos_idx{ip});
        for k = 1:Nact
            plot(cc(k), Nv+1-rr(k), 's', ...
                'MarkerSize',     16, ...
                'MarkerFaceColor', col_geo(ip,:), ...
                'MarkerEdgeColor', 'k', ...
                'LineWidth',       1.2);
        end

        % Draw grid lines
        for r = 0.5:1:Nv+0.5
            plot([0.5 Nh+0.5],[r r],'-','Color',[0.75 0.75 0.75],'LineWidth',0.5);
        end
        for c = 0.5:1:Nh+0.5
            plot([c c],[0.5 Nv+0.5],'-','Color',[0.75 0.75 0.75],'LineWidth',0.5);
        end

        axis equal;
        xlim([0.3 Nh+0.7]);
        ylim([0.3 Nv+0.7]);
        axis off;

        lbl2 = pos_label{ip};
        if ip == iCenter
            lbl2 = [lbl2 ' (ctr)'];
            title(lbl2,'FontName','Times New Roman','FontSize',9,'Color',[0.8 0 0]);
        else
            title(lbl2,'FontName','Times New Roman','FontSize',9);
        end
    end

    title(tl, sprintf('All 2×2 Active Block Positions on %d×%d RIS   (■ active,  ● passive)',...
        Nv, Nh), 'FontName','Times New Roman','FontSize',11);
    print(fullfile(figDir,'code1_geometry.png'),'-dpng','-r300');
    log_msg('Geometry figure saved.');
catch ME
    log_msg('Geometry error: %s', ME.message);
end

%% ================================================================
%% CHECKPOINT
%% ================================================================
ckptFile = fullfile(ckptDir,'code1_progress.mat');
if ~FORCE_RESTART && exist(ckptFile,'file')
    load(ckptFile,'NMSE_LS','NMSE_NN');
    log_msg('Checkpoint loaded.');
else
    NMSE_LS = nan(nPos, numel(SNR_dB_vec));
    NMSE_NN = nan(nPos, numel(SNR_dB_vec));
end

%% ================================================================
%% MAIN LOOP OVER ALL 9 POSITIONS
%% ================================================================
for ip = 1:nPos
    act_idx = pos_idx{ip};
    lbl     = pos_label{ip};
    [row_act, col_act] = ind2sub([Nv, Nh], act_idx);

    log_msg('--- Position %d/%d: %s  (rows=[%d,%d], cols=[%d,%d]) ---',...
        ip, nPos, lbl,...
        pos_r0(ip), pos_r0(ip)+1, pos_c0(ip), pos_c0(ip)+1);

    %% Build LS geometry matrix A (same for all SNR trials at this position)
    [ii_p, jj_p] = find(tril(ones(Nact), -1));
    A_geo = zeros(numel(ii_p), 2);
    for p = 1:numel(ii_p)
        A_geo(p,1) = col_act(ii_p(p)) - col_act(jj_p(p));  % delta_col -> u
        A_geo(p,2) = row_act(ii_p(p)) - row_act(jj_p(p));  % delta_row -> v
    end
    nz    = any(A_geo ~= 0, 2);
    A_geo = A_geo(nz,:);
    ii_nz = ii_p(nz);
    jj_nz = jj_p(nz);

    %% Train MLP-NN for this position
    netFile = fullfile(ckptDir, sprintf('code1_net_pos%d.mat', ip));

    if FORCE_RESTART || ~exist(netFile,'file')
        try
            log_msg('  Training NN for position %s...', lbl);

            [Xtr, Ytr, Xm, Xs, um, us, vm, vs] = ...
                build_dataset(D_train, SNR_dB_vec, k0, d_elem, beta0, d0,...
                d_k_set, alpha_PL, s, act_idx, A_geo, ii_nz, jj_nz, Nv, Nh);

            nTr   = round(0.85 * D_train);
            idx   = randperm(D_train);
            Xtr_t = Xtr(idx(1:nTr),     :);
            Ytr_t = Ytr(idx(1:nTr),     :);
            Xval  = Xtr(idx(nTr+1:end), :);
            Yval  = Ytr(idx(nTr+1:end), :);

            layNN = train_mlp_nn(size(Xtr_t,2), numHU, numHL, 'tanh');
            opts  = make_opts(maxEpochs, 256, 1e-3, 30, 0.5,...
                              {Xval, Yval}, execEnv);
            netNN = trainNetwork(Xtr_t, Ytr_t, layNN, opts);

            save(netFile, 'netNN','Xm','Xs','um','us','vm','vs');
            log_msg('  NN saved for position %s.', lbl);

        catch ME
            log_msg('  ERROR training NN pos %s: %s', lbl, ME.message);
            for kk = 1:numel(ME.stack)
                log_msg('    >> %s (line %d)', ME.stack(kk).name, ME.stack(kk).line);
            end
        end
    else
        load(netFile, 'netNN','Xm','Xs','um','us','vm','vs');
        log_msg('  NN loaded for position %s.', lbl);
    end

    %% Evaluate LS-DC and NN at every SNR point
    for iS = 1:numel(SNR_dB_vec)
        if ~isnan(NMSE_LS(ip,iS)) && ~isnan(NMSE_NN(ip,iS))
            continue;
        end

        snr_dB  = SNR_dB_vec(iS);
        SNR_lin = 10^(snr_dB / 10);

        try
            %% Generate channels
            [H_full, y_act_mc] = gen_chan(numMCS, N, act_idx, k0, d_elem,...
                beta0, d0, d_k_set, alpha_PL, s, SNR_lin, Nv, Nh);

            %% LS-DC
            [u_ls, v_ls] = run_lsdc(y_act_mc, s, A_geo, ii_nz, jj_nz);
            H_ls = reconstruct(u_ls, v_ls, y_act_mc, act_idx, s, Nv, Nh, N);
            NMSE_LS(ip,iS) = 10*log10(compute_nmse(H_full, H_ls));

            %% MLP-NN
            if exist('netNN','var')
                [u_nn, v_nn] = run_nn(netNN, y_act_mc, s, A_geo,...
                    ii_nz, jj_nz, Xm, Xs, um, us, vm, vs);
                H_nn = reconstruct(u_nn, v_nn, y_act_mc, act_idx, s, Nv, Nh, N);
                NMSE_NN(ip,iS) = 10*log10(compute_nmse(H_full, H_nn));
            end

            log_msg('  pos=%s  SNR=%2d dB  LS=%.2f  NN=%.2f dB',...
                lbl, snr_dB, NMSE_LS(ip,iS), NMSE_NN(ip,iS));

            save(ckptFile, 'NMSE_LS','NMSE_NN');

        catch ME
            log_msg('  ERROR pos=%s SNR=%ddB: %s', lbl, snr_dB, ME.message);
            save(ckptFile, 'NMSE_LS','NMSE_NN');
        end
    end

    clear netNN;  % prevent stale network leaking to next position

    %% Save NMSE progress after EVERY completed position
    save(fullfile(ckptDir,'code1_progress.mat'), 'NMSE_LS','NMSE_NN');
    log_msg('  Progress checkpoint saved after position %s.', lbl);
end

%% ================================================================
%% FIND BEST POSITIONS
%% ================================================================
avg_LS = mean(NMSE_LS, 2, 'omitnan');
avg_NN = mean(NMSE_NN, 2, 'omitnan');
[~, iBest_LS] = min(avg_LS);
[~, iBest_NN] = min(avg_NN);

fprintf('\n=== POSITION RANKING (average NMSE across all SNRs) ===\n');
fprintf('%-12s  %12s  %12s\n','Position','LS-DC [dB]','NN [dB]');
fprintf('%s\n', repmat('-',1,40));
[~, sort_order] = sort(avg_LS);
for k = 1:nPos
    ip = sort_order(k);
    tag = '';
    if ip == iBest_LS; tag = [tag '  <- BEST LS']; end
    if ip == iBest_NN; tag = [tag '  <- BEST NN']; end
    if ip == iCenter;  tag = [tag '  (paper center)']; end
    fprintf('  %-10s  %10.2f  %10.2f%s\n',...
        pos_label{ip}, avg_LS(ip), avg_NN(ip), tag);
end

%% ================================================================
%% PLOTTING
%% ================================================================
col_p = lines(nPos);
mk9   = {'o','s','^','d','p','h','v','>','<'};

%% FIGURE 1: LS-DC — NMSE vs SNR, all 9 positions
try
    figure('Position',[50 50 820 530]);
    hold on;
    for ip = 1:nPos
        v  = ~isnan(NMSE_LS(ip,:));
        lw = 1.4;  ms = 6;  ls = '-';
        if ip == iBest_LS;  lw = 2.6;  ms = 10; end
        if ip == iCenter;   lw = 2.0;  ms = 8;  ls = '--'; end
        lbl2 = pos_label{ip};
        if ip == iBest_LS; lbl2 = [lbl2 ' BEST']; end
        if ip == iCenter;  lbl2 = [lbl2 ' (center)']; end
        plot(SNR_dB_vec(v), NMSE_LS(ip,v), [mk9{ip} ls],...
            'Color',col_p(ip,:), 'LineWidth',lw, 'MarkerSize',ms,...
            'DisplayName', lbl2);
    end
    grid on;
    xlabel('SNR [dB]',  'FontName','Times New Roman','FontSize',12);
    ylabel('NMSE [dB]', 'FontName','Times New Roman','FontSize',12);
    title(sprintf('LS-DC Estimator: All 2×2 Block Positions (N=%d, N_{act}=%d)',N,Nact),...
        'FontName','Times New Roman','FontSize',12);
    legend('Location','southwest','FontName','Times New Roman','FontSize',9,...
        'NumColumns',3);
    set(gca,'FontName','Times New Roman','FontSize',11,'FontWeight','bold');
    xlim([SNR_dB_vec(1)-1, SNR_dB_vec(end)+1]);
    print(fullfile(figDir,'code1_lsdc_all.png'),'-dpng','-r300');
    log_msg('LS-DC figure saved.');
catch ME; log_msg('Fig1 error: %s',ME.message); end

%% FIGURE 2: MLP-NN — NMSE vs SNR, all 9 positions
try
    figure('Position',[50 50 820 530]);
    hold on;
    for ip = 1:nPos
        v = ~isnan(NMSE_NN(ip,:));
        if ~any(v); continue; end
        lw = 1.4;  ms = 6;  ls = '--';
        if ip == iBest_NN;  lw = 2.6;  ms = 10; ls = '-'; end
        if ip == iCenter;   lw = 2.0;  ms = 8; end
        lbl2 = pos_label{ip};
        if ip == iBest_NN; lbl2 = [lbl2 '  BEST']; end
        if ip == iCenter;  lbl2 = [lbl2 ' (center)']; end
        plot(SNR_dB_vec(v), NMSE_NN(ip,v), [mk9{ip} ls],...
            'Color',col_p(ip,:), 'LineWidth',lw, 'MarkerSize',ms,...
            'DisplayName', lbl2);
    end
    grid on;
    xlabel('SNR [dB]',  'FontName','Times New Roman','FontSize',12);
    ylabel('NMSE [dB]', 'FontName','Times New Roman','FontSize',12);
    title(sprintf('MLP-NN Estimator: All 2×2 Block Positions (N=%d, N_{act}=%d)',N,Nact),...
        'FontName','Times New Roman','FontSize',12);
    legend('Location','southwest','FontName','Times New Roman','FontSize',9,...
        'NumColumns',3);
    set(gca,'FontName','Times New Roman','FontSize',11,'FontWeight','bold');
    xlim([SNR_dB_vec(1)-1, SNR_dB_vec(end)+1]);
    print(fullfile(figDir,'code1_nn_all.png'),'-dpng','-r300');
    log_msg('MLP-NN figure saved.');
catch ME; log_msg('Fig2 error: %s',ME.message); end

%% FIGURE 3: NMSE HEATMAP at SNR=20dB (both estimators side by side)
try
    [~, iS20] = min(abs(SNR_dB_vec - 20));

    hmap_LS = nan(Nv-1, Nh-1);
    hmap_NN = nan(Nv-1, Nh-1);
    for ip = 1:nPos
        hmap_LS(pos_r0(ip), pos_c0(ip)) = NMSE_LS(ip, iS20);
        hmap_NN(pos_r0(ip), pos_c0(ip)) = NMSE_NN(ip, iS20);
    end

    figure('Position',[50 50 820 370]);

    subplot(1,2,1);
    imagesc(hmap_LS);
    colorbar;
    colormap(gca,'jet');
    % Add NMSE value text on each cell
    for r = 1:Nv-1
        for c = 1:Nh-1
            if ~isnan(hmap_LS(r,c))
                text(c, r, sprintf('%.1f',hmap_LS(r,c)),...
                    'HorizontalAlignment','center',...
                    'VerticalAlignment','middle',...
                    'FontName','Times New Roman','FontSize',10,...
                    'FontWeight','bold','Color','w');
            end
        end
    end
    xlabel('Block col start','FontName','Times New Roman','FontSize',11);
    ylabel('Block row start','FontName','Times New Roman','FontSize',11);
    title(sprintf('LS-DC NMSE [dB] at SNR=%d dB',SNR_dB_vec(iS20)),...
        'FontName','Times New Roman','FontSize',11);
    set(gca,'XTick',1:Nh-1,'YTick',1:Nv-1,...
        'XTickLabel',{'col1','col2','col3'},...
        'YTickLabel',{'row1','row2','row3'},...
        'FontName','Times New Roman','FontSize',10,'FontWeight','bold');

    subplot(1,2,2);
    imagesc(hmap_NN);
    colorbar;
    colormap(gca,'jet');
    for r = 1:Nv-1
        for c = 1:Nh-1
            if ~isnan(hmap_NN(r,c))
                text(c, r, sprintf('%.1f',hmap_NN(r,c)),...
                    'HorizontalAlignment','center',...
                    'VerticalAlignment','middle',...
                    'FontName','Times New Roman','FontSize',10,...
                    'FontWeight','bold','Color','w');
            end
        end
    end
    xlabel('Block col start','FontName','Times New Roman','FontSize',11);
    ylabel('Block row start','FontName','Times New Roman','FontSize',11);
    title(sprintf('MLP-NN NMSE [dB] at SNR=%d dB',SNR_dB_vec(iS20)),...
        'FontName','Times New Roman','FontSize',11);
    set(gca,'XTick',1:Nh-1,'YTick',1:Nv-1,...
        'XTickLabel',{'col1','col2','col3'},...
        'YTickLabel',{'row1','row2','row3'},...
        'FontName','Times New Roman','FontSize',10,'FontWeight','bold');

    sgtitle(sprintf('NMSE Heatmap: 2×2 Block Position (N=%d)  [lower=better]',N),...
        'FontName','Times New Roman','FontSize',12);
    print(fullfile(figDir,'code1_heatmap.png'),'-dpng','-r300');
    log_msg('Heatmap saved.');
catch ME; log_msg('Fig3 error: %s',ME.message); end

%% FIGURE 4: BEST POSITION vs CENTER (both estimators)
try
    figure('Position',[50 50 750 500]);
    hold on;

    % LS-DC curves
    vb = ~isnan(NMSE_LS(iBest_LS,:));
    vc = ~isnan(NMSE_LS(iCenter, :));
    plot(SNR_dB_vec(vb), NMSE_LS(iBest_LS,vb), 'r^-',...
        'LineWidth',2.2,'MarkerSize',9,...
        'DisplayName',sprintf('Best pos (%s) — LS-DC', pos_label{iBest_LS}));
    plot(SNR_dB_vec(vc), NMSE_LS(iCenter, vc), 'ro--',...
        'LineWidth',1.7,'MarkerSize',8,...
        'DisplayName',sprintf('Center (%s) — LS-DC', pos_label{iCenter}));

    % NN curves
    vbn = ~isnan(NMSE_NN(iBest_NN,:));
    vcn = ~isnan(NMSE_NN(iCenter, :));
    if any(vbn)
        plot(SNR_dB_vec(vbn), NMSE_NN(iBest_NN,vbn), 'bs-',...
            'LineWidth',2.2,'MarkerSize',9,...
            'DisplayName',sprintf('Best pos (%s) — MLP-NN', pos_label{iBest_NN}));
    end
    if any(vcn)
        plot(SNR_dB_vec(vcn), NMSE_NN(iCenter, vcn), 'bd--',...
            'LineWidth',1.7,'MarkerSize',8,...
            'DisplayName',sprintf('Center (%s) — MLP-NN', pos_label{iCenter}));
    end

    grid on;
    xlabel('SNR [dB]',  'FontName','Times New Roman','FontSize',12);
    ylabel('NMSE [dB]', 'FontName','Times New Roman','FontSize',12);
    title(sprintf('Best 2×2 Position vs Center Block (N=%d)',N),...
        'FontName','Times New Roman','FontSize',12);
    legend('Location','southwest','FontName','Times New Roman','FontSize',10);
    set(gca,'FontName','Times New Roman','FontSize',11,'FontWeight','bold');
    xlim([SNR_dB_vec(1)-1, SNR_dB_vec(end)+1]);
    print(fullfile(figDir,'code1_best_vs_center.png'),'-dpng','-r300');
    log_msg('Best vs center figure saved.');
catch ME; log_msg('Fig4 error: %s',ME.message); end

%% FIGURE 5: OPTIMAL PHASE SHIFT PATTERNS per position
try
    [~, iS20] = min(abs(SNR_dB_vec - 20));

    figure('Position',[50 50 1050 280]);
    tl2 = tiledlayout(1, nPos, 'TileSpacing','compact', 'Padding','compact');

    for ip = 1:nPos
        nexttile;
        hold on;

        act_idx_ip = pos_idx{ip};
        [row_a, col_a] = ind2sub([Nv,Nh], act_idx_ip);

        % Generate one representative channel realization at SNR=20dB
        SNR_lin_20 = 10^(SNR_dB_vec(iS20)/10);
        rng(ip * 101 + 7);   % fixed seed for reproducibility
        dk  = d_k_set(randi(numel(d_k_set)));
        bk  = beta0*(dk/d0)^(-alpha_PL);
        th  = -pi/2 + pi*rand();
        ph  = -pi/2 + pi*rand();
        u_s = k0*d_elem*cos(th)*cos(ph);
        v_s = k0*d_elem*cos(th)*sin(ph);
        a_h = steering_vector_UPA(Nv, Nh, u_s, v_s);
        h   = sqrt(bk) * a_h;   % UE->RIS channel

        % Independent RIS->AP channel g
        rng(ip * 303 + 13);
        dk2 = d_k_set(randi(numel(d_k_set)));
        bk2 = beta0*(dk2/d0)^(-alpha_PL);
        th2 = -pi/2 + pi*rand();
        ph2 = -pi/2 + pi*rand();
        u2  = k0*d_elem*cos(th2)*cos(ph2);
        v2  = k0*d_elem*cos(th2)*sin(ph2);
        a2  = steering_vector_UPA(Nv, Nh, u2, v2);
        g   = sqrt(bk2) * a2;   % RIS->AP channel

        % Optimal phase shifts: theta_n = -angle(g_n* x h_n)
        theta_opt  = -angle(conj(g) .* h);        % N x 1
        theta_panel = reshape(theta_opt, Nv, Nh) * 180/pi;

        % Plot as heatmap on the 4x4 grid
        imagesc(theta_panel);
        colormap(gca, 'hsv');
        caxis([-180 180]);

        % Overlay all elements as circles
        for r = 1:Nv
            for c = 1:Nh
                plot(c, r, 'o', 'MarkerSize',10,...
                    'MarkerFaceColor','none',...
                    'MarkerEdgeColor',[0.3 0.3 0.3],'LineWidth',0.7);
            end
        end

        % Mark active elements as white squares
        scatter(col_a, row_a, 70, 'w', 's', 'filled',...
            'MarkerEdgeColor','k','LineWidth',1.2);

        axis equal tight;
        set(gca,'XTick',[],'YTick',[],'YDir','normal','FontWeight','bold');
        lbl2 = pos_label{ip};
        if ip == iCenter; lbl2 = [lbl2 '(ctr)']; end
        title(lbl2,'FontName','Times New Roman','FontSize',9);
    end

    % Add shared colorbar
    cb = colorbar('eastoutside');
    cb.Label.String = 'Phase shift [deg]';
    cb.Label.FontName = 'Times New Roman';
    cb.Label.FontSize = 9;

    title(tl2,...
        sprintf('Optimal RIS Phase Shifts [deg] per Position  (SNR=%d dB, N=%d)  ■=active elem',...
        SNR_dB_vec(iS20), N),...
        'FontName','Times New Roman','FontSize',10);
    print(fullfile(figDir,'code1_phase_shifts.png'),'-dpng','-r300');
    log_msg('Phase shift figure saved.');
catch ME
    log_msg('Fig5 error: %s', ME.message);
end

%% FIGURE 6: BAR CHART — gain over center at SNR=20dB
try
    [~, iS20] = min(abs(SNR_dB_vec - 20));
    gain_ls = NMSE_LS(iCenter,iS20) - NMSE_LS(:,iS20);  % positive = better than center
    gain_nn = NMSE_NN(iCenter,iS20) - NMSE_NN(:,iS20);

    figure('Position',[50 50 800 420]);
    x = 1:nPos;
    b1 = bar(x - 0.2, gain_ls, 0.35, 'FaceColor',[0.2 0.5 0.8]);
    hold on;
    b2 = bar(x + 0.2, gain_nn, 0.35, 'FaceColor',[0.9 0.35 0.2]);
    yline(0,'k--','LineWidth',1.2);
    grid on;
    set(gca,'XTick',1:nPos,'XTickLabel',pos_label,...
        'FontName','Times New Roman','FontSize',10,'FontWeight','bold');
    xlabel('Block Position','FontName','Times New Roman','FontSize',12);
    ylabel('NMSE gain over Center [dB]','FontName','Times New Roman','FontSize',12);
    title(sprintf('Gain over Center Position at SNR=%d dB (N=%d)',...
        SNR_dB_vec(iS20),N),...
        'FontName','Times New Roman','FontSize',12);
    legend([b1,b2],{'LS-DC','MLP-NN'},'Location','best',...
        'FontName','Times New Roman','FontSize',10);
    print(fullfile(figDir,'code1_gain_bar.png'),'-dpng','-r300');
    log_msg('Bar chart saved.');
catch ME; log_msg('Fig6 error: %s',ME.message); end

log_msg('=== CODE 1 COMPLETE ===');
fprintf('\nAll figures saved in figures/ folder.\n');

%% ================================================================
%% LOCAL FUNCTIONS  (fully self-contained)
%% ================================================================

function [H_full, y_act] = gen_chan(numMCS, N, act_idx, k0, d_elem,...
        beta0, d0, d_k_set, alpha_PL, s, SNR_lin, Nv, Nh)
%GEN_CHAN Generate channel realizations for a given active element set.
    Nact   = numel(act_idx);
    H_full = zeros(numMCS, N);
    y_act  = zeros(numMCS, Nact);
    for i = 1:numMCS
        dk = d_k_set(randi(numel(d_k_set)));
        bk = beta0 * (dk/d0)^(-alpha_PL);
        th = -pi/2 + pi*rand();
        ph = -pi/2 + pi*rand();
        u  = k0*d_elem*cos(th)*cos(ph);
        v  = k0*d_elem*cos(th)*sin(ph);
        a  = steering_vector_UPA(Nv, Nh, u, v);
        g  = sqrt(bk) * a;
        H_full(i,:) = g.';
        ga   = g(act_idx);
        nvar = (norm(ga)^2 * s^2) / (2*SNR_lin);
        n    = sqrt(nvar) * (randn(Nact,1) + 1j*randn(Nact,1));
        y_act(i,:) = (ga*s + n).';
    end
end

function [X, Y, Xm, Xs, um, us, vm, vs] = build_dataset(D, SNR_dB_vec,...
        k0, d_elem, beta0, d0, d_k_set, alpha_PL, s, act_idx,...
        A_geo, ii_nz, jj_nz, Nv, Nh)
%BUILD_DATASET Build NN training set for a custom active element position.
% Input  features: [u_ls, v_ls, Re(y_norm)', Im(y_norm)', log10(||y||)]
% Output targets:  [delta_u, delta_v]  (residual on top of LS-DC)
    Nact    = numel(act_idx);
    nFeatIn = 2*Nact + 3;
    X_raw   = zeros(D, nFeatIn);
    Y_raw   = zeros(D, 2);
    snr_lev = SNR_dB_vec(:);

    for i = 1:D
        snr_dB  = snr_lev(randi(numel(snr_lev)));
        SNR_lin = 10^(snr_dB / 10);
        dk  = d_k_set(randi(numel(d_k_set)));
        bk  = beta0 * (dk/d0)^(-alpha_PL);
        th  = -pi/2 + pi*rand();
        ph  = -pi/2 + pi*rand();
        u_t = k0*d_elem*cos(th)*cos(ph);
        v_t = k0*d_elem*cos(th)*sin(ph);
        a   = steering_vector_UPA(Nv, Nh, u_t, v_t);
        g   = sqrt(bk)*a;
        ga  = g(act_idx);
        nvar = (norm(ga)^2 * s^2) / (2*SNR_lin);
        n    = sqrt(nvar) * (randn(Nact,1) + 1j*randn(Nact,1));
        y    = ga*s + n;
        gh   = y / s;

        % LS-DC coarse estimate
        phi = zeros(size(A_geo,1), 1);
        for p = 1:size(A_geo,1)
            d     = angle(gh(ii_nz(p))) - angle(gh(jj_nz(p)));
            phi(p) = atan2(sin(d), cos(d));
        end
        mw = sqrt(abs(gh(ii_nz)) .* abs(gh(jj_nz)));
        uv = solve_phase_ls(A_geo, phi, mw);

        % Residual for NN to learn
        du = atan2(sin(u_t - uv(1)), cos(u_t - uv(1)));
        dv = atan2(sin(v_t - uv(2)), cos(v_t - uv(2)));

        % Input features
        py = max(norm(y), 1e-30);
        yn = y / py;
        X_raw(i,:) = [uv(1), uv(2), real(yn).', imag(yn).', log10(py)];
        Y_raw(i,:) = [du, dv];
    end

    % Z-score normalise
    Xm = mean(X_raw,1);
    Xs = std(X_raw, 0, 1);
    Xs(Xs < 1e-10) = 1;
    X  = (X_raw - Xm) ./ Xs;

    um = mean(Y_raw(:,1));  us = std(Y_raw(:,1));  if us < 1e-10; us = 1; end
    vm = mean(Y_raw(:,2));  vs = std(Y_raw(:,2));  if vs < 1e-10; vs = 1; end
    Y  = [(Y_raw(:,1)-um)/us,  (Y_raw(:,2)-vm)/vs];
end

function [u_hat, v_hat] = run_lsdc(y_act, s, A_geo, ii_nz, jj_nz)
%RUN_LSDC  LS-DC angle estimation for all MC trials.
    numMCS = size(y_act, 1);
    u_hat  = zeros(numMCS, 1);
    v_hat  = zeros(numMCS, 1);
    for i = 1:numMCS
        gh  = y_act(i,:).' / s;
        phi = zeros(size(A_geo,1), 1);
        for p = 1:size(A_geo,1)
            d     = angle(gh(ii_nz(p))) - angle(gh(jj_nz(p)));
            phi(p) = atan2(sin(d), cos(d));
        end
        mw  = sqrt(abs(gh(ii_nz)) .* abs(gh(jj_nz)));
        uv  = solve_phase_ls(A_geo, phi, mw);
        u_hat(i) = uv(1);
        v_hat(i) = uv(2);
    end
end

function [u_hat, v_hat] = run_nn(netNN, y_act, s, A_geo,...
        ii_nz, jj_nz, Xm, Xs, um, us, vm, vs)
%RUN_NN  MLP-NN angle estimation: build features then predict residual.
    numMCS = size(y_act, 1);

    % Compute LS-DC base estimate
    [u_ls, v_ls] = run_lsdc(y_act, s, A_geo, ii_nz, jj_nz);

    % Build same features as training
    py = max(sqrt(sum(abs(y_act).^2, 2)), 1e-30);
    yn = y_act ./ py;
    X_raw = [u_ls, v_ls, real(yn), imag(yn), log10(py)];

    % Normalise and predict
    Xn = (X_raw - Xm) ./ Xs;
    Yp = predict(netNN, Xn);

    % De-normalise residual and add to LS-DC
    u_hat = u_ls + Yp(:,1)*us + um;
    v_hat = v_ls + Yp(:,2)*vs + vm;
end

function H_hat = reconstruct(u_hat, v_hat, y_act, act_idx, s, Nv, Nh, N)
%RECONSTRUCT  Full N-element channel from estimated angles.
    numMCS = numel(u_hat);
    H_hat  = zeros(numMCS, N);
    for i = 1:numMCS
        a      = steering_vector_UPA(Nv, Nh, u_hat(i), v_hat(i));
        a_act  = a(act_idx);
        ya     = y_act(i,:).';
        alpha  = (a_act' * ya) / (s * (a_act' * a_act));
        H_hat(i,:) = (alpha * a).';
    end
end

function opts = make_opts(maxEp, bSz, lr, dPer, dFac, val, env)
%MAKE_OPTS  Build MATLAB trainingOptions struct.
    opts = trainingOptions('adam', ...
        'MaxEpochs',            maxEp, ...
        'MiniBatchSize',        bSz,   ...
        'InitialLearnRate',     lr,    ...
        'LearnRateSchedule',    'piecewise', ...
        'LearnRateDropPeriod',  dPer,  ...
        'LearnRateDropFactor',  dFac,  ...
        'Shuffle',              'every-epoch', ...
        'ValidationData',       val,   ...
        'ValidationFrequency',  50,    ...
        'Verbose',              false, ...
        'Plots',                'none',...
        'ExecutionEnvironment', env);
end
