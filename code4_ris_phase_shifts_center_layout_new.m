%% code4_ris_phase_shifts_center_layout.m
%
% =========================================================================
% USES YOUR ALREADY-TRAINED NN — NO RETRAINING AT ALL
%   Loads:  checkpoints/fig5_net_center_N16.mat
%   Variables: netNN, X_mean, X_std, u_mean, u_std, v_mean, v_std
%
% SIGNAL FLOW (answers your question):
%
%   STEP 1 — USER TRANSMITS PILOT:
%     User transmits pilot s. The 4 CENTER active elements receive:
%       y_act(n) = h_n * s + noise    (n in rows 2-3, cols 2-3)
%
%   STEP 2 — CHANNEL ESTIMATION FROM PILOT:
%     Your trained NN estimates the full UE->RIS channel h (all 16
%     elements) from y_act. LS-DC does the same without NN correction.
%
%   STEP 3 — OPTIMAL PHASE SHIFT COMPUTATION:
%     For each RIS element n:
%       theta_n = -angle( conj(g_n) * h_n_estimated )
%     This CANCELS the cascade phase at every element so all 16 reflected
%     paths arrive at the AP in phase (constructive interference).
%
%   STEP 4 — DATA TRANSMISSION:
%     RIS applies Phi = diag(exp(j*theta)).
%     AP received power = |sum_n g_n * exp(j*theta_n) * h_n|^2 * |s|^2
%     All 16 contributions add coherently -> maximum SNR.
%
% YES — the pilot tells you the user direction (via u,v), your NN
%   reconstructs h from that, then theta_n steers each element's
%   reflection to arrive at the AP in phase. This is RIS beamforming.
%
% FIGURES PRODUCED (saved to figures/):
%   code4_phase_shifts_SNR20.png  -- theta_n on every element (4x4 grid)
%   code4_phase_error_map.png     -- RMS phase error per element
%   code4_nmse_vs_snr.png         -- channel estimation NMSE
%   code4_rate_vs_snr.png         -- achievable rate comparison
%   code4_bf_gain_vs_snr.png      -- rate gain of RIS over no-RIS
%   code4_coherent_combining.png  -- phasor diagram of coherent combining
% =========================================================================

clear; clc; close all;

%% ================================================================
%% USER SETTINGS
%% ================================================================
SIM_MODE = 'slow';   % 'fast' = quick test | 'slow' = publication quality

if strcmpi(SIM_MODE,'fast')
    numMCS = 800;
else
    numMCS = 5000;
end

rng(2022,'twister');

%% ================================================================
%% SYSTEM PARAMETERS  (Table I of paper)
%% ================================================================
fc        = 30e9;
c0        = 3e8;
lambda    = c0/fc;
d_elem    = 0.5*lambda;
K_users   = 4;
s_pilot   = sqrt(K_users);
N         = 16;
Nv        = 4;  Nh = 4;
Nact      = 4;
layout    = 'center';
SNR_dB_vec = 0:5:35;
beta0     = 10^(-20/10);
d0        = 1;
d_k_set   = [5 10 15 20 25];
alpha_PL  = 2.2;
STORE_SNR_dB = 20;

% --- Resolve all paths relative to THIS script file, not MATLAB's cwd ---
% This ensures checkpoints/ and figures/ are found regardless of where
% you launched MATLAB from.
scriptDir   = fileparts(mfilename('fullpath'));
if isempty(scriptDir)          % running from Editor "Run" button
    scriptDir = pwd;
end
ckptDir    = fullfile(scriptDir, 'checkpoints');
figDir     = fullfile(scriptDir, 'figures');
if ~exist(figDir,'dir');  mkdir(figDir);  end
if ~exist(ckptDir,'dir'); mkdir(ckptDir); end

log_msg('=== CODE 4: RIS PHASE SHIFTS — CENTER LAYOUT (N=%d) ===', N);
log_msg('Script folder : %s', scriptDir);
log_msg('Checkpoints   : %s', ckptDir);

%% ================================================================
%% SHOW ACTIVE ELEMENT POSITIONS
%% ================================================================
act_idx = get_active_indices(Nv, Nh, Nact, layout);
[act_rows, act_cols] = ind2sub([Nv, Nh], act_idx);

fprintf('\n--- Active elements (center 2x2 block: rows 2-3, cols 2-3) ---\n');
for k = 1:Nact
    fprintf('  Linear index %2d  ->  row=%d, col=%d\n',...
        act_idx(k), act_rows(k), act_cols(k));
end
fprintf('\n');

%% ================================================================
%% LOAD YOUR ALREADY-TRAINED NN  (zero retraining)
%% ================================================================
netFile = fullfile(ckptDir,'fig5_net_center_N16.mat');

if ~exist(netFile,'file')
    error(['Trained NN not found at: %s\n'...
        'Make sure fig5_net_center_N16.mat is in your checkpoints/ folder.\n'...
        'This file is created when you run fig5_placement_comparison.m.'],...
        netFile);
end

log_msg('Loading trained NN from: %s', netFile);
ld = load(netFile, 'netNN','X_mean','X_std','u_mean','u_std','v_mean','v_std');

netNN  = ld.netNN;    % your fully-trained SeriesNetwork / DAGNetwork
X_mean = ld.X_mean;  % [1 x 13]  input feature normalisation mean
X_std  = ld.X_std;   % [1 x 13]  input feature normalisation std
u_mean = ld.u_mean;  % scalar    output residual de-normalisation
u_std  = ld.u_std;
v_mean = ld.v_mean;
v_std  = ld.v_std;

log_msg('NN loaded. Feature vector length = %d (2*Nact+5 = %d). OK = %d',...
    numel(X_mean), 2*Nact+5, numel(X_mean)==2*Nact+5);

%% ================================================================
%% RESULT ARRAYS  (always fresh)
%% ================================================================
nSNR       = numel(SNR_dB_vec);
NMSE_LS    = nan(1, nSNR);
NMSE_NN    = nan(1, nSNR);
RATE_perf  = nan(1, nSNR);
RATE_nn    = nan(1, nSNR);
RATE_ls    = nan(1, nSNR);
RATE_none  = nan(1, nSNR);
BF_GAIN_NN = nan(1, nSNR);
BF_GAIN_LS = nan(1, nSNR);

H_h_store    = [];
H_ls_store   = [];
H_nn_store   = [];

%% ================================================================
%% MAIN EVALUATION LOOP
%% ================================================================
log_msg('--- Evaluating %d SNR points ---', nSNR);

for iS = 1:nSNR
    snr_dB  = SNR_dB_vec(iS);
    SNR_lin = 10^(snr_dB / 10);

    try
        %% STEP 1 — USER TRANSMITS PILOT
        % Returns true channel h and noisy pilot y at 4 active elements
        [H_h, y_act, ~, ~] = generate_channel(numMCS, N, Nact, fc, d_elem,...
            beta0, d0, d_k_set, alpha_PL, K_users, snr_dB, layout);
        % H_h   : [numMCS x 16]  true UE->RIS channel (all elements)
        % y_act : [numMCS x  4]  received pilot at active elements only

        %% STEP 2b — LS-DC ESTIMATION
        % Estimates full channel vector from pilot at 4 active elements
        [~, ~, H_ls] = ls_dc_estimator(y_act, N, Nact, fc, d_elem,...
            K_users, layout);

        %% STEP 2c — YOUR TRAINED NN ESTIMATION
        % Passes [Re(y_act), Im(y_act)] to nn_predict_angles which builds
        % the 13-feature vector internally, runs netNN, returns [u_hat, v_hat]
        X_test = [real(y_act), imag(y_act)];      % [numMCS x 8]
        uv_hat = nn_predict_angles(netNN, X_test,...
            u_mean, u_std, v_mean, v_std,...
            X_mean, X_std, N, Nact, layout);       % [numMCS x 2]
        H_nn = interpolate_passive_elements(...
            uv_hat(:,1), uv_hat(:,2),...
            N, Nact, fc, d_elem, y_act, K_users, layout);  % [numMCS x 16]

        %% NMSE
        NMSE_LS(iS) = 10*log10(compute_nmse(H_h, H_ls));
        NMSE_NN(iS) = 10*log10(compute_nmse(H_h, H_nn));

        %% STEP 3 — OPTIMAL PHASE SHIFTS
        % The paper uses a SINGLE effective channel vector g = H_full.
        % The RIS-AP and UE-RIS channels are combined into one cascade vector.
        % Phase shift rule (from ris_beamforming_eval.m in the paper):
        %   theta_n = -angle( g_n_estimated )
        % This aligns all N reflected paths coherently at the receiver.
        theta_perf = -angle(H_h);     % [numMCS x 16]  perfect CSI
        theta_ls   = -angle(H_ls);    % [numMCS x 16]  LS-DC estimated
        theta_nn   = -angle(H_nn);    % [numMCS x 16]  NN estimated

        %% STEP 4 — ACHIEVABLE RATE  (matching paper's ris_beamforming_eval.m)
        % R = log2(1 + |g_true' * exp(j*theta)|^2 / noise_pow)
        % noise_pow = 1 (paper convention — channel already encodes SNR via path loss)
        % SNR_dB is encoded in the channel magnitude through beta0 and path loss.
        % The paper passes noise_pow=1 and sweeps SNR by scaling the channel.
        %
        % To sweep SNR properly: scale channel by sqrt(SNR_lin) so that
        % at SNR_lin=1 (0dB) the unscaled channel gives unit-power noise baseline.
        noise_pow = 1;

        % Coherent combining gain = |sum_n g_n * exp(j*theta_n)|^2
        gain_perf = abs(sum(H_h .* exp(1j*theta_perf), 2)).^2;   % [MC x 1]
        gain_ls   = abs(sum(H_h .* exp(1j*theta_ls),   2)).^2;
        gain_nn   = abs(sum(H_h .* exp(1j*theta_nn),   2)).^2;

        % Scale by SNR_lin to sweep SNR (noise_pow stays at 1)
        RATE_perf(iS) = mean(log2(1 + gain_perf * SNR_lin / noise_pow));
        RATE_ls(iS)   = mean(log2(1 + gain_ls   * SNR_lin / noise_pow));
        RATE_nn(iS)   = mean(log2(1 + gain_nn   * SNR_lin / noise_pow));

        % No-RIS baseline: single receive antenna, no beamforming
        % Power = |g_1|^2 (first element, no phase alignment)
        gain_none     = abs(H_h(:,1)).^2;
        RATE_none(iS) = mean(log2(1 + gain_none * SNR_lin / noise_pow));

        BF_GAIN_NN(iS) = RATE_nn(iS) - RATE_none(iS);
        BF_GAIN_LS(iS) = RATE_ls(iS) - RATE_none(iS);

        log_msg(['  SNR=%2d dB | NMSE: LS=%.2f  NN=%.2f dB | '...
            'Rate: perf=%.2f  nn=%.2f  ls=%.2f  noRIS=%.2f'],...
            snr_dB, NMSE_LS(iS), NMSE_NN(iS),...
            RATE_perf(iS), RATE_nn(iS), RATE_ls(iS), RATE_none(iS));

        if snr_dB == STORE_SNR_dB
            H_h_store=H_h;
            H_ls_store=H_ls; H_nn_store=H_nn;
        end

    catch ME
        log_msg('ERROR at SNR=%d dB: %s', snr_dB, ME.message);
        for kk=1:numel(ME.stack)
            log_msg('  >> %s (line %d)', ME.stack(kk).name, ME.stack(kk).line);
        end
    end
end

%% Quick sanity check
if all(isnan(NMSE_LS))
    error('All results NaN. Check that all .m helper files are on the MATLAB path.');
end

%% ================================================================
%% FIGURE 1: PER-ELEMENT PHASE SHIFT PANELS
%% ================================================================
try
    if isempty(H_h_store)
        [H_h_store,y_fb,~,~] = generate_channel(500,N,Nact,fc,d_elem,...
            beta0,d0,d_k_set,alpha_PL,K_users,STORE_SNR_dB,layout);

        [~,~,H_ls_store] = ls_dc_estimator(y_fb,N,Nact,fc,d_elem,K_users,layout);
        Xfb = [real(y_fb),imag(y_fb)];
        uv_fb = nn_predict_angles(netNN,Xfb,u_mean,u_std,v_mean,v_std,...
            X_mean,X_std,N,Nact,layout);
        H_nn_store = interpolate_passive_elements(uv_fb(:,1),uv_fb(:,2),...
            N,Nact,fc,d_elem,y_fb,K_users,layout);
    end

    nT=3; nmc=size(H_h_store,1);
    step=max(1,floor(nmc/(nT+1)));
    tid=min(step*(1:nT),nmc);

    H_list = {H_h_store, H_ls_store, H_nn_store};
    cnames = {'Perfect CSI','LS-DC Estimated','NN Estimated (pre-trained)'};

    figure('Position',[50 50 1000 nT*260]);
    tl=tiledlayout(nT,3,'TileSpacing','compact','Padding','compact');

    for iT=1:nT
        mc=tid(iT);
        for iC=1:3
            nexttile; hold on;
            h_est= H_list{iC}(mc,:).';   % cascade channel estimate [N x 1]

            % PAPER FORMULA: theta_n = -angle(g_n_hat)
            % g_n_hat is the estimated cascade channel (UE->RIS->AP combined)
            theta      = -angle(h_est);
            th_panel   = reshape(theta,Nv,Nh)*180/pi;

            imagesc(th_panel); colormap(gca,'hsv'); caxis([-180 180]);
            for r=1:Nv; for c=1:Nh
                rectangle('Position',[c-0.5 r-0.5 1 1],...
                    'EdgeColor',[0.3 0.3 0.3],'LineWidth',0.6);
            end; end
            for k=1:Nact
                rectangle('Position',[act_cols(k)-0.5 act_rows(k)-0.5 1 1],...
                    'EdgeColor','c','LineWidth',2.8);
            end
            for n=1:N
                [rn,cn]=ind2sub([Nv,Nh],n);
                text(cn,rn,sprintf('%+.0f°',theta(n)*180/pi),...
                    'HorizontalAlignment','center','VerticalAlignment','middle',...
                    'FontName','Times New Roman','FontSize',7.5,...
                    'Color','k','FontWeight','bold');
            end
            cb=colorbar; cb.Label.String='θ_n [deg]';
            cb.Label.FontName='Times New Roman'; cb.FontSize=8;
            set(gca,'XTick',1:Nh,'YTick',1:Nv,'YDir','normal',...
                'XTickLabel',{'c1','c2','c3','c4'},...
                'YTickLabel',{'r1','r2','r3','r4'},...
                'FontName','Times New Roman','FontSize',8,'FontWeight','bold');
            axis equal tight;
            if iT==1; title(cnames{iC},'FontName','Times New Roman',...
                    'FontSize',10,'FontWeight','bold'); end
            if iC==1; ylabel(sprintf('Trial %d',iT),...
                    'FontName','Times New Roman','FontSize',9); end
        end
    end
    title(tl,sprintf(['θ_n = -∠(g_n*·h_n) for all 16 RIS elements  '...
        '(SNR=%d dB, N=%d, center layout)\n'...
        'Cyan = active element (pilot received here)   '...
        'Number = phase applied in degrees'],...
        STORE_SNR_dB,N),'FontName','Times New Roman','FontSize',10);
    print(fullfile(figDir,sprintf('code4_phase_shifts_SNR%02d.png',STORE_SNR_dB)),'-dpng','-r300');
    log_msg('Fig1 saved: phase shifts per element.');
catch ME
    log_msg('Fig1 error: %s',ME.message);
    for kk=1:numel(ME.stack); log_msg('  >> %s (line %d)',ME.stack(kk).name,ME.stack(kk).line); end
end

%% ================================================================
%% FIGURE 2: PHASE ERROR MAP
%% ================================================================
try
    if ~isempty(H_h_store)
        tp = -angle(H_h_store);    % perfect CSI phases  [MC x N]
        tl2= -angle(H_ls_store);   % LS-DC phases
        tn = -angle(H_nn_store);   % NN phases
        el = angle(exp(1j*(tl2-tp))); en=angle(exp(1j*(tn-tp)));
        rms_ls=reshape(sqrt(mean(el.^2,1))*180/pi,Nv,Nh);
        rms_nn=reshape(sqrt(mean(en.^2,1))*180/pi,Nv,Nh);
        clim=[0 max(max(rms_ls(:)),max(rms_nn(:)))*1.05];

        figure('Position',[50 50 820 380]);
        dat_={rms_ls,rms_nn}; nm_={'LS-DC','MLP-NN (pre-trained)'};
        for ip=1:2
            subplot(1,2,ip); hold on;
            imagesc(dat_{ip}); colorbar; colormap(gca,'hot'); caxis(clim);
            for r=1:Nv; for c=1:Nh
                text(c,r,sprintf('%.0f°',dat_{ip}(r,c)),...
                    'HorizontalAlignment','center','VerticalAlignment','middle',...
                    'FontName','Times New Roman','FontSize',9,...
                    'Color','w','FontWeight','bold');
            end; end
            for k=1:Nact
                rectangle('Position',[act_cols(k)-0.5 act_rows(k)-0.5 1 1],...
                    'EdgeColor','c','LineWidth',2.5);
            end
            set(gca,'XTick',1:Nh,'YTick',1:Nv,'YDir','normal',...
                'XTickLabel',{'c1','c2','c3','c4'},...
                'YTickLabel',{'r1','r2','r3','r4'},...
                'FontName','Times New Roman','FontSize',9,'FontWeight','bold');
            axis equal tight;
            title(sprintf('%s  Phase Error RMS [deg]\nSNR=%d dB',nm_{ip},STORE_SNR_dB),...
                'FontName','Times New Roman','FontSize',10);
            xlabel('Col','FontName','Times New Roman','FontSize',10);
            ylabel('Row','FontName','Times New Roman','FontSize',10);
        end
        sgtitle({'Phase Estimation Error per RIS Element  (lower = better)',...
            'Cyan = active element (pilot received here)'},...
            'FontName','Times New Roman','FontSize',11);
        print(fullfile(figDir,'code4_phase_error_map.png'),'-dpng','-r300');
        log_msg('Fig2 saved: phase error map.');
    end
catch ME; log_msg('Fig2 error: %s',ME.message); end

%% ================================================================
%% FIGURE 3: NMSE vs SNR
%% ================================================================
try
    figure('Position',[50 50 700 480]); hold on;
    vl=~isnan(NMSE_LS); vn=~isnan(NMSE_NN);
    if any(vl); plot(SNR_dB_vec(vl),NMSE_LS(vl),'rs--','LineWidth',2,...
            'MarkerSize',8,'DisplayName','LS-DC'); end
    if any(vn); plot(SNR_dB_vec(vn),NMSE_NN(vn),'b^-','LineWidth',2,...
            'MarkerSize',8,'DisplayName','MLP-NN (pre-trained)'); end
    grid on;
    xlabel('SNR [dB]','FontName','Times New Roman','FontSize',12);
    ylabel('NMSE [dB]','FontName','Times New Roman','FontSize',12);
    title('Channel Estimation NMSE vs SNR — Center Layout (N=16)',...
        'FontName','Times New Roman','FontSize',12);
    legend('Location','southwest','FontName','Times New Roman','FontSize',11);
    set(gca,'FontName','Times New Roman','FontSize',11,'FontWeight','bold');
    xlim([SNR_dB_vec(1)-1 SNR_dB_vec(end)+1]);
    print(fullfile(figDir,'code4_nmse_vs_snr.png'),'-dpng','-r300');
    log_msg('Fig3 saved: NMSE vs SNR.');
catch ME; log_msg('Fig3 error: %s',ME.message); end

%% ================================================================
%% FIGURE 4: ACHIEVABLE RATE vs SNR
%% ================================================================
try
    figure('Position',[50 50 750 500]); hold on;
    vp=~isnan(RATE_perf); vn=~isnan(RATE_nn);
    vl=~isnan(RATE_ls);   vno=~isnan(RATE_none);
    if any(vp); plot(SNR_dB_vec(vp),RATE_perf(vp),'k-','LineWidth',2.2,...
            'DisplayName','Perfect CSI (upper bound)'); end
    if any(vn); plot(SNR_dB_vec(vn),RATE_nn(vn),'b^-','LineWidth',2,...
            'MarkerSize',8,'DisplayName','MLP-NN CSI (pre-trained)'); end
    if any(vl); plot(SNR_dB_vec(vl),RATE_ls(vl),'rs--','LineWidth',1.8,...
            'MarkerSize',7,'DisplayName','LS-DC CSI'); end
    if any(vno); plot(SNR_dB_vec(vno),RATE_none(vno),'kd:','LineWidth',1.5,...
            'MarkerSize',7,'DisplayName','No RIS (direct link only)'); end
    grid on;
    xlabel('SNR [dB]','FontName','Times New Roman','FontSize',12);
    ylabel('Achievable Rate [bits/s/Hz]','FontName','Times New Roman','FontSize',12);
    title({'Achievable Rate vs SNR — RIS Center Layout (N=16)',...
        '\theta_n = -\angle(g_n^* \times h_n)  applied to all 16 elements'},...
        'FontName','Times New Roman','FontSize',11);
    legend('Location','northwest','FontName','Times New Roman','FontSize',10);
    set(gca,'FontName','Times New Roman','FontSize',11,'FontWeight','bold');
    xlim([SNR_dB_vec(1)-1 SNR_dB_vec(end)+1]);
    print(fullfile(figDir,'code4_rate_vs_snr.png'),'-dpng','-r300');
    log_msg('Fig4 saved: rate vs SNR.');
catch ME; log_msg('Fig4 error: %s',ME.message); end

%% ================================================================
%% FIGURE 5: BEAMFORMING GAIN vs SNR
%% ================================================================
try
    figure('Position',[50 50 700 450]); hold on;
    vn=~isnan(BF_GAIN_NN); vl=~isnan(BF_GAIN_LS);
    if any(vn); plot(SNR_dB_vec(vn),BF_GAIN_NN(vn),'b^-','LineWidth',2,...
            'MarkerSize',8,'DisplayName','RIS + MLP-NN CSI'); end
    if any(vl); plot(SNR_dB_vec(vl),BF_GAIN_LS(vl),'rs--','LineWidth',1.8,...
            'MarkerSize',7,'DisplayName','RIS + LS-DC CSI'); end
    yline(0,'k--','LineWidth',1.5,'DisplayName','No-RIS baseline');
    grid on;
    xlabel('SNR [dB]','FontName','Times New Roman','FontSize',12);
    ylabel('\DeltaRate over No-RIS [bits/s/Hz]','FontName','Times New Roman','FontSize',12);
    title({'RIS Beamforming Gain — Center Layout (N=16)',...
        'Gain = Rate(with RIS) \minus Rate(direct link only)'},...
        'FontName','Times New Roman','FontSize',11);
    legend('Location','northwest','FontName','Times New Roman','FontSize',10);
    set(gca,'FontName','Times New Roman','FontSize',11,'FontWeight','bold');
    xlim([SNR_dB_vec(1)-1 SNR_dB_vec(end)+1]);
    print(fullfile(figDir,'code4_bf_gain_vs_snr.png'),'-dpng','-r300');
    log_msg('Fig5 saved: BF gain vs SNR.');
catch ME; log_msg('Fig5 error: %s',ME.message); end

%% ================================================================
%% FIGURE 6: COHERENT COMBINING PHASOR DIAGRAM
%% ================================================================
try
    if ~isempty(H_h_store)
        mc  =tid(min(2,numel(tid)));
        h_ex=H_h_store(mc,:).';    % true cascade channel  [N x 1]
        h_ls=H_ls_store(mc,:).';   % LS-DC estimate
        h_nn=H_nn_store(mc,:).';   % NN estimate
        % Paper model: theta_n = -angle(g_n_hat)
        % Contribution of element n = g_n_true * exp(j*theta_n)
        cr = h_ex;                                        % no correction: raw channel
        co = h_ex .* exp(1j*(-angle(h_ex)));             % perfect CSI: all align to real axis
        cl = h_ex .* exp(1j*(-angle(h_ls)));             % LS-DC correction
        cn = h_ex .* exp(1j*(-angle(h_nn)));             % NN correction

        % Normalise for display — actual values are ~1e-7, causing |sum|=0.000
        % We normalise by the max magnitude so arrows are visible and readable.
        % The SHAPE (alignment) is what the diagram shows, not absolute power.
        sc = max(abs([cr;co;cl;cn]));
        if sc < 1e-20; sc = 1; end
        cr=cr/sc; co=co/sc; cl=cl/sc; cn=cn/sc;

        ftitles={'No correction','Perfect CSI','LS-DC','MLP-NN'};
        allc={cr,co,cl,cn};
        alim=max(abs([cr;co]))*1.3; alim=max(alim,0.01);

        figure('Position',[50 50 1050 300]);
        for iC=1:4
            subplot(1,4,iC); hold on;
            cv=allc{iC};
            for n=1:N
                quiver(0,0,real(cv(n)),imag(cv(n)),0,...
                    'Color',[0.65 0.65 0.65],'LineWidth',0.9,'MaxHeadSize',0.8);
            end
            cs=sum(cv);
            quiver(0,0,real(cs),imag(cs),0,'Color','r','LineWidth',2.5,'MaxHeadSize',0.5);
            plot(real(cv),imag(cv),'b.','MarkerSize',10);
            plot(0,0,'k+','MarkerSize',8,'LineWidth',1.5);
            axis equal; grid on;
            xlim([-alim alim]); ylim([-alim alim]);
            xlabel('Re','FontName','Times New Roman','FontSize',9);
            ylabel('Im','FontName','Times New Roman','FontSize',9);
            title({ftitles{iC},sprintf('|sum|=%.3f  (normalised)',abs(cs))},...
                'FontName','Times New Roman','FontSize',9);
            set(gca,'FontName','Times New Roman','FontSize',8,'FontWeight','bold');
        end
        sgtitle({'Coherent Combining: Each grey arrow = one RIS element contribution  (amplitudes normalised for display)',...
            'Red arrow = vector sum   |   No correction: random directions cancel   |   With phase alignment: all arrows point same way'},...
            'FontName','Times New Roman','FontSize',10);
        print(fullfile(figDir,'code4_coherent_combining.png'),'-dpng','-r300');
        log_msg('Fig6 saved: coherent combining phasor diagram.');
    end
catch ME; log_msg('Fig6 error: %s',ME.message); end

%% ================================================================
%% SUMMARY TABLE
%% ================================================================
[~,iS20]=min(abs(SNR_dB_vec-20));
fprintf('\n===== RESULTS SUMMARY — Center Layout (N=%d, N_act=%d) =====\n',N,Nact);
fprintf('%-10s %10s %10s %10s %10s %10s %10s\n',...
    'SNR[dB]','LS-DC[dB]','NN[dB]','R_perf','R_NN','R_LS','R_noRIS');
fprintf('%s\n',repmat('-',1,74));
for iS=1:nSNR
    fprintf('  %-8d %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f\n',...
        SNR_dB_vec(iS),NMSE_LS(iS),NMSE_NN(iS),...
        RATE_perf(iS),RATE_nn(iS),RATE_ls(iS),RATE_none(iS));
end
fprintf('\n--- KEY at SNR=%d dB ---\n',SNR_dB_vec(iS20));
fprintf('  NMSE  : LS-DC=%.2f dB  NN=%.2f dB\n',NMSE_LS(iS20),NMSE_NN(iS20));
fprintf('  Rate  : Perfect=%.2f  NN=%.2f  LS=%.2f  No-RIS=%.2f [bits/s/Hz]\n',...
    RATE_perf(iS20),RATE_nn(iS20),RATE_ls(iS20),RATE_none(iS20));
fprintf('  BF Gain (NN): +%.2f bits/s/Hz over direct link\n',BF_GAIN_NN(iS20));
fprintf('  BF Gain (LS): +%.2f bits/s/Hz over direct link\n',BF_GAIN_LS(iS20));
log_msg('=== CODE 4 COMPLETE ===');
fprintf('\nAll figures saved in figures/ folder.\n');
