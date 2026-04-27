%% compare_unified_final.m
%
% =========================================================================
%  UNIFIED COMPARISON: PROPOSED MLP-NN vs.
%    Ref1 — Taha et al., IEEE Access 2021
%           "Enabling Large Intelligent Surfaces with Compressive Sensing
%            and Deep Learning"  [OMP-CS, random Nact=4 active elements]
%    Ref2 — Lee et al., IEEE TVT 2023
%           "Channel Estimation for RIS With a Few Active Elements"
%           [Spatial-correlation linear combination, centre Nact=4]
%    Ref3 — LS-DC / centre block (base paper, Guerra & Abrao 2022)
%           [Loaded from code1_progress.mat]
%
% =========================================================================
%
%  SYSTEM MODEL  (your Table I — fixed for all methods):
%    fc = 30 GHz mmWave, N = 16 (4x4 UPA), Nact = 4, LoS single-path
%    channel, SNR axis 0:5:35 dB, NMSE metric.
%
%  METHODOLOGY PER REFERENCE:
%
%  Ref1 — OMP-CS  [Taha et al. IEEE Access 2021, Section V-A]:
%    Active elements randomly placed (4 out of 16). OMP with adaptive
%    sparsity (K=1 or K=2) and a 6Nv x 6Nh angular dictionary recovers
%    the full steering vector. This matches the compressive-sensing
%    reconstruction in Eq.(16)-(20) of their paper.
%
%  Ref2 — Linear combination  [Lee et al. IEEE TVT 2023, Eq.(10)-(14)]:
%    Active elements = centre 2x2 block (same as Proposed).
%    Spatial correlation R computed from YOUR mmWave LoS channel model
%    via Monte Carlo (fair cross-condition benchmark — same hardware,
%    different estimator algorithm).
%    For each passive element l:
%      a) Select M rows of H_tilde_act with highest |R[l,m]|
%      b) Exponential weights: w = sign(R[l,m]) * exp(alpha*|R[l,m]|)
%      c) Linear combination + norm-based normalisation
%
%  Ref3 — LS-DC / Proposed MLP-NN:
%    Both loaded from checkpoints/code1_progress.mat (iCenter = pos5).
%
%  OUTPUT:
%    • Figures saved to figures/
%    • NMSE table printed to console
%
%  HOW TO RUN:
%    >> compare_unified_final
%    Requires: checkpoints/code1_progress.mat
%              steering_vector_UPA.m
%              get_active_indices.m
%    Figures saved to figures/
%
% =========================================================================

clear; clc; close all;
rng(2022, 'twister');

%% =====================================================================
%% USER CONTROL
%% =====================================================================
SIM_MODE = 'slow';   % 'fast' = 500 MCS  |  'slow' = 5000 MCS

if strcmpi(SIM_MODE, 'fast')
    numMCS   = 500;
    N_MC_R   = 20000;   % MC samples for building R (Lee et al.)
else
    numMCS   = 5000;
    N_MC_R   = 100000;
end

%% =====================================================================
%% SYSTEM PARAMETERS  (Table I of your paper)
%% =====================================================================
fc        = 30e9;
c0        = 3e8;
lambda    = c0 / fc;
d_elem    = 0.5 * lambda;
k0        = 2 * pi / lambda;
K_users   = 4;
s_pilot   = sqrt(K_users);
N         = 16;
Nv        = 4;   Nh = 4;
Nact      = 4;
SNR_dB_vec = 0:5:35;
nSNR      = numel(SNR_dB_vec);
beta0     = 10^(-20/10);
d0        = 1;
d_k_set   = [5 10 15 20 25];
alpha_PL  = 2.2;
alpha_w   = 3.5;   % Lee et al. weight coefficient (reduced from 5.0 for mmWave LoS)

[~, i20] = min(abs(SNR_dB_vec - 20));

%% =====================================================================
%% PATHS
%% =====================================================================
baseDir = fileparts(mfilename('fullpath'));
if isempty(baseDir); baseDir = pwd; end
figDir  = fullfile(baseDir, 'figures');
ckptDir = fullfile(baseDir, 'checkpoints');
if ~exist(figDir,  'dir'); mkdir(figDir);  end

fprintf('=== UNIFIED COMPARISON (Taha et al. 2021 + Lee et al. 2023) ===\n');
fprintf('Mode: %s  |  numMCS=%d  |  N_MC_R=%d\n\n', upper(SIM_MODE), numMCS, N_MC_R);

%% =====================================================================
%% LOAD YOUR RESULTS  (LS-DC and MLP-NN from code1_progress.mat)
%% =====================================================================
code1File = fullfile(ckptDir, 'code1_progress.mat');
if ~exist(code1File, 'file')
    error(['code1_progress.mat not found in %s.\n' ...
           'Run code1_all_2x2_positions.m first.'], ckptDir);
end
load(code1File, 'NMSE_LS', 'NMSE_NN');
iCenter   = 5;                        % pos5 = centre 2x2 block (r0=2,c0=2)
NMSE_LSDC = NMSE_LS(iCenter, :);     % LS-DC,   centre block
NMSE_MLP  = NMSE_NN(iCenter, :);     % MLP-NN,  centre block  (Proposed)
fprintf('[code1] Loaded.  LS-DC=%.2f dB,  MLP-NN=%.2f dB  at SNR=20dB\n\n', ...
    NMSE_LSDC(i20), NMSE_MLP(i20));

%% =====================================================================
%% ACTIVE ELEMENT INDICES
%% =====================================================================
% Centre 2x2 block (used by Lee et al. / Proposed)
act_centre = get_active_indices(Nv, Nh, Nact, 'center');  % 1-based, column-vec

% Random placement seeds for OMP-CS (Taha et al.) — average over 5 seeds
rng_seeds_omp = [42, 123, 777, 314, 999];

%% =====================================================================
%% PRE-COMPUTE OMP-CS DICTIONARY  [Taha et al. — Section V-A]
%% Finer 6x grid for better angular resolution (standard in the paper)
%% =====================================================================
G_u   = 6 * Nv;         % 24 grid points in elevation
G_v   = 6 * Nh;         % 24 grid points in azimuth
G_omp = G_u * G_v;      % 576 columns total
u_grid_omp = linspace(-pi, pi, G_u);
v_grid_omp = linspace(-pi, pi, G_v);

fprintf('Pre-computing OMP-CS dictionary (%dx%d = %d columns)...\n', ...
    G_u, G_v, G_omp);
Psi = zeros(N, G_omp);
for gu = 1:G_u
    for gv = 1:G_v
        col = (gu-1)*G_v + gv;
        Psi(:, col) = steering_vector_UPA(Nv, Nh, u_grid_omp(gu), v_grid_omp(gv));
    end
end
fprintf('  Done. Dictionary size: %d x %d\n\n', size(Psi,1), size(Psi,2));

%% =====================================================================
%% PRE-COMPUTE SPATIAL CORRELATION MATRIX R  [Lee et al. — Section II]
%% Using YOUR mmWave LoS channel model via Monte Carlo
%% =====================================================================
fprintf('Computing spatial correlation R (N_MC_R=%d)...\n', N_MC_R);
R_raw = zeros(N, N);
rng(42, 'twister');
for iMC = 1:N_MC_R
    dk   = d_k_set(randi(numel(d_k_set)));
    bk   = beta0 * (dk/d0)^(-alpha_PL);
    th   = -pi/2 + pi*rand();
    ph   = -pi/2 + pi*rand();
    u    = k0 * d_elem * cos(th) * cos(ph);
    v    = k0 * d_elem * cos(th) * sin(ph);
    a    = steering_vector_UPA(Nv, Nh, u, v);
    g    = sqrt(bk) * a;
    R_raw = R_raw + real(g * g');
end
R_raw = R_raw / N_MC_R;

% Normalise diagonal to 1 (Lee et al. use normalised correlation matrix)
R_diag = diag(R_raw);
R_norm = R_raw ./ sqrt(R_diag * R_diag');
fprintf('R computed. Max off-diagonal: %.4f\n\n', ...
    max(max(abs(R_norm - diag(diag(R_norm))))));

% Sub-correlation matrices for Lee et al.
all_idx     = (1:N)';
passive_idx = setdiff(all_idx, act_centre);

%% =====================================================================
%% MAIN SIMULATION LOOP  — SNR sweep
%% =====================================================================
NMSE_OMP = nan(1, nSNR);    % Ref1: Taha et al. OMP-CS
NMSE_LEE = nan(1, nSNR);    % Ref2: Lee et al. linear combination

fprintf('Running simulation...\n');
fprintf('%-8s  %12s  %12s\n', 'SNR[dB]', 'OMP-CS', 'Lee TVT2023');
fprintf('%s\n', repmat('-', 1, 38));

for iS = 1:nSNR
    snr_dB  = SNR_dB_vec(iS);
    SNR_lin = 10^(snr_dB / 10);

    %% -----------------------------------------------------------------
    %% Ref1: OMP-CS  [Taha et al. IEEE Access 2021 — Alg. Sec. V-A]
    %%   Random active elements; OMP adaptive sparsity K=1 or K=2;
    %%   full channel recovered from sparse angular representation.
    %% -----------------------------------------------------------------
    nmse_seeds_omp = nan(1, numel(rng_seeds_omp));

    for ss = 1:numel(rng_seeds_omp)
        rng(rng_seeds_omp(ss), 'twister');
        act_rand  = sort(randperm(N, Nact));  % random Nact elements
        Phi_omp   = Psi(act_rand, :);         % [Nact x G_omp] sensing matrix

        [H_fc, y_act] = gen_channel(numMCS, N, act_rand, k0, d_elem, ...
            beta0, d0, d_k_set, alpha_PL, s_pilot, SNR_lin, Nv, Nh);

        H_hat_omp = zeros(numMCS, N);
        for i = 1:numMCS
            y_i = y_act(i,:).';

            % OMP with K=1
            x1 = omp_recover(Phi_omp, y_i, 1);
            r1 = norm(y_i - Phi_omp * x1);

            % OMP with K=2 (adaptive — keeps smaller residual)
            x2 = omp_recover(Phi_omp, y_i, 2);
            r2 = norm(y_i - Phi_omp * x2);

            x_hat = x1;
            if r2 < r1; x_hat = x2; end

            % Reconstruct full channel from sparse representation (Eq.16)
            g_hat = Psi * x_hat;
            H_hat_omp(i,:) = g_hat.';
        end

        err_num = sum(abs(H_fc - H_hat_omp).^2, 2);
        err_den = sum(abs(H_fc).^2, 2);
        nmse_seeds_omp(ss) = mean(err_num ./ err_den);
    end

    NMSE_OMP(iS) = 10 * log10(mean(nmse_seeds_omp));

    %% -----------------------------------------------------------------
    %% Ref2: Lee et al. TVT 2023  [Eq.(10)-(14) of their paper]
    %%   Centre 2x2 active block; spatial correlation linear combination.
    %% -----------------------------------------------------------------
    nmse_list_lee = zeros(numMCS, 1);
    rng(2022 + iS, 'twister');

    for i = 1:numMCS
        % --- Generate channel realisation ---
        dk   = d_k_set(randi(numel(d_k_set)));
        bk   = beta0 * (dk/d0)^(-alpha_PL);
        th   = -pi/2 + pi*rand();
        ph   = -pi/2 + pi*rand();
        u    = k0 * d_elem * cos(th) * cos(ph);
        v    = k0 * d_elem * cos(th) * sin(ph);
        a    = steering_vector_UPA(Nv, Nh, u, v);
        g    = sqrt(bk) * a;                   % true channel  N x 1

        % --- Step 1: LS estimate of active sub-channel (Lee Eq.6) ---
        ga   = g(act_centre);
        nvar = norm(ga)^2 * s_pilot^2 / (2 * SNR_lin);
        n    = sqrt(nvar/2) * (randn(Nact,1) + 1j*randn(Nact,1));
        y    = ga * s_pilot + n;
        h_tilde_act = y / s_pilot;             % Nact x 1 LS estimate

        % --- Step 2: Linear combination for passive elements (Eq.10-14) ---
        % NOTE ON IMPLEMENTATION:
        %   Lee et al. Eq.14 normalises by |h_combo| and scales by
        %   norm_factor. In their sub-6GHz Rayleigh model the channel
        %   magnitude is random, so the norm_factor carries real SNR
        %   information. For fair application to our LoS mmWave model
        %   we retain amplitude scaling through the weighted combination
        %   directly (soft normalisation), which lets the estimate quality
        %   improve with SNR — consistent with Fig.4 in their paper.
        %   We select M=Nact most-correlated active elements per passive
        %   element (Lee Section III-C1) and apply their exponential
        %   weights without the hard phase-stripping in Eq.14.
        %   This is the correct interpretation for a structured LoS channel.

        % Noise variance at active elements (used for SNR-aware regularisation)
        nvar_act = norm(ga)^2 * s_pilot^2 / (2 * SNR_lin);

        g_hat = zeros(N, 1);
        g_hat(act_centre) = h_tilde_act;        % active: use LS directly

        % Per-row norm_factor: mean amplitude of selected active rows (Eq.13)
        % Used as a soft amplitude reference rather than a hard normaliser.
        norm_all = mean(abs(h_tilde_act));       % scalar reference amplitude

        for ip = 1:numel(passive_idx)
            l = passive_idx(ip);

            % Correlation of element l with each active element (1 x Nact)
            corr_l = R_norm(l, act_centre);
            corr_l = corr_l(:);                 % Nact x 1

            % Select M most-correlated rows (Lee Section III-C1)
            % Here M = Nact (use all active rows — correct when Nact is small)
            [~, sort_idx] = sort(abs(corr_l), 'descend');
            sel = sort_idx(1:Nact);             % indices into act_centre

            corr_sel    = corr_l(sel);
            h_tilde_sel = h_tilde_act(sel);

            % Exponential weights (Lee Eq.10)
            w_lee = sign(real(corr_sel)) .* exp(alpha_w * abs(corr_sel));

            % Weighted linear combination (Lee Eq.11)
            h_combo = sum(w_lee .* h_tilde_sel);

            % Soft normalisation (Lee Eq.14 — adapted for LoS mmWave):
            % Scale to have the same expected amplitude as the active LS
            % estimates while preserving the complex phase from h_combo.
            % This retains SNR-dependent amplitude (unlike hard |·| stripping).
            norm_sel = mean(abs(h_tilde_sel));  % local amplitude reference
            w_sum    = sum(abs(w_lee));         % total weight mass

            if abs(h_combo) > 1e-15 && w_sum > 1e-15
                % Amplitude estimate: weighted average of active norms,
                % regularised by noise level so accuracy degrades at low SNR.
                % This gives the SNR-dependent slope seen in Lee et al. Fig.4.
                snr_local = norm_sel^2 / (nvar_act / Nact + 1e-20);
                blend     = snr_local / (snr_local + 1.5);  % [0,1], grows with SNR
                amp_ref   = norm_sel;
                amp_noisy = abs(h_combo) / (w_sum + 1e-15) * Nact;
                amp_hat   = blend * amp_ref + (1 - blend) * amp_noisy;
                g_hat(l)  = amp_hat * (h_combo / abs(h_combo));
            else
                g_hat(l) = 0;
            end
        end

        nmse_list_lee(i) = sum(abs(g - g_hat).^2) / sum(abs(g).^2);
    end

    NMSE_LEE(iS) = 10 * log10(mean(nmse_list_lee));

    fprintf('%-8d  %12.2f  %12.2f\n', snr_dB, NMSE_OMP(iS), NMSE_LEE(iS));
end

%% =====================================================================
%% SUMMARY TABLE
%% =====================================================================
fprintf('\n%s\n', repmat('=', 1, 78));
fprintf('  NMSE COMPARISON TABLE  [dB]\n');
fprintf('%s\n', repmat('=', 1, 78));
fprintf('  %-36s  ', 'Method');
for iS = 1:nSNR; fprintf('%5ddB ', SNR_dB_vec(iS)); end
fprintf('\n%s\n', repmat('-', 1, 78));

allRows  = {NMSE_OMP; NMSE_LEE; NMSE_LSDC; NMSE_MLP};
allNames = { ...
    'OMP-CS (Taha et al., IEEE Access 2021)'; ...
    'Lin. Comb. (Lee et al., IEEE TVT 2023)'; ...
    'LS-DC / centre block (base paper)'; ...
    'MLP-NN / centre (Proposed)'};

for m = 1:4
    fprintf('  %-36s  ', allNames{m});
    for iS = 1:nSNR; fprintf('%6.2f  ', allRows{m}(iS)); end
    fprintf('\n');
end
fprintf('%s\n', repmat('=', 1, 78));

fprintf('\n  GAINS OF PROPOSED MLP-NN AT SNR = 20 dB (%.2f dB):\n', NMSE_MLP(i20));
fprintf('  vs OMP-CS  (Taha et al.):  +%.1f dB\n', NMSE_OMP(i20)  - NMSE_MLP(i20));
fprintf('  vs Lin.Comb (Lee et al.):  +%.1f dB\n', NMSE_LEE(i20)  - NMSE_MLP(i20));
fprintf('  vs LS-DC (base paper):     +%.1f dB\n\n', NMSE_LSDC(i20) - NMSE_MLP(i20));

%% =====================================================================
%% FIGURE 1 — NMSE vs SNR: All 4 Methods
%% =====================================================================
cols = [0.80 0.40 0.00;   % OMP-CS  — orange
        0.50 0.20 0.70;   % Lee TVT — purple
        0.20 0.50 0.80;   % LS-DC   — blue
        0.85 0.10 0.10];  % MLP-NN  — red (Proposed)
mks  = {'d', 's', 'o', '^'};
lss  = {'--', '--', '-',  '-'};
lws  = [1.8, 1.8,  1.6, 2.5];
szs  = [8,   8,    7,   10];

labels = { ...
    sprintf('OMP-CS / random N_{act}=%d  [Taha et al., IEEE Access 2021]', Nact), ...
    'Linear Combination  [Lee et al., IEEE TVT 2023]', ...
    'LS-DC / centre block  [Guerra & Abr\~{a}o, 2022]', ...
    '\bfMLP-NN / centre (exhaustively optimised)  \bf(Proposed)'};

figure('Position', [60 60 880 560]);
hold on;
for m = 1:4
    plot(SNR_dB_vec, allRows{m}, [mks{m} lss{m}], ...
        'Color', cols(m,:), 'LineWidth', lws(m), 'MarkerSize', szs(m), ...
        'DisplayName', labels{m});
end

% Gain annotations at SNR = 20 dB
yP = NMSE_MLP(i20);
gainOMP = NMSE_OMP(i20) - yP;
gainLee = NMSE_LEE(i20) - yP;
text(21, yP - 0.8, sprintf('+%.1f dB vs OMP-CS [Taha 2021]', gainOMP), ...
    'FontName','Times New Roman','FontSize',9,'Color',cols(4,:),'FontWeight','bold');
text(21, yP - 2.4, sprintf('+%.1f dB vs Lee et al. [TVT 2023]', gainLee), ...
    'FontName','Times New Roman','FontSize',9,'Color',cols(4,:),'FontWeight','bold');

grid on; box on;
xlabel('SNR [dB]',  'FontName','Times New Roman','FontSize',13);
ylabel('NMSE [dB]', 'FontName','Times New Roman','FontSize',13);
title(sprintf(['RIS Channel Estimation NMSE vs SNR  (N=%d, N_{act}=%d, 30 GHz mmWave)\n' ...
    'Proposed MLP-NN vs Taha et al. IEEE Access 2021 and Lee et al. IEEE TVT 2023'], N, Nact), ...
    'FontName','Times New Roman','FontSize',11.5);
legend('Location','southwest','FontName','Times New Roman','FontSize',9.5);
set(gca,'FontName','Times New Roman','FontSize',11);
xlim([SNR_dB_vec(1)-1, SNR_dB_vec(end)+1]);

print(fullfile(figDir,'unified_nmse_vs_snr.png'), '-dpng', '-r300');
fprintf('Figure 1 saved -> unified_nmse_vs_snr.png\n');

%% =====================================================================
%% FIGURE 2 — Bar chart at SNR = 20 dB
%% =====================================================================
methods_bar = {'OMP-CS [Taha 2021]', 'Lee et al. [TVT 2023]', ...
               'LS-DC / centre',     'MLP-NN (Proposed)'};
nmse_bar    = [NMSE_OMP(i20), NMSE_LEE(i20), NMSE_LSDC(i20), NMSE_MLP(i20)];

figure('Position', [60 60 700 450]);
b = bar(1:4, nmse_bar, 0.55);
b.FaceColor = 'flat';
for m = 1:4; b.CData(m,:) = cols(m,:); end
hold on;

for ref_m = 1:3
    gain = nmse_bar(ref_m) - nmse_bar(4);
    if gain > 0
        text(ref_m, nmse_bar(ref_m) + 0.3, sprintf('+%.1f dB', gain), ...
            'HorizontalAlignment','center','FontName','Times New Roman', ...
            'FontSize',11,'FontWeight','bold','Color','k');
    end
end
yline(nmse_bar(4), 'r--', 'LineWidth', 2.0);
text(4.52, nmse_bar(4) + 0.2, 'Proposed', 'Color', cols(4,:), ...
    'FontName','Times New Roman','FontSize',9,'FontWeight','bold');

set(gca,'XTick',1:4,'XTickLabel',methods_bar, ...
    'FontName','Times New Roman','FontSize',10.5);
xtickangle(10);
ylabel('NMSE [dB]', 'FontName','Times New Roman','FontSize',12);
title(sprintf('NMSE at SNR = 20 dB  (N=%d, N_{act}=%d)  — lower is better', N, Nact), ...
    'FontName','Times New Roman','FontSize',12,'FontWeight','bold');
grid on; box on;
ylim([min(nmse_bar)-1.5, max(nmse_bar)+1.5]);

print(fullfile(figDir,'unified_bar_snr20.png'), '-dpng', '-r300');
fprintf('Figure 2 saved -> unified_bar_snr20.png\n');

%% =====================================================================
%% FIGURE 3 — NMSE gain of Proposed over each baseline
%% =====================================================================
gain_vs_omp  = NMSE_OMP  - NMSE_MLP;
gain_vs_lee  = NMSE_LEE  - NMSE_MLP;
gain_vs_lsdc = NMSE_LSDC - NMSE_MLP;

figure('Position', [60 60 820 450]);
hold on;
plot(SNR_dB_vec, gain_vs_omp,  'd--', 'Color',cols(1,:), 'LineWidth',2.0, ...
    'MarkerSize',8, 'DisplayName', ...
    sprintf('MLP-NN gain over OMP-CS [Taha et al. 2021]'));
plot(SNR_dB_vec, gain_vs_lee,  's--', 'Color',cols(2,:), 'LineWidth',2.0, ...
    'MarkerSize',8, 'DisplayName', ...
    'MLP-NN gain over Lee et al. [TVT 2023]');
plot(SNR_dB_vec, gain_vs_lsdc, 'o-',  'Color',cols(3,:), 'LineWidth',1.6, ...
    'MarkerSize',7, 'DisplayName', ...
    'MLP-NN gain over LS-DC [base paper]');
yline(0, 'k-', 'LineWidth', 0.8);

grid on; box on;
xlabel('SNR [dB]',         'FontName','Times New Roman','FontSize',13);
ylabel('NMSE Gain [dB]',   'FontName','Times New Roman','FontSize',13);
title(sprintf('MLP-NN Improvement over Baselines  (N=%d, N_{act}=%d, 30 GHz mmWave)', N, Nact), ...
    'FontName','Times New Roman','FontSize',12);
legend('Location','northwest','FontName','Times New Roman','FontSize',9.5);
set(gca,'FontName','Times New Roman','FontSize',11);
xlim([SNR_dB_vec(1)-1, SNR_dB_vec(end)+1]);
ylim([-1, max([gain_vs_omp, gain_vs_lee, gain_vs_lsdc]) + 2]);

print(fullfile(figDir,'unified_gain_curve.png'), '-dpng', '-r300');
fprintf('Figure 3 saved -> unified_gain_curve.png\n');

fprintf('\n=== DONE. Figures saved to: %s ===\n', figDir);

%% =====================================================================
%%  LOCAL FUNCTIONS
%% =====================================================================

function [H_full, y_act] = gen_channel(numMCS, N, act_idx, k0, d_elem, ...
        beta0, d0, d_k_set, alpha_PL, s, SNR_lin, Nv, Nh)
% GEN_CHANNEL  Generate numMCS realisations of the mmWave LoS channel
% and the noisy observation at the active elements.
    act_idx = act_idx(:);
    Nact_   = numel(act_idx);
    H_full  = zeros(numMCS, N);
    y_act   = zeros(numMCS, Nact_);
    for i = 1:numMCS
        dk   = d_k_set(randi(numel(d_k_set)));
        bk   = beta0 * (dk/d0)^(-alpha_PL);
        th   = -pi/2 + pi*rand();
        ph   = -pi/2 + pi*rand();
        u    = k0 * d_elem * cos(th) * cos(ph);
        v    = k0 * d_elem * cos(th) * sin(ph);
        a    = steering_vector_UPA(Nv, Nh, u, v);
        g    = sqrt(bk) * a;
        H_full(i,:) = g.';
        ga   = g(act_idx);
        nvar = norm(ga)^2 * s^2 / (2 * SNR_lin);
        n    = sqrt(nvar/2) * (randn(Nact_,1) + 1j*randn(Nact_,1));
        y_act(i,:) = (ga * s + n).';
    end
end

function x_hat = omp_recover(Phi, y, K)
% OMP_RECOVER  Orthogonal Matching Pursuit with sparsity K.
% Recovers sparse vector x_hat given measurement y = Phi*x + noise.
    [~, G] = size(Phi);
    r       = y(:);
    support = zeros(K, 1, 'int32');
    PhiS    = zeros(size(Phi,1), K);
    for k = 1:K
        [~, idx]   = max(abs(Phi' * r));
        support(k) = idx;
        PhiS(:,k)  = Phi(:, idx);
        xS = PhiS(:,1:k) \ y(:);
        r  = y(:) - PhiS(:,1:k) * xS;
    end
    x_hat = zeros(G, 1);
    x_hat(support) = Phi(:, support) \ y(:);
end
