function [best_layout, best_nmse, results] = optimal_layout_selection(...
    N, Nact, fc, d_elem, beta0, d0, d_k_set, alpha_PL, K_users, SNR_dB_range, numMCS)
%OPTIMAL_LAYOUT_SELECTION  Contribution 1: Systematic evaluation of six
%   candidate active-element layouts on an N-element RIS panel, and
%   automated selection of the configuration that minimises NMSE.
%
%   [best_layout, best_nmse, results] = OPTIMAL_LAYOUT_SELECTION(
%       N, Nact, fc, d_elem, beta0, d0, d_k_set, alpha_PL,
%       K_users, SNR_dB_range, numMCS)
%
%   Layouts evaluated
%   -----------------
%   'center'    : Nact elements in a square block at the array centre.
%   'corners'   : one element near each corner, remainder distributed.
%   'cross'     : elements along horizontal and vertical centre lines.
%   'edges'     : elements uniformly distributed on the outer border.
%   'diagonal'  : elements along the main diagonal.
%   'random'    : random placement (averaged over 5 realisations).
%
%   Outputs
%   -------
%   best_layout : string, name of the layout with lowest average NMSE.
%   best_nmse   : scalar NMSE achieved by best_layout (dB), averaged over SNR.
%   results     : struct with fields layout_names, nmse_db (layouts × SNRs).

if nargin < 11; numMCS = 500; end

layout_names = {'center','corners','cross','edges','diagonal','random'};
nLayouts = numel(layout_names);
nSNR     = numel(SNR_dB_range);

nmse_db = zeros(nLayouts, nSNR);

Nv = round(sqrt(N));
Nh = Nv;
side_act = round(sqrt(Nact));

for li = 1:nLayouts
    lname = layout_names{li};
    fprintf('\n=== Layout: %s ===\n', lname);

    for si = 1:nSNR
        snr = SNR_dB_range(si);

        % Accumulate NMSE over random seeds (random layout: 5 seeds)
        n_seeds = 1;
        if strcmp(lname,'random'); n_seeds = 5; end
        nmse_acc = 0;

        for seed = 1:n_seeds
            % Get active element indices for this layout
            act_idx = get_layout_indices(lname, Nv, Nh, Nact, side_act, seed);

            % Generate channel realisations
            [H_full, y_act, ~, ~] = generate_channel_layout(numMCS, N, act_idx, ...
                fc, d_elem, beta0, d0, d_k_set, alpha_PL, K_users, snr);

            % LS-DC estimation using this layout
            [u_hat, v_hat] = ls_dc_layout(y_act, N, act_idx, K_users);

            % Interpolate full channel
            H_hat = interpolate_from_layout(u_hat, v_hat, N, act_idx, y_act, K_users);

            nmse_acc = nmse_acc + 10*log10(compute_nmse(H_full, H_hat));
        end

        nmse_db(li, si) = nmse_acc / n_seeds;
        fprintf('  SNR=%+3d dB  NMSE=%.2f dB\n', snr, nmse_db(li,si));
    end
end

%% Select best layout (lowest mean NMSE across SNR range)
mean_nmse = mean(nmse_db, 2);
[best_nmse, best_idx] = min(mean_nmse);
best_layout = layout_names{best_idx};

fprintf('\n--- Layout Selection Result ---\n');
for li = 1:nLayouts
    flag = '';
    if li == best_idx; flag = ' <-- BEST'; end
    fprintf('  %-10s  mean NMSE = %.2f dB%s\n', layout_names{li}, mean_nmse(li), flag);
end
fprintf('Optimal layout: %s  (%.2f dB)\n\n', best_layout, best_nmse);

%% Plot
figure('Name','Contribution 1 - Layout NMSE Comparison');
styles = {'-o','-s','-^','-d','-v','-p'};
colors = lines(nLayouts);
hold on; grid on;
for li = 1:nLayouts
    plot(SNR_dB_range, nmse_db(li,:), styles{li}, ...
        'Color', colors(li,:), 'LineWidth', 1.8, 'MarkerSize', 7, ...
        'DisplayName', layout_names{li});
end
xlabel('SNR (dB)'); ylabel('NMSE (dB)');
title(sprintf('Contribution 1: Active Layout NMSE  |  N=%d, Nact=%d', N, Nact));
legend('Location','northeast'); set(gca,'FontSize',12);
hold off;

%% Store results
results.layout_names = layout_names;
results.nmse_db      = nmse_db;
results.mean_nmse    = mean_nmse;
results.SNR_range    = SNR_dB_range;
end

% =========================================================================
%  LOCAL HELPERS
% =========================================================================

function idx = get_layout_indices(name, Nv, Nh, Nact, side_act, seed)
%GET_LAYOUT_INDICES  Return linear indices (in [Nv x Nh] column-major order)
%   for 'Nact' active elements under the chosen layout.

switch name

    case 'center'
        v_s = floor((Nv - side_act)/2) + 1;
        h_s = floor((Nh - side_act)/2) + 1;
        [HH, VV] = meshgrid(h_s:h_s+side_act-1, v_s:v_s+side_act-1);
        idx = sub2ind([Nv,Nh], VV(:), HH(:));

    case 'corners'
        % Place one element at each corner; fill remaining near corners
        corner_v = [1, 1, Nv, Nv];
        corner_h = [1, Nh, 1, Nh];
        idx = unique(sub2ind([Nv,Nh], corner_v(:), corner_h(:)));
        % Fill extra elements spiraling inward from corners
        candidates = [];
        for off = 1:max(Nv,Nh)
            for ci = 1:4
                v0 = corner_v(ci); h0 = corner_h(ci);
                dv = sign(Nv/2+0.5 - v0); dh = sign(Nh/2+0.5 - h0);
                nv = v0 + off*dv; nh = h0 + off*dh;
                if nv>=1&&nv<=Nv&&nh>=1&&nh<=Nh
                    candidates(end+1) = sub2ind([Nv,Nh], nv, nh); %#ok
                end
            end
            if numel(idx) + numel(unique(candidates)) >= Nact; break; end
        end
        idx = unique([idx; candidates(:)]);
        idx = idx(1:Nact);

    case 'cross'
        % Horizontal centre row + vertical centre column
        mid_v = ceil(Nv/2);  mid_h = ceil(Nh/2);
        row_h = 1:Nh;
        col_v = 1:Nv;
        all_v = [repmat(mid_v,1,Nh), col_v, ];
        all_h = [row_h, repmat(mid_h,1,Nv)];
        raw = unique(sub2ind([Nv,Nh], all_v, all_h));
        % Pick Nact elements uniformly from cross
        sel = round(linspace(1,numel(raw),Nact));
        idx = raw(unique(sel));
        if numel(idx) < Nact
            extra = setdiff(raw, idx);
            idx = [idx; extra(1:Nact-numel(idx))];
        end
        idx = idx(1:Nact);

    case 'edges'
        % Elements on the outer border of the array
        border_v = [ones(1,Nh), Nv*ones(1,Nh), 2:Nv-1, 2:Nv-1];
        border_h = [1:Nh, 1:Nh, ones(1,Nv-2), Nh*ones(1,Nv-2)];
        raw = unique(sub2ind([Nv,Nh], border_v, border_h));
        sel = round(linspace(1,numel(raw),Nact));
        idx = raw(unique(sel));
        if numel(idx) < Nact
            extra = setdiff(raw, idx);
            idx = [idx; extra(1:Nact-numel(idx))];
        end
        idx = idx(1:Nact);

    case 'diagonal'
        % Main diagonal elements
        diag_len = min(Nv, Nh);
        dv = round(linspace(1, Nv, Nact));
        dh = round(linspace(1, Nh, Nact));
        idx = unique(sub2ind([Nv,Nh], dv(:), dh(:)));
        % Pad if needed
        if numel(idx) < Nact
            all_idx = 1:Nv*Nh;
            extra = setdiff(all_idx, idx);
            idx = [idx; extra(1:Nact-numel(idx))'];
        end
        idx = idx(1:Nact);

    case 'random'
        rng(seed * 42 + 7);
        perm = randperm(Nv*Nh);
        idx  = sort(perm(1:Nact))';

    otherwise
        error('Unknown layout: %s', name);
end

idx = sort(idx(:));
if numel(idx) ~= Nact
    error('Layout "%s" produced %d indices instead of %d', name, numel(idx), Nact);
end
end

% -------------------------------------------------------------------------
function [H_full, y_act, u_true, v_true] = generate_channel_layout(...
    numMCS, N, act_idx, fc, d_elem, beta0, d0, d_k_set, alpha_PL, K_users, SNR_dB)
%GENERATE_CHANNEL_LAYOUT  Like generate_channel but uses arbitrary act_idx.

c0 = 3e8;  lambda = c0/fc;
Nv = round(sqrt(N));

H_full = zeros(numMCS, N);
y_act  = zeros(numMCS, numel(act_idx));
u_true = zeros(numMCS,1);
v_true = zeros(numMCS,1);

SNR_lin = 10^(SNR_dB/10);
s = sqrt(K_users);

for i = 1:numMCS
    dk = d_k_set(randi(numel(d_k_set)));
    beta_k = beta0 * (dk/d0)^(-alpha_PL);
    theta = -pi/2 + pi*rand();
    phi   = -pi/2 + pi*rand();
    k0 = 2*pi/lambda;
    u = k0*d_elem*cos(theta)*cos(phi);
    v = k0*d_elem*cos(theta)*sin(phi);
    u_true(i) = u;  v_true(i) = v;

    a = steering_vector_UPA(Nv, Nv, u, v);
    g = sqrt(beta_k)*a;
    H_full(i,:) = g.';

    g_act = g(act_idx);
    noise_var = (norm(g_act)^2 * s^2) / SNR_lin;
    n = sqrt(noise_var/2)*(randn(size(g_act))+1j*randn(size(g_act)));
    y_act(i,:) = (g_act*s + n).';
end
end

% -------------------------------------------------------------------------
function [u_hat, v_hat] = ls_dc_layout(y_act, N, act_idx, K_users)
%LS_DC_LAYOUT  LS-DC estimator for an arbitrary (non-square-block) layout.
%   Uses the mean phase slope over all adjacent pairs in the layout.

Nv = round(sqrt(N));
numMCS = size(y_act,1);
s = sqrt(K_users);

[row_idx, col_idx] = ind2sub([Nv,Nv], act_idx);

u_hat = zeros(numMCS,1);
v_hat = zeros(numMCS,1);

for i = 1:numMCS
    g_hat = (y_act(i,:).' / s);  % Nact x 1

    % Collect horizontal phase differences (same row, adjacent columns)
    u_acc = 0;  u_cnt = 0;
    v_acc = 0;  v_cnt = 0;

    for a = 1:numel(act_idx)
        for b = 1:numel(act_idx)
            dr = row_idx(b) - row_idx(a);
            dc = col_idx(b) - col_idx(a);
            if dc == 1 && dr == 0
                dph = angle(g_hat(b)) - angle(g_hat(a));
                u_acc = u_acc + atan2(sin(dph), cos(dph));
                u_cnt = u_cnt + 1;
            end
            if dr == 1 && dc == 0
                dph = angle(g_hat(b)) - angle(g_hat(a));
                v_acc = v_acc + atan2(sin(dph), cos(dph));
                v_cnt = v_cnt + 1;
            end
        end
    end

    if u_cnt > 0
        u_hat(i) = u_acc / u_cnt;
    end
    if v_cnt > 0
        v_hat(i) = v_acc / v_cnt;
    end
end
end

% -------------------------------------------------------------------------
function H_hat = interpolate_from_layout(u_hat, v_hat, N, act_idx, y_act, K_users)
%INTERPOLATE_FROM_LAYOUT  Build full channel estimate for an arbitrary layout.

Nv = round(sqrt(N));
numMCS = numel(u_hat);
H_hat = zeros(numMCS, N);
s = sqrt(K_users);

for i = 1:numMCS
    a = steering_vector_UPA(Nv, Nv, u_hat(i), v_hat(i));
    a_act = a(act_idx);
    y_i = y_act(i,:).';
    alpha_hat = (a_act' * (y_i/s)) / (a_act' * a_act);
    H_hat(i,:) = (alpha_hat * a).';
end
end
