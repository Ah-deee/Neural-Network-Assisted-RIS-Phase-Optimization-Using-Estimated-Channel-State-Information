function [uv, u_coarse, v_coarse] = solve_phase_ls(A, phi_vec_wrapped, mag_weight)
%SOLVE_PHASE_LS  Two-stage robust weighted LS for the phase-difference system.
%  FIXED VERSION - corrects Stage-1 fallback for large-baseline layouts.

w = mag_weight(:);
w = w / (sum(w) + 1e-30);

%% ---- Stage 1: Coarse unambiguous estimate ----
unit_mask = (abs(A(:,1)) <= 1) & (abs(A(:,2)) <= 1) & (any(A ~= 0, 2));

if any(unit_mask)
    A_u   = A(unit_mask, :);
    phi_u = phi_vec_wrapped(unit_mask);
    w_u   = w(unit_mask);
    w_u   = w_u / (sum(w_u) + 1e-30);
    AtW_u = A_u' * diag(w_u);
    lhs_u = AtW_u * A_u;
    rhs_u = AtW_u * phi_u;
    lam_u = max(max(diag(lhs_u)) * 1e-6, 1e-10);
    uv_coarse = (lhs_u + lam_u*eye(2)) \ rhs_u;

else
    % Fallback: no unit-baseline pairs (corners, edges, diagonal).
    %
    % BUG IN PREVIOUS VERSION: scored candidates by (A*uv_LS - phi_unwrapped).
    % Since uv_LS is solved FROM phi_unwrapped, this residual is ~0 for ALL
    % branches -- the loop cannot distinguish correct from incorrect branch.
    %
    % CORRECT scoring: re-wrap A*uv_candidate to [-pi,pi] and compare
    % directly against the original phi_vec_wrapped. The correct branch
    % produces near-zero re-wrap residual; wrong branches do not.

    u_candidates = 0;
    h_pairs = find(A(:,2) == 0 & A(:,1) ~= 0);
    if ~isempty(h_pairs)
        [~, ib]  = min(abs(A(h_pairs, 1)));
        ip       = h_pairs(ib);
        base_u   = A(ip, 1);
        phi0_u   = phi_vec_wrapped(ip);
        k_max    = ceil(abs(base_u) / 2);
        k_try    = (-k_max : k_max)';
        u_cand   = (phi0_u + 2*pi*k_try) / base_u;
        u_candidates = u_cand(abs(u_cand) <= pi + 1e-9);
        if isempty(u_candidates); u_candidates = 0; end
    end

    v_candidates = 0;
    v_pairs = find(A(:,1) == 0 & A(:,2) ~= 0);
    if ~isempty(v_pairs)
        [~, ib]  = min(abs(A(v_pairs, 2)));
        ip       = v_pairs(ib);
        base_v   = A(ip, 2);
        phi0_v   = phi_vec_wrapped(ip);
        k_max    = ceil(abs(base_v) / 2);
        k_try    = (-k_max : k_max)';
        v_cand   = (phi0_v + 2*pi*k_try) / base_v;
        v_candidates = v_cand(abs(v_cand) <= pi + 1e-9);
        if isempty(v_candidates); v_candidates = 0; end
    end

    % Diagonal-only fallback
    if isempty(h_pairs) && isempty(v_pairs)
        [~, ib] = min(sum(abs(A), 2));
        ip      = ib(1);
        base_u  = A(ip, 1);  base_v = A(ip, 2);
        phi0    = phi_vec_wrapped(ip);
        k_max   = ceil((abs(base_u) + abs(base_v)) / 2);
        k_try   = (-k_max : k_max)';
        u_sweep = linspace(-pi, pi, 37)';
        best_res  = inf;  uv_coarse = [0; 0];
        for ks = 1:numel(k_try)
            phi_unw_ks = phi0 + 2*pi*k_try(ks);
            for us = 1:numel(u_sweep)
                u_try = u_sweep(us);
                v_try = (phi_unw_ks - base_u*u_try) / base_v;
                if abs(v_try) > pi + 1e-9; continue; end
                uv_try       = [u_try; v_try];
                phi_rewrap   = atan2(sin(A*uv_try), cos(A*uv_try));
                res          = norm(w .* (phi_rewrap - phi_vec_wrapped));
                if res < best_res; best_res = res; uv_coarse = uv_try; end
            end
        end
        phi_pred = A * uv_coarse;
        k_vec    = round((phi_pred - phi_vec_wrapped) / (2*pi));
        phi_unw  = phi_vec_wrapped + 2*pi*k_vec;
        AtW = A' * diag(w); lhs = AtW*A; rhs = AtW*phi_unw;
        lam = max(max(diag(lhs))*1e-6, 1e-10);
        uv  = (lhs + lam*eye(2)) \ rhs;
        u_coarse = uv_coarse(1); v_coarse = uv_coarse(2);
        return;
    end

    % Score each (u,v) candidate pair using re-wrap residual
    best_res  = inf;
    uv_coarse = [0; 0];
    for ku = 1:numel(u_candidates)
        for kv = 1:numel(v_candidates)
            uv_try     = [u_candidates(ku); v_candidates(kv)];
            phi_rewrap = atan2(sin(A*uv_try), cos(A*uv_try));
            res        = norm(w .* (phi_rewrap - phi_vec_wrapped));
            if res < best_res
                best_res  = res;
                uv_coarse = uv_try;
            end
        end
    end
end

u_coarse = uv_coarse(1);
v_coarse = uv_coarse(2);

%% ---- Stage 2: Unwrap all pairs using coarse estimate, then refine ----
phi_pred      = A * uv_coarse;
k_vec         = round((phi_pred - phi_vec_wrapped) / (2*pi));
phi_unwrapped = phi_vec_wrapped + 2*pi * k_vec;

AtW  = A' * diag(w);
lhs  = AtW * A;
rhs  = AtW * phi_unwrapped;
lam  = max(max(diag(lhs)) * 1e-6, 1e-10);
uv   = (lhs + lam*eye(2)) \ rhs;

end
