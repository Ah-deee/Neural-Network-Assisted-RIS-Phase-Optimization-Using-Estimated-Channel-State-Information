function [H_full, y_act, u_true, v_true] = generate_channel(numMCS, N, Nact, ...
    fc, d_elem, beta0, d0, d_k_set, alpha_PL, K_users, SNR_dB, layout)
%GENERATE_CHANNEL Generate RIS-AP channels and active-element observations.
%   [H_full, y_act, u_true, v_true] = GENERATE_CHANNEL(...) generates
%   numMCS Monte-Carlo realizations of the RIS-AP channel vector, the
%   corresponding received signal at the active RIS elements, and the true
%   spatial frequency parameters u and v.
%
%   ADDED PARAMETER:
%     layout : (optional, default 'center') active element placement layout.
%              Passed directly to GET_ACTIVE_INDICES. Options:
%              'center' | 'corners' | 'cross' | 'edges' | 'diagonal' | 'random'
%
%   All other parameters unchanged from the original version.

if nargin < 12 || isempty(layout)
    layout = 'center';
end

c0 = 3e8;
lambda = c0 / fc;

Nv = sqrt(N);
Nh = Nv;
if abs(Nv - round(Nv)) > 1e-12
    error('generate_channel: N must correspond to a square RIS (Nv*Nh).');
end
Nv = round(Nv);
Nh = Nv;

%% Active element indices via centralised helper
act_lin_idx = get_active_indices(Nv, Nh, Nact, layout);

H_full = zeros(numMCS, N);
y_act  = zeros(numMCS, Nact);
u_true = zeros(numMCS, 1);
v_true = zeros(numMCS, 1);

SNR_lin = 10.^(SNR_dB/10);

for i = 1:numMCS
    % Draw user distance and path loss
    dk = d_k_set(randi(numel(d_k_set)));
    beta_k = beta0 * (dk/d0)^(-alpha_PL);

    % Angles (AoA) uniformly in [-pi/2, pi/2]
    theta = -pi/2 + pi*rand();   % elevation
    phi   = -pi/2 + pi*rand();   % azimuth

    % Spatial frequency parameters
    k0 = 2*pi/lambda;
    u = k0 * d_elem * cos(theta) * cos(phi);
    v = k0 * d_elem * cos(theta) * sin(phi);

    u_true(i) = u;
    v_true(i) = v;

    % RIS-AP channel vector (LoS dominated, single path)
    a = steering_vector_UPA(Nv, Nh, u, v);
    g = sqrt(beta_k) * a;   % N x 1

    H_full(i,:) = g.';

    % Active element subset
    g_act = g(act_lin_idx);

    % Received pilot observation with AWGN
    s = sqrt(K_users);
    noise_var = (norm(g_act)^2 * s^2) / SNR_lin;
    n = sqrt(noise_var/2) * (randn(size(g_act)) + 1j*randn(size(g_act)));

    y_act(i,:) = (g_act * s + n).';
end

end
