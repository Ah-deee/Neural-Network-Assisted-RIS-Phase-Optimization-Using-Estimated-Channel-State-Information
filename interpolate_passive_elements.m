function H_hat = interpolate_passive_elements(u_hat, v_hat, N, Nact, fc, d_elem, y_act, K_users, layout)
%INTERPOLATE_PASSIVE_ELEMENTS Interpolate full RIS channel from (u,v).
%   H_hat = INTERPOLATE_PASSIVE_ELEMENTS(u_hat, v_hat, N, Nact, fc, d_elem,
%                                         y_act, K_users, layout)
%
%   ADDED PARAMETER:
%     layout : (optional, default 'center') active element placement layout.
%              Passed directly to GET_ACTIVE_INDICES. Must match the layout
%              used in generate_channel and ls_dc_estimator for the same trial.

if nargin < 9 || isempty(layout)
    layout = 'center';
end

numMCS = numel(u_hat);
H_hat = zeros(numMCS, N);

Nv = sqrt(N); Nh = Nv;
if abs(Nv - round(Nv)) > 1e-12
    error('interpolate_passive_elements: N must correspond to a square RIS.');
end
Nv = round(Nv);
Nh = Nv;

%% Active element indices via centralised helper
act_lin_idx = get_active_indices(Nv, Nh, Nact, layout);

% Pilot magnitude
s = sqrt(K_users);

for i = 1:numMCS
    u = u_hat(i);
    v = v_hat(i);
    a = steering_vector_UPA(Nv, Nh, u, v);

    % Noisy observation at active elements
    y_i = y_act(i,:).';

    % Active portion of steering vector
    a_act = a(act_lin_idx);

    % Matched-filter gain estimation
    alpha_hat = (a_act' * y_i) / (s * (a_act' * a_act));

    % Full channel = gain * steering vector
    H_hat(i,:) = (alpha_hat * a).';
end

end
