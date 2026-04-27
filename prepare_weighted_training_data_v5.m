%% SOLUTION: Weighted Training via Adaptive Data Augmentation
%
% Since MATLAB's trainNetwork doesn't support sample weights directly,
% we use a clever workaround: REPLICATE high-SNR samples in the training set.
%
% Strategy:
% 1. Generate base dataset with SNR-weighted sampling (already done)
% 2. For samples with SNR > 25 dB, replicate them 2-3x in training set
% 3. This gives the optimizer more "votes" from high-SNR samples
% 4. Network learns to minimize error where it matters most (high SNR, large N)

function [X_train, Y_train, X_val, Y_val] = prepare_weighted_training_data(...
    X, Y, SNR_labels, N, train_ratio)
%PREPARE_WEIGHTED_TRAINING_DATA Augment high-SNR samples via replication.
%
% Replication factor:
%   SNR >= 30 dB: 4x replication
%   SNR >= 25 dB: 3x replication  
%   SNR >= 20 dB: 2x replication
%   SNR < 20 dB:  1x (no replication)
%
% For N=64, add +1 to all replication factors (emphasize large arrays)

if nargin < 5
    train_ratio = 0.85;
end

[D, ~] = size(X);

%% Compute replication factors
rep_factor = ones(D, 1);

for i = 1:D
    snr = SNR_labels(i);
    
    % Base replication by SNR (more aggressive at high SNR)
    if snr >= 35
        rep_factor(i) = 6;      % was 4
    elseif snr >= 30
        rep_factor(i) = 5;      % was 4
    elseif snr >= 25
        rep_factor(i) = 4;      % was 3
    elseif snr >= 20
        rep_factor(i) = 2;
    end
    
    % CRITICAL: Scale by N to match NMSE penalty
    % NMSE penalty scales as N, so replication should too
    if N >= 64
        rep_factor(i) = round(rep_factor(i) * 1.5);  % 50% more for N=64
    elseif N >= 36
        rep_factor(i) = round(rep_factor(i) * 1.25); % 25% more for N=36
    end
    
    % Ensure minimum replication for high SNR + large N
    if N >= 64 && snr >= 30
        rep_factor(i) = max(rep_factor(i), 8);  % At least 8x for N=64 at 30+ dB
    elseif N >= 36 && snr >= 30
        rep_factor(i) = max(rep_factor(i), 6);  % At least 6x for N=36 at 30+ dB
    end
end

%% Replicate samples
X_aug = [];
Y_aug = [];
SNR_aug = [];

for i = 1:D
    n_copies = rep_factor(i);
    X_aug = [X_aug; repmat(X(i,:), n_copies, 1)]; %#ok
    Y_aug = [Y_aug; repmat(Y(i,:), n_copies, 1)]; %#ok
    SNR_aug = [SNR_aug; repmat(SNR_labels(i), n_copies, 1)]; %#ok
end

D_aug = size(X_aug, 1);

fprintf('Data augmentation for N=%d:\n', N);
fprintf('  Original samples: %d\n', D);
fprintf('  Augmented samples: %d (%.1fx expansion)\n', D_aug, D_aug/D);

% Show replication by SNR
for snr = unique(SNR_labels(:))'
    mask_orig = abs(SNR_labels - snr) < 0.1;
    mask_aug  = abs(SNR_aug - snr) < 0.1;
    fprintf('  %2d dB: %d -> %d samples (%.1fx)\n', snr, ...
        sum(mask_orig), sum(mask_aug), sum(mask_aug)/sum(mask_orig));
end
fprintf('\n');

%% Split train/val
numTrain = round(train_ratio * D_aug);
idx = randperm(D_aug);
trainIdx = idx(1:numTrain);
valIdx = idx(numTrain+1:end);

X_train = X_aug(trainIdx, :);
Y_train = Y_aug(trainIdx, :);
X_val = X_aug(valIdx, :);
Y_val = Y_aug(valIdx, :);

end
