function nmse = compute_nmse(H_true, H_hat)
%COMPUTE_NMSE Compute normalized mean squared error (NMSE).
%   nmse = COMPUTE_NMSE(H_true, H_hat) computes the NMSE averaged over all
%   Monte-Carlo realizations, following the definition in the paper.
%
%   NMSE = E[||H_true - H_hat||^2] / E[||H_true||^2]
%
%   where E[.] denotes the average over Monte-Carlo trials (rows).
%   Each row is one realization; columns are the N channel coefficients.

if ~isequal(size(H_true), size(H_hat))
    error('compute_nmse: Size mismatch between H_true and H_hat.');
end

diff = H_true - H_hat;
num  = mean(sum(abs(diff).^2, 2));
den  = mean(sum(abs(H_true).^2, 2));

nmse = num ./ den;

end
