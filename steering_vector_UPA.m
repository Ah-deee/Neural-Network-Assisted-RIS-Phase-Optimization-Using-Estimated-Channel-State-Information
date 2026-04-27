function a = steering_vector_UPA(Nv, Nh, u, v)
%STEERING_VECTOR_UPA UPA steering vector as in Eq. (3) of the paper.
%   a = STEERING_VECTOR_UPA(Nv, Nh, u, v) returns the N-by-1 steering
%   vector for a Uniform Planar Array (UPA) with Nv vertical elements and
%   Nh horizontal elements, where N = Nv * Nh.
%
%   The spatial frequency parameters u and v are related to the elevation
%   and azimuth angles as in the paper. The array response is constructed
%   as a Kronecker product between vertical and horizontal steering
%   vectors.

% Vertical and horizontal indices (0-based)
m = (0:Nh-1); % horizontal index
n = (0:Nv-1); % vertical index

% Horizontal steering
a_h = exp(1j * u .* m);

% Vertical steering
a_v = exp(1j * v .* n).';

% Kronecker product to obtain full UPA steering vector
A = kron(a_h, a_v);

% Normalize
a = A(:) / sqrt(Nv*Nh);

end

