% DC position of mask assumed to be on the edge (fftshifted)

%% GRAPPA with ACS 20

GRAPPA_mask = single(zeros(128,128));
GRAPPA_mask(:, 1:2:end) = 1;

% 2x undersampling mask
two_times_mask = GRAPPA_mask;

%ACS region 20
GRAPPA_mask(:,55:74) = 1;
GRAPPA_mask = fftshift(GRAPPA_mask);

PSF_GRAPPA = ifftshift(ifft2(GRAPPA_mask));

%% Toeplitz, radial, 280 spokes
addpath ./MT_CUDA/

Nx = 128;
Ny = 128;
Nkx = Nx;
Ntrajs = 280;

NM = Nkx * Ntrajs;

r   = linspace(-pi, pi, Nx+1); r(end)=[]; 
om1 = sin((0:Ntrajs-1)'*pi/Ntrajs)*r;
om2 = cos((0:Ntrajs-1)'*pi/Ntrajs)*r;
om  = single([om1(:), om2(:)]);

% Get NU weights of all ones
NU_impuse = complex(single(1 * ones(Ntrajs,Nkx)));

% Spread to get initial circulant weights of the toeplitz form
Circulatant_weights =  cufinufftfspread2d1(single(om(:,1)), single(om(:,2)), NU_impuse, +1,1e-6,Ny,Nx,NM, 1);
Circulatant_weights = gather(Circulatant_weights);

% Ensure the inpuse response is real
Circulatant_mask = fft2(abs(ifft2(fftshift(Circulatant_weights))));

PSF_circulant = ifftshift(ifft2(Circulatant_weights));

%% Save
save('two_times_mask.mat', 'two_times_mask')
save('GRAPPA_mask.mat', 'GRAPPA_mask')
save('Circulant_mask.mat', 'Circulatant_mask')