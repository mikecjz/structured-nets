% DC position of mask assumed to be on the edge (fftshifted)

%% GRAPPA with ACS 20

mask = generate_GRAPPA_mask(128, 128);
PSF_GRAPPA = ifftshift(ifft2(mask));
save('GRAPPA_mask.mat', 'mask');


%% 2x undersampling mask
mask = generate_two_times_mask(128, 128);
PSF_two_times = ifftshift(ifft2(mask));
save('two_times_mask.mat', 'mask');

mask_illustration = zeros(128,128);
for i = 1:16
    start_col = (i-1)*8 + 1;  % Space lines evenly across the image (128/16 = 8)
    mask_illustration(:, start_col:start_col+1) = 1;  % Make each line 2 pixels wide
end

figure;imshow(mask_illustration);

%save the figure
imwrite(mask_illustration, '2x_mask.png');


%% 4x undersampling mask
mask = generate_four_times_mask(128, 128);
PSF_four_times = ifftshift(ifft2(mask));
save('four_times_mask.mat', 'mask');

mask_illustration = zeros(128,128);
for i = 1:8
    start_col = (i-1)*16 + 1;  % Space lines evenly across the image (128/16 = 8)
    mask_illustration(:, start_col:start_col+1) = 1;  % Make each line 2 pixels wide
end

figure;imshow(mask_illustration);

%save the figure
imwrite(mask_illustration, '4x_mask.png');

%% Toeplitz, radial, 280 spokes
addpath ./MT_CUDA/

mask = generate_Circulant_mask(128, 128, 280);
PSF_Circulant = ifftshift(ifft2(mask));
save('toep_mask.mat', 'mask');


%% Functions

function mask = generate_two_times_mask(Nx, Ny)
    mask = zeros(Nx, Ny,'single');
    mask(:, 1:2:end) = 1;
end

function mask = generate_four_times_mask(Nx, Ny)
    mask = zeros(Nx, Ny,'single');
    mask(:, 1:4:end) = 1;
end

function mask = generate_GRAPPA_mask(Nx, Ny)
    mask = zeros(Nx, Ny,'single');
    mask(:, 1:2:end) = 1;

    %ACS region 20
    mask(:,55:74) = 1;
    mask = fftshift(mask);
end

function mask = generate_Circulant_mask(Nx, Ny, Ntrajs)
    Nkx = Nx;

    NM = Nkx * Ntrajs;

    r   = linspace(-pi, pi, Nx+1); r(end)=[]; 
    om1 = sin((0:Ntrajs-1)'*pi/Ntrajs)*r;
    om2 = cos((0:Ntrajs-1)'*pi/Ntrajs)*r;
    om  = single([om1(:), om2(:)]);

    % Get NU weights of all ones
    NU_impuse = complex(single(1 * ones(Ntrajs,Nkx)));

    % Plot with black background, white markers, no axis
    figure('Color', 'k');  % black background
    plot(om1(1:10:end), om2(1:10:end), 'w.'); % white dots
    axis off;  % remove axes and ticks
    axis square;  % make plot square
    
    % Save the figure as PNG with black background
    exportgraphics(gcf, 'radial_trajectory.png', 'BackgroundColor', 'k');

    % Spread to get initial circulant weights of the toeplitz form
    Circulatant_weights =  cufinufftfspread2d1(single(om(:,1)), single(om(:,2)), NU_impuse, +1,1e-6,Ny,Nx,NM, 1);
    Circulatant_weights = gather(Circulatant_weights);

    % Ensure the inpuse response is real
    mask = fft2(abs(ifft2(fftshift(Circulatant_weights))));
end