% DC position of mask assumed to be on the edge (fftshifted)

%% GRAPPA with ACS 20

mask = generate_GRAPPA_mask(128, 128);
PSF_GRAPPA = ifftshift(ifft2(mask));
save('GRAPPA_mask.mat', 'mask');


%% 2x undersampling mask
mask = generate_two_times_mask(256, 256);
PSF_two_times = ifftshift(ifft2(mask));
save('two_times_mask_256.mat', 'mask');

mask_illustration = zeros(128,128);
for i = 1:16
    start_col = (i-1)*8 + 1;  % Space lines evenly across the image (128/16 = 8)
    mask_illustration(:, start_col:start_col+1) = 1;  % Make each line 2 pixels wide
end

figure;imshow(mask_illustration);

%save the figure
imwrite(mask_illustration, '2x_mask.png');


%% 4x undersampling mask
mask = generate_four_times_mask(256, 256);
PSF_four_times = ifftshift(ifft2(mask));
save('four_times_mask_256.mat', 'mask');

mask_illustration = zeros(128,128);
for i = 1:8
    start_col = (i-1)*16 + 1;  % Space lines evenly across the image (128/16 = 8)
    mask_illustration(:, start_col:start_col+1) = 1;  % Make each line 2 pixels wide
end

figure;imshow(mask_illustration);

%save the figure
imwrite(mask_illustration, '4x_mask.png');

%% 8x undersampling mask
mask = generate_eight_times_mask(128, 128);
PSF_four_times = ifftshift(ifft2(mask));
save('eight_times_mask.mat', 'mask');

mask_illustration = zeros(128,128);
for i = 1:4
    start_col = (i-1)*32 + 1;  % Space lines evenly across the image (128/32 = 4)
    mask_illustration(:, start_col:start_col+1) = 1;  % Make each line 2 pixels wide
end

figure;imshow(mask_illustration,[]);

%save the figure
imwrite(mask_illustration, '8x_mask.png');

%% Toeplitz, radial, 280 spokes
addpath ./MT_CUDA/

mask = generate_radia_mask(128, 128, 280);
PSF_Circulant = ifftshift(ifft2(mask));
save('toep_mask.mat', 'mask');

%% Spiral mask

load('/home/jc_350/Toolboxes/gridding/spiralexampledata.mat')

spirals = 1:9:18;

locs = [];

for i = spirals

    locs = [locs, (1:744) + ((i-1)* 744)];
end

kx = real(kspacelocations(locs))*10;
ky = imag(kspacelocations(locs))*10;

mask = generate_toep_mask(128, 128, kx, ky, 'spiral');
save('spiral_mask.mat', 'mask');

%% Random

[kx, ky] = random_curves();
mask = generate_toep_mask(128, 128, kx, ky, 'random');
save('random_mask.mat', 'mask');


%% Functions

function mask = generate_two_times_mask(Nx, Ny)
    mask = zeros(Nx, Ny,'single');
    mask(:, 1:2:end) = 1;
end

function mask = generate_four_times_mask(Nx, Ny)
    mask = zeros(Nx, Ny,'single');
    mask(:, 1:4:end) = 1;
end

function mask = generate_eight_times_mask(Nx, Ny)
    mask = zeros(Nx, Ny,'single');
    mask(:, 1:8:end) = 1;
end

function mask = generate_GRAPPA_mask(Nx, Ny)
    mask = zeros(Nx, Ny,'single');
    mask(:, 1:2:end) = 1;

    %ACS region 20
    mask(:,55:74) = 1;
    mask = fftshift(mask);
end

function mask = generate_radia_mask(Nx, Ny, Ntrajs)
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

function mask = generate_toep_mask(Nx, Ny, kx, ky, name)
    
    NM = numel(kx);

    om  = single([kx(:), ky(:)]);

    % Get NU weights of all ones
    NU_impuse = complex(single(1 * ones(NM,1)));

    % Plot with black background, white markers, no axis
    figure('Color', 'k');  % black background
    plot(kx(:) ,ky(:), 'w.'); % white dots
    axis off;  % remove axes and ticks
    axis square;  % make plot square
    
    % Save the figure as PNG with black background
    exportgraphics(gcf, [name,'.png'], 'BackgroundColor', 'k');

    % Spread to get initial circulant weights of the toeplitz form
    Circulatant_weights =  cufinufftfspread2d1(single(om(:,1)), single(om(:,2)), NU_impuse, +1,1e-6,Ny,Nx,NM, 1);
    Circulatant_weights = gather(Circulatant_weights);

    
    mask = fft2((ifft2(fftshift(Circulatant_weights))));
end

function [x,y] = random_curves()
%% random_curves.m
% Draw a bunch of random smooth curves inside [-1,1] x [-1,1]

rng('shuffle');  % different curves each run

num_curves    = 30;   % how many separate curves to draw
num_ctrl_pts  = 10;    % how wiggly each curve is (more pts = more wiggles)
num_samples   = 400;  % how smooth each curve looks
spread        = 0.4;   % typical radius (smaller -> fewer boundary hits)


figure; hold on;

x = [];
y = [];

for k = 1:num_curves

    % Parameter along the curve
    t_ctrl = sort(rand(1, num_ctrl_pts));

    % Random control points, centered near 0, mostly inside [-1,1]
    x_ctrl = spread * randn(1, num_ctrl_pts);
    y_ctrl = spread * randn(1, num_ctrl_pts);

    % Clip to [-1,1] just in case
    x_ctrl = max(min(x_ctrl, 1), -1);
    y_ctrl = max(min(y_ctrl, 1), -1);

    % Smooth spline
    t_fine = linspace(0, 1, num_samples);
    x_smooth = interp1(t_ctrl, x_ctrl, t_fine, 'spline');
    y_smooth = interp1(t_ctrl, y_ctrl, t_fine, 'spline');

    % Keep inside box

    oob = (x_smooth>=1 | x_smooth<=-1 | y_smooth>=1 | y_smooth<=-1);
    x_smooth(oob) = [];
    y_smooth(oob) = [];

    plot(x_smooth, y_smooth, 'LineWidth', 1.5);
    x = [x, x_smooth* pi];
    y = [y, y_smooth *pi];

end

% --- 6. Make it look nice
axis equal;
xlim([-1 1]);
ylim([-1 1]);
box on;
xlabel('x');
ylabel('y');
title('Random curves inside [-1,1]');

hold off;
end