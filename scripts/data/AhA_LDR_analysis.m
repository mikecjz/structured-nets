close all hidden
load('/home/jc_350/fastMRI/multicoil_train/file_brain_AXT1_201_6002779/processed/slice_3.mat')
load('four_times_mask.mat');

%% Extract 1D

sizes = size(SEs_slice);
Nx = sizes(1);
Ny = sizes(2);
Ncoils = sizes(3);

%Extract a single line from SEs_slice
SEs_slice_1D = squeeze(SEs_slice(64,:,:));

%Extract a single line from mask
mask_1D = squeeze(mask(64,:));

%% Make System Matrix

F = dftmtx(Ny);

Fh = conj(F);

AhA = zeros(Ny, Ny);

for i = 1:Ncoils

    AhA = AhA + (diag(conj(SEs_slice_1D(:,i))) * Fh * diag(mask_1D) * F * diag(SEs_slice_1D(:,i))) /Ny;

end

% plot the operator, square
figure;
imagesc(abs(AhA));
colormap turbo;
axis square off;  % Make plot square and remove axes
colorbar;         % Add colorbar

% save figure as png using gcf
saveas(gcf, 'AhA.png');

%% Make System Matrix for Circulant

 

AhA_circulant = Fh * diag(mask_1D) * F /Ny;

% plot the operator, square
figure;
imagesc(abs(AhA_circulant));
colormap turbo;
axis square off;  % Make plot square and remove axes

% save figure as png using gcf
saveas(gcf, 'AhA_circulant.png');

%% Displacement

%lower shift matrix
c = zeros(Ny,1);
c(2) = 1;

r = zeros(Ny,1);
Z = toeplitz(c,r);

R = AhA - Z * AhA * Z';

% plot the residual, square
figure;
imagesc(abs(R));
colormap turbo;
axis square off;  % Make plot square and remove axes

% save figure as png using gcf
saveas(gcf, 'AhA_displacement.png');

[U,S,V] = svd(R,'econ');

% plot the abs of the singular values
figure;
set(gcf, 'Position', [100 100 800 200]); % Sets window to 300x600 pixels (1:2 ratio)
plot(1:15, abs(diag(S(1:15,1:15))), 'LineWidth', 2, 'Color', 'k', 'LineStyle', '-', 'Marker', 'o', 'MarkerSize', 10);
xlim([0,15])
xticks(0:15);
xlabel('Index');
ylabel('Singular Value Magnitude');

% save figure as png using gcf
exportgraphics(gcf, 'AhA_singular_values.eps')





