load('/home/jc_350/fastMRI/multicoil_train/file_brain_AXT1_201_6002779/processed/slice_3.mat')

%% Circulant 4x
load('four_times_mask.mat');

x_orig = image_slice./max(abs(image_slice(:)));
sizes = size(x_orig);
AhAx = AhA(x_orig, SEs_slice, mask, sizes);
Ahb = AhAx;
AhAx = reshape(AhAx, sizes);
mkdir('temp/Circulant_4x');

for i = 1:20

    x_solution = pcg(@(x) AhA(x, SEs_slice, mask, sizes), Ahb(:), [],i,[],[]);
    x_solution = reshape(x_solution, sizes);
    x_solution = x_solution./max(abs(x_solution(:)));

    diff = abs(x_solution - x_orig);
    diff = diff./max(abs(x_orig(:)));
    diffx10 = diff*10;

    %save the solution as a png
    imwrite([imrotate(abs(x_solution), 90), imrotate(diff, 90), imrotate(diffx10, 90)], ['temp/Circulant_4x/solution_', num2str(i), '.png']);
end

%% Circulant 2x
load('two_times_mask.mat');

x_orig = image_slice./max(abs(image_slice(:)));
sizes = size(x_orig);
AhAx = AhA(x_orig, SEs_slice, mask, sizes);
Ahb = AhAx;
AhAx = reshape(AhAx, sizes);
mkdir('temp/Circulant_2x');

for i = 1:20
    x_solution = pcg(@(x) AhA(x, SEs_slice, mask, sizes), Ahb(:), [],i,[],[]);
    x_solution = reshape(x_solution, sizes);
    x_solution = x_solution./max(abs(x_solution(:)));

    diff = abs(x_solution - x_orig);
    diff = diff./max(abs(x_orig(:)));
    diffx10 = diff*10;

    %save the solution as a png
    imwrite([imrotate(abs(x_solution), 90), imrotate(diff, 90), imrotate(diffx10, 90)], ['temp/Circulant_2x/solution_', num2str(i), '.png']);
end 

%% Toeplitz
load('toep_mask.mat');
x_orig = image_slice./max(abs(image_slice(:)));
sizes = size(x_orig);
AhAx = AhA_toep(x_orig, SEs_slice, mask, sizes);
Ahb = AhAx;
AhAx = reshape(AhAx, sizes);
mkdir('temp/Toeplitz');
for i = 1:20
    x_solution = pcg(@(x) AhA_toep(x, SEs_slice, mask, sizes), Ahb(:), [],i,[],[]);
    x_solution = reshape(x_solution, sizes);
    x_solution = x_solution./max(abs(x_solution(:)));

    diff = abs(x_solution - x_orig);
    diff = diff./max(abs(x_orig(:)));

    diffx10 = diff*10;

    %save the solution as a png
    imwrite([imrotate(abs(x_solution), 90), imrotate(diff, 90), imrotate(diffx10, 90)], ['temp/Toeplitz/solution_', num2str(i), '.png']);
end

%% functions

function AhAx = AhA(x, SEs, mask, sizes)
    x = reshape(x, sizes);
    Ax = fft2(ifftshift(ifftshift(x .* SEs, 1),2)) .* mask;
    temp = fftshift(fftshift(ifft2(Ax),1),2);
    AhAx = sum(temp .* conj(SEs),3);
    AhAx = AhAx(:);
end


function AhAx = AhA_toep(x, SEs, mask, sizes)
    x = reshape(x, sizes);
    x = x .* SEs;
    % center pad to 2x the sizes in each dimension
    x = padarray(x, [sizes(1)/2, sizes(2)/2], 'both');
    Ax = fft2(ifftshift(ifftshift(x, 1),2)) .* mask;
    temp = fftshift(fftshift(ifft2(Ax),1),2);

    %crop to the original sizes
    temp = temp(sizes(1)/2+1:3*sizes(1)/2, sizes(2)/2+1:3*sizes(2)/2, :);

    AhAx = sum(temp .* conj(SEs),3);
    AhAx = AhAx(:);
end