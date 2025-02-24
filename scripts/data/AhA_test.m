load('/home/jc_350/fastMRI/multicoil_train/file_brain_AXT1_201_6002779/processed/slice_3.mat')
load('four_times_mask.mat');

%% Circulant
x_orig = image_slice./max(abs(image_slice(:)));
sizes = size(x_orig);
AhAx = AhA(x_orig, SEs_slice, mask, sizes);
Ahb = AhAx;
AhAx = reshape(AhAx, sizes);

x_solution = pcg(@(x) AhA(x, SEs_slice, mask, sizes), Ahb(:), [],20,[],[]);
x_solution = reshape(x_solution, sizes);


%% Toeplitz
load('toep_mask.mat');
x_orig = image_slice./max(abs(image_slice(:)));
sizes = size(x_orig);
AhAx = AhA_toep(x_orig, SEs_slice, mask, sizes);
Ahb = AhAx;
AhAx = reshape(AhAx, sizes);

x_solution = pcg(@(x) AhA_toep(x, SEs_slice, mask, sizes), Ahb(:), [],25,[],[]);
x_solution = reshape(x_solution, sizes);

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