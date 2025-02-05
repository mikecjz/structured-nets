load('/home/jc_350/fastMRI/multicoil_train/file_brain_AXT1_201_6002779/processed/slice_3.mat')
load('GRAPPA_mask.mat');
temp_mask = GRAPPA_mask;
temp_mask(:,2:2:end) = 0;

x_orig = image_slice_abs./max(image_slice_abs(:));
sizes = size(x_orig);
AhAx = AhA(x_orig, SEs_slice_abs, temp_mask, sizes);
Ahb = AhAx;
AhAx = reshape(AhAx, sizes);

x_solution = pcg(@(x) AhA(x, SEs_slice_abs, temp_mask, sizes), Ahb(:), [],20,[],[]);
x_solution = reshape(x_solution, sizes);

Ax2 = fft2(ifftshift(ifftshift(x_orig, 1),2)) .* temp_mask;
temp2 = fftshift(fftshift(ifft2(Ax2),1),2);
AhAx2 = temp2;


function AhAx = AhA(x, SEs, mask, sizes)
    x = reshape(x, sizes);
    Ax = fft2(ifftshift(ifftshift(x .* SEs, 1),2)) .* mask;
    temp = fftshift(fftshift(ifft2(Ax),1),2);
    AhAx = sum(temp .* SEs,3);
    AhAx = AhAx(:);
end
