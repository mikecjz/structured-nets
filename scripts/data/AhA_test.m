load('/home/jc_350/fastMRI/multicoil_train/file_brain_AXT1_201_6002779/processed/slice_3.mat')
load('GRAPPA_mask.mat');
temp_mask = zeros(128,128);
temp_mask(:,2:4:end) = 1;

x_orig = image_slice./max(abs(image_slice(:)));
sizes = size(x_orig);
AhAx = AhA(x_orig, SEs_slice, temp_mask, sizes);
Ahb = AhAx;
AhAx = reshape(AhAx, sizes);

x_solution = pcg(@(x) AhA(x, SEs_slice, temp_mask, sizes), Ahb(:), [],25,[],[]);
x_solution = reshape(x_solution, sizes);

Ax2 = fft2(ifftshift(ifftshift(x_orig, 1),2)) .* temp_mask;
temp2 = fftshift(fftshift(ifft2(Ax2),1),2);
AhAx2 = temp2;


function AhAx = AhA(x, SEs, mask, sizes)
    x = reshape(x, sizes);
    Ax = fft2(ifftshift(ifftshift(x .* SEs, 1),2)) .* mask;
    temp = fftshift(fftshift(ifft2(Ax),1),2);
    AhAx = sum(temp .* conj(SEs),3);
    AhAx = AhAx(:);
end
