function process_h5_file(dirname)
    % get the parent directory
    parent_dir = dirname;

    display(['Processing file: ', dirname]);
    % find the h5 file in the RAW directory
    h5_files_list = dir(fullfile(dirname, 'RAW', '*.h5'));
    if isempty(h5_files_list)
        error('Error: no h5 file found in the RAW directory');
    end
    file_fullpath = fullfile(dirname, 'RAW', h5_files_list(1).name);
    info = h5info(file_fullpath);

    kspace = h5read(file_fullpath,'/kspace');

    % Convert to complex
    kspace_complex = complex(kspace.r, kspace.i);

    % Get the sizes
    sizes = size(kspace_complex);
    Nx = sizes(1);
    Ny = sizes(2);
    Ncoils = sizes(3);
    Nz = sizes(4);
    display(['Nx:', num2str(Nx), ' Ny:', num2str(Ny), ' Ncoils:', num2str(Ncoils), ' Nz:', num2str(Nz)]);

    % Apply the inverse 2D FFT to the kspace
    image_complex = ifftshift(ifftshift(ifft(ifft(fftshift(fftshift(kspace_complex,1),2), [],1),[],2),1),2);

    % Crop the second dimension to half in the center
    image_complex_half = image_complex(:,round(Ny/4)+1:round(Ny/4) + Ny/2,:,:);

    % check the cropped image has half the size in the second dimension
    if size(image_complex_half,2) ~= Ny/2
        error('Error: cropped image has not half the size in the second dimension');
    else
        disp('Cropped image has half the size in the second dimension');
        Ny = Ny/2;
    end


    % convert the cropped image to kspace
    kspace_half = fftshift(fftshift(fft(fft(fftshift(fftshift(image_complex_half,1),2), [],1),[],2),1),2);

    % keep the center 128x128 of the kspace
    kspace_small = kspace_half(round(Nx/2-64)+1:round(Nx/2-64)+128,round(Ny/2-64)+1:round(Ny/2-64)+128,:,:);

    % convert the kspace to image
    image_small = ifftshift(ifftshift(ifft(ifft(fftshift(fftshift(kspace_small,1),2), [],1),[],2),1),2);

    % calculate the sensitivity maps
    disp('Calculating sensitivity maps');
    SEs = calculate_SEs(kspace_small);

    % save the image and the sensitivity maps by slice
    disp('Saving the image and the sensitivity maps by slice');
    processed_dir = fullfile(parent_dir, 'processed');
    mkdir(processed_dir);
    for i = 1:size(image_small,4)
        image_slice = image_small(:,:,:,i);
        SEs_slice = SEs(:,:,:,i);
        image_slice = sum(image_slice.*conj(SEs_slice), 3);
        save(fullfile(processed_dir, ['slice_', num2str(i), '.mat']), 'image_slice', 'SEs_slice');
    end
    disp('Saving done');


end

% calculate Sensitivity Maps
function SEs = calculate_SEs(kspace_small)
    SEs = zeros(size(kspace_small));
    for i = 1:size(kspace_small,4)
        kspace_slice = kspace_small(:,:,:,i);
        kspace_slice = reshape(kspace_slice, 1, size(kspace_slice,1), size(kspace_slice,2), size(kspace_slice,3));
        [~] = evalc('[SEs_slice, ~] = bart(''ecalib -m 1 -c 0 -r 24'', kspace_slice);');
        SEs(:,:,:,i) = SEs_slice;
    end
end