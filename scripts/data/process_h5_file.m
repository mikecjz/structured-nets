function process_h5_file(case_dir)
    % get the parent directory
    parent_dir = case_dir;

    display(['Processing file: ', case_dir]);
    % find the h5 file in the RAW directory
    h5_files_list = dir(fullfile(case_dir, 'RAW', '*.h5'));
    if isempty(h5_files_list)
        error('Error: no h5 file found in the RAW directory');
    end
    file_fullpath = fullfile(case_dir, 'RAW', h5_files_list(1).name);
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

    % Get the sensitivity maps in all real format
    SEs_abs = abs(SEs);
    sos_SEs_abs = sum(SEs_abs.^2,3); % sum over coil dimension (dim 3)
    SEs_abs = SEs_abs./sos_SEs_abs; % ensures SEs_abs has 1 after sum of squares

    % save the image and the sensitivity maps by slice
    disp('Saving the image and the sensitivity maps by slice');
    processed_dir = fullfile(parent_dir, 'processed');
    mkdir(processed_dir);
    mkdir(fullfile(processed_dir, 'images'));
    for i = 1:size(image_small,4)
        image_slice = single(image_small(:,:,:,i));
        SEs_slice = single(SEs(:,:,:,i));

        save_coil_images(image_slice, SEs_slice, i, fullfile(processed_dir, 'images'));

        image_slice = single(sum(image_slice.*conj(SEs_slice), 3));

        SEs_slice_abs = single(SEs_abs(:,:,:,i));
        image_slice_abs = single(abs(image_slice));
        save(fullfile(processed_dir, ['slice_', num2str(i), '.mat']), 'image_slice', 'SEs_slice', 'image_slice_abs', 'SEs_slice_abs');
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

function save_coil_images(image_small_slice, SEs_slice, slice_idx, processed_dir)
    % create tiled image from image_small_slice
    image_tile = imtile(abs(image_small_slice)./max(abs(image_small_slice(:))));
    % save the tiled image
    imwrite(image_tile, fullfile(processed_dir, ['slice_', num2str(slice_idx), '_coil_images.png']));

    % create tiled image from SEs_slice
    SEs_tile = imtile(abs(SEs_slice)./max(abs(SEs_slice(:))));
    % save the tiled image
    imwrite(SEs_tile, fullfile(processed_dir, ['slice_', num2str(slice_idx), '_coil_SEs.png']));
end