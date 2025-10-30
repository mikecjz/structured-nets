function crawl_data(data_dir_root, output_csv_path)
    % CRAWL_DATA Scans directories and extracts MAT file information to CSV
    %
    % Usage: crawl_data(data_dir_root, output_csv_path)
    %   data_dir_root: Root directory containing case directories
    %   output_csv_path: Path for output CSV file (optional, defaults to 'mat_files_info.csv')
    
    if nargin < 2
        output_csv_path = 'mat_files_info.csv';
    end
    
    % Initialize results table
    results = {};
    row_idx = 1;
    
    % List all first level directories in the provided data directory
    dir_list = dir(data_dir_root);
    dir_list = dir_list([dir_list.isdir]);
    dir_list = {dir_list.name};
    dir_list = dir_list(~ismember(dir_list,{'.','..'}));

    % Find all mat files
    all_mat_files_list = fullfile(data_dir_root,'**/*.mat');

    % pre allocate results array
    results = cell(length(all_mat_files_list), 6);
    
    fprintf('Found %d directories to process\n', length(dir_list));
    
    % Generate random train/val/test assignments for all cases
    % Ratio 7:2:1 (train:val:test)
    num_cases = length(dir_list);
    rng(42); % Set seed for reproducibility
    rand_indices = randperm(num_cases);
    
    train_end = round(0.7 * num_cases);
    val_end = train_end + round(0.2 * num_cases);
    
    train_val_test = cell(num_cases, 1);
    train_val_test(rand_indices(1:train_end)) = {'TRAIN'};
    train_val_test(rand_indices(train_end+1:val_end)) = {'VAL'};
    train_val_test(rand_indices(val_end+1:end)) = {'TEST'};
    
    % Process each directory
    for i = 1:numel(dir_list)
        case_dir = fullfile(data_dir_root, dir_list{i});
        case_name = dir_list{i};
        fprintf('Processing directory %d/%d: %s\n', i, length(dir_list), case_name);
        
        try
            % Extract contrast type from case name
            % Expected format: file_brain_<CONTRAST>_<numbers>_<case_id>
            contrast_type = 'UNKNOWN';
            name_parts = split(case_name, '_');
            if length(name_parts) >= 3
                contrast_type = name_parts{3};
            end
            
            % Look for processed directory
            processed_dir = fullfile(case_dir, 'processed');
            if ~exist(processed_dir, 'dir')
                fprintf('  Warning: no processed directory found for %s\n', case_name);
                % Add entry with no MAT files found
                results{row_idx, 1} = '';  % relative_path
                results{row_idx, 2} = case_name; % case_number
                results{row_idx, 3} = contrast_type; % contrast_type
                results{row_idx, 4} = NaN; % slice_number
                results{row_idx, 5} = train_val_test{i}; % TRAIN_VAL_TEST
                results{row_idx, 6} = 'No processed directory found'; % status
                row_idx = row_idx + 1;
                continue;
            end
            
            % Find all MAT files in the processed directory
            mat_files_list = dir(fullfile(processed_dir, 'slice_*.mat'));
            
            if isempty(mat_files_list)
                fprintf('  Warning: no MAT files found in processed directory for %s\n', case_name);
                % Add entry with no MAT files found
                results{row_idx, 1} = '';  % relative_path
                results{row_idx, 2} = case_name; % case_number
                results{row_idx, 3} = contrast_type; % contrast_type
                results{row_idx, 4} = NaN; % slice_number
                results{row_idx, 5} = train_val_test{i}; % TRAIN_VAL_TEST
                results{row_idx, 6} = 'No MAT files found'; % status
                row_idx = row_idx + 1;
                continue;
            end
            
            fprintf('  Found %d MAT files\n', length(mat_files_list));

            % Natural sort the mat_files_list
            mat_files_cells = struct2cell(mat_files_list);
            [~, ndx] = natsort(mat_files_cells(1,:));
            mat_files_list = mat_files_list(ndx);
            
            % Process each MAT file
            for j = 1:length(mat_files_list)
                mat_filename = mat_files_list(j).name;
                
                % Extract slice number from filename (slice_<number>.mat)
                slice_number = NaN;
                tokens = regexp(mat_filename, 'slice_(\d+)\.mat', 'tokens');
                if ~isempty(tokens)
                    slice_number = str2double(tokens{1}{1});
                end
                
                % Create relative path from data_dir_root
                relative_path = fullfile(case_name, 'processed', mat_filename);
                
                % Store results
                results{row_idx, 1} = relative_path;
                results{row_idx, 2} = case_name;
                results{row_idx, 3} = contrast_type;
                results{row_idx, 4} = slice_number;
                results{row_idx, 5} = train_val_test{i};
                results{row_idx, 6} = 'Success';
                row_idx = row_idx + 1;
            end
            
        catch ME
            fprintf('  Error processing %s: %s\n', case_name, ME.message);
            % Add entry with error information
            results{row_idx, 1} = '';
            results{row_idx, 2} = case_name;
            results{row_idx, 3} = 'UNKNOWN';
            results{row_idx, 4} = NaN;
            results{row_idx, 5} = train_val_test{i};
            results{row_idx, 6} = sprintf('Error: %s', ME.message);
            row_idx = row_idx + 1;
        end
    end
    
    % Define CSV header
    header = {'relative_path', 'case_number', 'contrast_type', 'slice_number', 'TRAIN_VAL_TEST', 'status'};
    
    % Write to CSV file
    fprintf('Writing results to: %s\n', output_csv_path);
    
    % Open file for writing
    fid = fopen(output_csv_path, 'w');
    if fid == -1
        error('Could not open file for writing: %s', output_csv_path);
    end
    
    % Write header
    fprintf(fid, '%s,%s,%s,%s,%s,%s\n', header{:});
    
    % Write data rows
    for i = 1:size(results, 1)
        if isnan(results{i, 4}) % Handle NaN slice numbers (now in column 4)
            fprintf(fid, '%s,%s,%s,,%s,%s\n', results{i,1}, results{i,2}, ...
                    results{i,3}, results{i,5}, results{i,6});
        else
            fprintf(fid, '%s,%s,%s,%d,%s,%s\n', results{i,1}, results{i,2}, ...
                    results{i,3}, results{i,4}, results{i,5}, results{i,6});
        end
    end
    
    fclose(fid);
    
    % Count successful entries
    successful_entries = strcmp(results(:,6), 'Success');
    num_successful = sum(successful_entries);
    
    fprintf('Successfully processed %d MAT files from %d directories and saved results to %s\n', ...
            num_successful, length(dir_list), output_csv_path);
    
    % Display summary statistics
    if num_successful > 0
        successful_results = results(successful_entries, :);
        unique_cases = unique(successful_results(:,2));
        unique_contrasts = unique(successful_results(:,3));
        
        fprintf('\nSummary Statistics:\n');
        fprintf('Total MAT files: %d\n', num_successful);
        fprintf('Unique cases: %d\n', length(unique_cases));
        fprintf('Contrast types found: %s\n', strjoin(unique_contrasts, ', '));
        
        % Count train/val/test distribution
        train_count = sum(strcmp(successful_results(:,5), 'TRAIN'));
        val_count = sum(strcmp(successful_results(:,5), 'VAL'));
        test_count = sum(strcmp(successful_results(:,5), 'TEST'));
        
        fprintf('Train/Val/Test distribution: %d/%d/%d\n', train_count, val_count, test_count);
        
        if ~isempty([successful_results{:,4}])
            slice_numbers = [successful_results{:,4}];
            slice_numbers = slice_numbers(~isnan(slice_numbers));
            if ~isempty(slice_numbers)
                fprintf('Slice number range: %d - %d\n', min(slice_numbers), max(slice_numbers));
            end
        end
    end
end