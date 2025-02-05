function process_data(data_dir_root)

HOMEDIR = getenv('HOME');
SYSTYPE = getenv('SYSTYPE');

if strcmp(SYSTYPE,'DISCOVERY')
    BART_dir = '/project/zfan0804_715/Junzhou/bart';
else
    BART_dir = fullfile(HOMEDIR,'BART/bart-0.8.00');
end

if exist(fullfile(BART_dir,'matlab'),'dir')
    addpath(fullfile(BART_dir,'matlab'));   
    setenv('TOOLBOX_PATH', BART_dir);
    fprintf('-- BART path added.\n');
end

%% process the data

% list all first level directories in the provided data directory
dir_list = dir(data_dir_root);
dir_list = dir_list([dir_list.isdir]);
dir_list = {dir_list.name};
dir_list = dir_list(~ismember(dir_list,{'.','..'}));

for i = 1:numel(dir_list)
    process_h5_file(fullfile(data_dir_root, dir_list{i}));
end
end