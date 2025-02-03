HOMEDIR = getenv('HOME');
SYSTYPE = getenv('SYSTYPE');

if strcmp(SYSTYPE,'DISCOVERY')
    BART_dir = '/project/zfan0804_715/Junzhou/bart';
else
    BART_dir = fullfile(HOMEDIR,'BART/bart-0.7.00');
end

if exist(fullfile(BART_dir,'matlab'),'dir')
    addpath(fullfile(BART_dir,'matlab'));   
    setenv('TOOLBOX_PATH', BART_dir);
    fprintf('-- BART path added.\n');
end

%% process the data

% list all first level directories in multicoil_train
dir_list = dir('multicoil_train');
dir_list = dir_list([dir_list.isdir]);
dir_list = {dir_list.name};
dir_list = dir_list(~ismember(dir_list,{'.','..'}));

for i = 1:numel(dir_list)
    process_h5_file(fullfile('multicoil_train', dir_list{i}));
end
