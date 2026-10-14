function combine_results()
    
    %%% Combine the Lanczos results
    mat_files = dir('C:\Users\nhajl\Desktop\greedysparse\suitesparse\results\lanczos\*.mat');
    addpath('C:\Users\nhajl\Desktop\greedysparse\suitesparse\results\lanczos')
    combined_filename = 'lanczos_unsparsified.mat';
    combined = struct();
    for i = 1:length(mat_files)
        load(mat_files(i).name);  % loads struct M
        info = fields(M);

        for j = 1:length(info)
            f = info{j};  % field name

            if ~isfield(combined, f)
                combined.(f) = M.(f);  % initialize
            else
                % Check row compatibility
                if size(combined.(f), 1) == size(M.(f), 1)
                    combined.(f) = [combined.(f), M.(f)];  % horizontal concat
                elseif size(combined.(f), 2) == size(M.(f), 1)
                    combined.(f) = [combined.(f); M.(f)];  % vertical concat
                else
                    warning('Size mismatch in field %s from file %s. Skipping.', ...
                        f, mat_files(i).name);
                end
            end
        end
    end

    M_lanczos = combined;
    save(combined_filename, 'M_lanczos');
    
    
    
    %%% Combine the Greedy results
    mat_files = dir('C:\Users\nhajl\Desktop\greedysparse\suitesparse\results\greedy\*.mat');
    addpath('C:\Users\nhajl\Desktop\greedysparse\suitesparse\results\greedy')
    combined_filename = 'greedy_unsparsified.mat';
    combined = struct();
    for i = 1:length(mat_files)
        load(mat_files(i).name);  % loads struct M
        info = fields(M);
    
        for j = 1:length(info)
            f = info{j};  % field name
    
            if ~isfield(combined, f)
                combined.(f) = M.(f);  % initialize
            else
                % Check row compatibility
                if size(combined.(f), 1) == size(M.(f), 1)
                    combined.(f) = [combined.(f), M.(f)];  % horizontal concat
                elseif size(combined.(f), 2) == size(M.(f), 1)
                    combined.(f) = [combined.(f); M.(f)];  % vertical concat
                else
                    warning('Size mismatch in field %s from file %s. Skipping.', ...
                        f, mat_files(i).name);
                end
            end
        end
    end
    
    M_greedy = combined;
    save(combined_filename, 'M_greedy');
    
    
    
    
    %%% Combine the Power results
    mat_files = dir('C:\Users\nhajl\Desktop\greedysparse\suitesparse\results\power\*.mat');
    addpath('C:\Users\nhajl\Desktop\greedysparse\suitesparse\results\power')
    combined_filename = 'power_unsparsified.mat';
    combined = struct();
    for i = 1:length(mat_files)
        load(mat_files(i).name);  % loads struct M
        info = fields(M);

        for j = 1:length(info)
            f = info{j};  % field name

            if ~isfield(combined, f)
                combined.(f) = M.(f);  % initialize
            else
                % Check row compatibility
                if size(combined.(f), 1) == size(M.(f), 1)
                    combined.(f) = [combined.(f), M.(f)];  % horizontal concat
                elseif size(combined.(f), 2) == size(M.(f), 1)
                    combined.(f) = [combined.(f); M.(f)];  % vertical concat
                else
                    warning('Size mismatch in field %s from file %s. Skipping.', ...
                        f, mat_files(i).name);
                end
            end
        end
    end

    M_power = combined;
    save(combined_filename, 'M_power');
end