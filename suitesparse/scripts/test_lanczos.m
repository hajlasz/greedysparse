function test_lanczos(ind_start, ind_end)
    addpath '/home/nh2429/suite_sparse_api'
    index = ssget; 
    
    % get the indices of the real symmetric matrices
    f = find(index.numerical_symmetry == 1 & index.isReal);
    
    % sort the indices in f based on number of rows, low to high
    [~, j] = sort(index.nrows(f));
    f = f(j);

    % getting our matrix names from the ones downloaded
    M_names = dir('/home/nh2429/matrices_under_2G');
    M_names = {M_names.name};
    M_names = strrep(M_names, '.mat', '');
    [M_names] = cellfun(@str2num, M_names(3:end));
    M_names = sort(M_names);
    M = struct();

    f = f(M_names); % these are the matrices I actually have downloaded

    filename = ['../results/lanczos/lanczos_results_' num2str(ind_start) '_to_' num2str(ind_end) '.mat'];

    M.filenames = M_names; % names of files are i.mat in my folder
    f =  f(M.filenames >= ind_start);
    M.filenames = M.filenames((M.filenames >= ind_start));
    f = f(M.filenames <= ind_end);
    M.filenames = M.filenames((M.filenames <= ind_end));
    M.inds = f; % indices of matrices in ssget

    M_names_restricted = M.filenames;

    % % row norm init
    % M = run_lanczos(M, "row", M_names_restricted);

    % diagonal init
    M = run_lanczos(M, "diag", M_names_restricted);
    
    % saving the struct to file
    save(filename, 'M');
end
    
