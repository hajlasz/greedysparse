function find_individual_matrices(init_cond)
    %% Prints out a list of matrices where greedy viewed fewer entries than Lanczos at 1e-5 tolerance, with nontrivial number of iterations

    %% Configuration
    MATRIX_DIR   = '/home/nh2429/matrices_under_2G';
    TARGET_LEVEL = 5;       % tolerance 10^{-5}
    MIN_ITERS    = 10;

    %% Load results and find candidates where greedy < lanczos at 1e-5
    G = load('/scratch/nh2429/greedysparse/suitesparse/plotting/greedy_unsparsified.mat',  'M_greedy');   M_g = G.M_greedy;
    L = load('/scratch/nh2429/greedysparse/suitesparse/plotting/lanczos_unsparsified.mat', 'M_lanczos');  M_l = L.M_lanczos;

    g_files   = M_g.filenames;
    g_viewed  = M_g.(sprintf('greedy_viewed_%s', init_cond));    % 10 × n_greedy

    l_files   = M_l.filenames;
    l_viewed  = M_l.(sprintf('lanczos_viewed_%s', init_cond));   % 10 × n_lanczos

    % Build lookup: filename -> lanczos column index
    l_map = containers.Map(double(l_files), 1:length(l_files));

    cand_files = [];
    cand_gv    = [];
    cand_lv    = [];

    for j = 1:length(g_files)
        fn = double(g_files(j));
        if ~l_map.isKey(fn), continue; end
        lj = l_map(fn);

        gv = g_viewed(TARGET_LEVEL, j);
        lv = l_viewed(TARGET_LEVEL, lj);
        if isnan(gv) || isnan(lv), continue; end
        if gv >= lv, continue; end

        cand_files(end+1) = fn;         
        cand_gv(end+1)    = gv;         
        cand_lv(end+1)    = lv;         
    end

    % Sort by ratio (biggest greedy advantage first)
    ratio = cand_lv ./ cand_gv;
    [~, ord] = sort(ratio, 'descend');
    cand_files = cand_files(ord);
    cand_gv    = cand_gv(ord);
    cand_lv    = cand_lv(ord);

    fprintf('Found %d candidates where greedy < lanczos at 1e-%d.\n', ...
        length(cand_files), TARGET_LEVEL);
    fprintf('Re-running greedy on each to count iterations...\n\n');

    %% Re-run greedy on each candidate, count iterations to 1e-5
    warning('off', 'MATLAB:eigs:NotConverged');

    res_files  = [];
    res_gv     = [];
    res_lv     = [];
    res_ratio  = [];
    res_iters  = [];

    for ci = 1:length(cand_files)
        fn = cand_files(ci);
        matpath = fullfile(MATRIX_DIR, sprintf('%d.mat', fn));
        fprintf('  [%d/%d] matrix %d ... ', ci, length(cand_files), fn);

        try
            data = load(matpath);
            A = sparse(data.A);
        catch
            fprintf('file not found, skipping\n');
            continue
        end

        ni = count_greedy_iters(A, init_cond, TARGET_LEVEL);

        if isnan(ni)
            fprintf('did not converge\n');
            continue
        end
        fprintf('converged in %d iterations\n', ni);

        if ni < MIN_ITERS, continue; end

        res_files(end+1) = fn;             
        res_gv(end+1)    = cand_gv(ci);     
        res_lv(end+1)    = cand_lv(ci);     
        res_ratio(end+1) = cand_lv(ci) / cand_gv(ci);  
        res_iters(end+1) = ni;             
    end

    %% Report
    [~, ord] = sort(res_ratio, 'descend');
    res_files = res_files(ord);
    res_gv    = res_gv(ord);
    res_lv    = res_lv(ord);
    res_ratio = res_ratio(ord);
    res_iters = res_iters(ord);

    fprintf('\n%s\n', repmat('=', 1, 80));
    fprintf('greedy << lanczos at 1e-%d  AND  greedy needed >= %d iterations\n', ...
        TARGET_LEVEL, MIN_ITERS);
    fprintf('%s\n\n', repmat('=', 1, 80));
    fprintf('%-10s %-16s %-16s %-10s %-8s\n', ...
        'Filename', 'Greedy viewed', 'Lanczos viewed', 'Ratio', 'Iters');
    fprintf('%s\n', repmat('-', 1, 80));

    for i = 1:length(res_files)
        fprintf('%-10d %-16.6e %-16.6e %-10.2f %-8d\n', ...
            res_files(i), res_gv(i), res_lv(i), res_ratio(i), res_iters(i));
    end

    fprintf('\nTotal: %d matrices found\n', length(res_files));
end


function ni = count_greedy_iters(A, init_cond, target_level)
% COUNT_GREEDY_ITERS  Run the greedy method (same loop as run_greedy.m)
%   and return the number of while-loop iterations to reach
%   relerr < 10^{-target_level}.  Returns NaN if it fails.

    n = size(A, 1);

    % Reference eigenvalue
    try
        [~, eigval] = eigs(A, 1, 'largestabs', 'SubspaceDimension', min(n, 100));
    catch
        ni = NaN; return
    end
    if isnan(eigval), ni = NaN; return; end

    % Initial index
    switch init_cond
        case 'row'
            [~, ind] = max(sum(A.^2, 1));
        case 'diag'
            [~, ind] = max(abs(diag(A)));
    end

    S           = [ind];
    Smask       = false(1, n);
    found_level = 0;
    ni          = 0;

    while found_level < 10
        ni = ni + 1;
        Asub = A(S, S);

        [~, eigval_greedy] = eigs(Asub, 1, 'largestabs');
        if isnan(eigval_greedy), ni = NaN; return; end

        relerr = abs((abs(eigval_greedy) - abs(eigval)) / abs(eigval));

        for k = 1:10
            if relerr < 10^(-k) && found_level < k
                found_level = k;
            end
        end

        % Early exit once we've reached the target
        if found_level >= target_level
            return
        end

        % Expand
        r = A(:, S) * randn(length(S), 1);
        Snew = unique([S, find(r ~= 0)']);

        % Stalling check
        if isempty(Snew) || (length(S) == length(Snew) && isequal(S, Snew))
            Smask(S) = true;
            Sc = find(~Smask);
            if isempty(Sc), ni = NaN; return; end

            switch init_cond
                case 'row'
                    [~, ind] = max(sum(A(Sc, Sc).^2, 1));
                case 'diag'
                    [~, ind] = max(abs(diag(A(Sc, Sc))));
            end
            Snew = unique([Snew, Sc(ind)]);
        end

        S = Snew;
    end
end