function M = run_power(M, init_cond, M_names)
    warning('off', 'MATLAB:eigs:NotConverged');

    % viewed_neg(k, :) stores the num entries read to reach 10^-k relerr

    % viewed_neg is no antistalling: NaN if not reached, otherwise fraction of A viewed
    viewed_neg = NaN(10, length(M_names)); 

    % viewed_neg_antistallA is: if stalled, set entries read to 100% of all entries of A 
    viewed_neg_antistallA = NaN(10, length(M_names));
    stalled = false(1, length(M_names));
    
    c = 1;
    for i = M_names
        i
        try
            data = load(strcat('/home/nh2429/matrices_under_2G/',string(i),'.mat'));
            A = sparse(data.A);
        catch
            warning("Skipping %s: could not load file.", string(i));
            c = c + 1;
            continue
        end
        
        if nnz(A) == nnz(diag(A)) || ~issymmetric(A) 
            warning("Skipping %s: matrix is diagonal or not symmetric.", string(i));
            c = c + 1;
            continue
        end

        n = size(A, 1);
        total_nnz = nnz(A);
        [~, eigval] = eigs( A,1, 'largestabs', 'SubspaceDimension', min(n, 100));
        if isnan(eigval)
            c = c + 1;
            continue
        end
        
        switch init_cond
            case "row"
                [~,ind] = max(sum(A.^2,1));
            case "diag"
                [~,ind] = max(abs(diag(A)));
        end
        S = [ind];

        found_level = 0;
        stall = false;
        ever_stalled = false;

        v = zeros(n,1);
        v(ind) = 1;

        prev_eval = 0;
        max_iters = 10000;
        iter_num = 0;

        while found_level < 10 && ~ever_stalled && iter_num < max_iters
            eigval_power = dot(v, A*v);
            curr_eval = abs(eigval_power);
            relerr = abs((abs(eigval_power) - abs(eigval))/abs(eigval));
            for k=1:10
                if relerr < 10^(-k) && found_level < k
                    viewed = nnz(A(S, S)) / total_nnz;
                    viewed_neg(k,c) = viewed;
                    viewed_neg_antistallA(k,c) = viewed;
                    found_level = k;
                end
            end
            if found_level < 10
                iter_num = iter_num + 1;
                % get the next iterate
                v = A * v;
                v = v / norm(v);
                if ~stall
                    Snew = unique([S, find(v ~= 0)']);
                    % check if current neighborhood is stalled
                    if length(S) == length(Snew)
                        r = A(:,S) * randn(length(S),1); % get frontier set
                        Snew = unique([S, find(r~=0)']);
                        if length(S) == length(Snew)
                            stall = true;
                        end
                    else
                        S = Snew;
                    end
                end
                denom = max(abs(prev_eval), eps);
                if stall && (abs(abs(curr_eval) - abs(prev_eval)) / denom < 10^(-13) || length(S) == n)
                    ever_stalled = true;
                    stalled(1,c) = true;
                    viewed_neg_antistallA(found_level+1:end, c) = 1; % if stalled, we view all of A
                end       
            end
            prev_eval = curr_eval;
        end
        c = c + 1;
    end

    % saving everything in the struct
    M.(sprintf('power_viewed_%s', init_cond)) = viewed_neg;
    M.(sprintf('power_antistallA_%s', init_cond)) = viewed_neg_antistallA;
    M.(sprintf('power_stalled_%s', init_cond)) = stalled;
end