function M = run_greedy(M, init_cond, M_names)
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
            warning("Skipping %s: eigs failed to converge.", string(i));
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
        ever_stalled = false;

        while found_level < 10 && ~ever_stalled
            Asub = A(S,S);

            [~, eigval_greedy] = eigs(Asub, 1, 'largestabs');
            if isnan(eigval_greedy)
                warning("In some iteration, eigs failed to converge.", string(i));
                break
            end 
            relerr = abs((abs(eigval_greedy) - abs(eigval))/abs(eigval));

            for k=1:10
                if relerr < 10^(-k) && found_level < k
                    frac_viewed = nnz(Asub) / total_nnz;
                    viewed_neg_antistallA(k,c) = frac_viewed;
                    viewed_neg(k,c) = frac_viewed;
                    found_level = k;
                end
            end

            if found_level < 10
                % get the next bag
                r = A(:,S) * randn(length(S),1); % get frontier set
                Snew = unique([S, find(r~=0)']);
                
                % check for stalling
                if (isempty(Snew)) || (length(S) == length(Snew) && isequal(S, Snew))
                    stalled(1,c) = true;
                    ever_stalled = true;
                    viewed_neg_antistallA(found_level+1:end, c) = 1; % if stalled, we view all of A
                else
                    S = Snew;
                end
            end
        end
        c = c + 1;
    end

    % saving everything in the struct
    M.(sprintf('greedy_viewed_%s', init_cond)) = viewed_neg;
    M.(sprintf('greedy_antistallA_%s', init_cond)) = viewed_neg_antistallA;
    M.(sprintf('greedy_stalled_%s', init_cond)) = stalled;
end