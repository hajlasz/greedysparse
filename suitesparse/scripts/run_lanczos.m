function M = run_lanczos(M, init_cond, M_names)
    warning('off', 'MATLAB:eigs:NotConverged');

    % viewed_neg(k, :): fraction of nnz(A) read to first reach 10^-k relerr.
    %   no antistalling: NaN if the level is never reached, else the fraction.
    viewed_neg = NaN(10, length(M_names));

    % viewed_neg_antistallA(k, :): same, but if the run stalls, every level
    %   not yet reached is charged the full matrix (set to 1).
    viewed_neg_antistallA = NaN(10, length(M_names));

    stalled = false(1, length(M_names));

    settle_tol = 1e-13;   % "converged within block": relative change in top Ritz value.

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
        [~, eigval] = eigs(A, 1, 'largestabs', 'SubspaceDimension', min(n, 100));
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

        maxiter    = n;
        beta_floor = 1e-14 * max(abs(eigval), 1);  % numerical guard against 0/0 ONLY, not the stall detector.

        % Lanczos state (no reorthogonalization)
        alphas = zeros(n,1);              % diagonal of T
        betas  = zeros(n,1);              % off-diagonal of T
        q      = zeros(n,1); q(ind) = 1;  % v_1
        qprev  = zeros(n,1);              % v_0
        bprev  = 0;                       % beta_1

        prev_ritz = 0;

        for j = 1:maxiter
            % diagonal entry and current top-magnitude Ritz value
            Aq = A*q;
            alphas(j) = q'*Aq;
            if j == 1
                ritz = alphas(1);
            else
                d  = alphas(1:j);
                e  = betas(1:j-1);
                Tj = diag(d) + diag(e,1) + diag(e,-1);   % j-by-j tridiagonal
                ev = eig(Tj);
                [~, kk] = max(abs(ev));
                ritz = ev(kk);
            end
            curr_ritz = abs(ritz);
            relerr = abs((abs(ritz) - abs(eigval)) / abs(eigval));

            for k = 1:10
                if relerr < 10^(-k) && found_level < k
                    viewed = nnz(A(S,S)) / total_nnz;
                    viewed_neg(k,c)            = viewed;
                    viewed_neg_antistallA(k,c) = viewed;
                    found_level = k;
                end
            end
            if found_level == 10
                break   
            end

            % form next Lanczos vector w = beta_{j+1} * v_{j+1}
            w = Aq - alphas(j)*q - bprev*qprev;
            beta = norm(w);
            betas(j) = beta;

            % check if we have found the whole block
            if ~stall 
                Snew = unique([S, find(w ~= 0)']);
                if length(S) == length(Snew)
                    r = A(:,S) * randn(length(S),1); % find frontier set
                    Snew = unique([S, find(r ~= 0)']);
                    if length(S) == length(Snew)
                        stall = true;
                    else
                        S = Snew;
                    end
                else
                    S = Snew;
                end
            end

            % stall = support frozen AND converged within the block
            denom = max(prev_ritz, eps);
            if stall && abs(curr_ritz - prev_ritz)/denom < settle_tol
                stalled(1,c) = true;
                viewed_neg_antistallA(found_level+1:end, c) = 1;
                break
            end

            % stall = invariant subspace, cannot normalize
            if beta <= beta_floor
                stalled(1,c) = true;
                viewed_neg_antistallA(found_level+1:end, c) = 1;
                break
            end

            % advance recurrence
            qprev     = q;
            q         = w / beta;
            bprev     = beta;
            prev_ritz = curr_ritz;
        end

        c = c + 1;
    end

    % saving everything in the struct (field names match run_power convention)
    M.(sprintf('lanczos_viewed_%s', init_cond))     = viewed_neg;
    M.(sprintf('lanczos_antistallA_%s', init_cond)) = viewed_neg_antistallA;
    M.(sprintf('lanczos_stalled_%s', init_cond))    = stalled;
end