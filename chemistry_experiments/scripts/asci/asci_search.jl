#=------------------------------------------------------------------------------
asci_search.jl

Search and selection for ASCI in the spin-flip-symmetrized basis. 
------------------------------------------------------------------------------=#

# -----------------------------------------------------------------------------
# asci_keep_contrib  
#
# A single contribution (source j -> target i) is KEPT iff its first-order
# partial score clears the threshold:
#
#     |S_i^(j)| = | H_ij x_j N_d N_j / (E - H_ii) |  >=  eps_thresh        
#
# where H_ii is the spin-coupled diagonal of the TARGET determinant.  We test it
# in the division-free, degenerate-safe form  |contrib| >= eps_thresh*|E - H_ii|,
# with contrib = H_ij x_j N_d N_j (the numerator the loop already holds).
# -----------------------------------------------------------------------------
@inline function asci_keep_contrib(
    contrib::Float64,
    d2::Det{W},
    env::QCEnv{W},
    E::Float64,
    eps_thresh::Float64,
    t_up::Vector{Int},
    t_dn::Vector{Int},
    T_plan::SpinFlipPlan,
    C::CoupledConsts,
)::Bool where {W}
    eps_thresh <= 0.0 && return true
    Hii = asci_coupled_hdiag!(t_up, t_dn, d2, env, T_plan, C)
    gap = E - Hii
    abs(gap) <= 1e-10 && return true          # near-degenerate: |score| huge -> keep
    return abs(contrib) >= eps_thresh * abs(gap)
end

# -----------------------------------------------------------------------------
# Doubles emission for the ASCI expand.
# -----------------------------------------------------------------------------
function asci_emit_double_contributions_buf!(
    buf::UpdateBuffer{W},
    d::Det{W},
    xi::Float64,
    Nd::Float64,
    up_occ::Vector{Int},
    dn_occ::Vector{Int},
    env::QCEnv{W},
    T_plan::SpinFlipPlan,
    C::CoupledConsts,
    IRREPS::Vector{Int},
    P::Int,
    n::Int,
    n2::Int,
    n3::Int,
    frz::Int,
    acc_shards::Vector{RobinDict{Det{W},Float64}},
    locks::Vector{Threads.SpinLock},
    eps_thresh::Float64,
    E::Float64,
    t_up::Vector{Int},
    t_dn::Vector{Int},
) where {W}

    Dss = env.doub_same
    Dod = env.doub_diff
    eps = env.epsilon

    # =====================================================================
    # DOUBLES: UP-UP (same spin)
    # =====================================================================
    @inbounds for p in 1:(env.nup - 1)
        i = up_occ[p]
        (i <= frz) && continue
        i_so = i
        iri = IRREPS[i]

        for q in (p + 1):env.nup
            j = up_occ[q]
            (j <= frz) && continue
            j_so = j

            key = iri ⊻ IRREPS[j]
            cand = Dss[key + 1]

            for u in 1:length(cand)
                a, b = cand[u]
                (a <= frz || b <= frz) && continue
                a_so = a
                b_so = b

                if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                    d2  = flip4(d, i_so, j_so, a_so, b_so, env.om)
                    sgn = double_excitation_sign_via_two_singles(d, i_so, j_so, a_so, b_so, env.om)
                    hij = double_excitation_value(i_so, j_so, a_so, b_so, env.eris1, n;
                                                  n2=n2, n3=n3, sign=sgn)
                    Nj  = compute_target_norm(d2, T_plan, C)
                    contrib = hij * xi * Nd * Nj
                    asci_keep_contrib(contrib, d2, env, E, eps_thresh, t_up, t_dn, T_plan, C) || continue
                    buf_push!(buf, d2, -eps * contrib,
                              owner_of(d2, P), acc_shards, locks)
                end
            end
        end
    end

    # =====================================================================
    # DOUBLES: DOWN-DOWN (same spin)
    # =====================================================================
    @inbounds for p in 1:(env.ndn - 1)
        i_so = dn_occ[p]
        i = i_so - n
        (i <= frz) && continue
        iri = IRREPS[i]

        for q in (p + 1):env.ndn
            j_so = dn_occ[q]
            j = j_so - n
            (j <= frz) && continue

            key = iri ⊻ IRREPS[j]
            cand = Dss[key + 1]

            for u in 1:length(cand)
                a, b = cand[u]
                (a <= frz || b <= frz) && continue
                a_so = a + n
                b_so = b + n

                if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                    d2  = flip4(d, i_so, j_so, a_so, b_so, env.om)
                    sgn = double_excitation_sign_via_two_singles(d, i_so, j_so, a_so, b_so, env.om)
                    hij = double_excitation_value(i_so, j_so, a_so, b_so, env.eris1, n;
                                                  n2=n2, n3=n3, sign=sgn)
                    Nj  = compute_target_norm(d2, T_plan, C)
                    contrib = hij * xi * Nd * Nj
                    asci_keep_contrib(contrib, d2, env, E, eps_thresh, t_up, t_dn, T_plan, C) || continue
                    buf_push!(buf, d2, -eps * contrib,
                              owner_of(d2, P), acc_shards, locks)
                end
            end
        end
    end

    # =====================================================================
    # DOUBLES: UP-DOWN (opposite spin)
    # =====================================================================
    @inbounds for p in 1:env.nup
        i = up_occ[p]
        (i <= frz) && continue
        i_so = i
        iri = IRREPS[i]

        for q in 1:env.ndn
            j_so = dn_occ[q]
            j = j_so - n
            (j <= frz) && continue

            key = iri ⊻ IRREPS[j]
            cand = Dod[key + 1]

            for u in 1:length(cand)
                a, b = cand[u]
                (a <= frz || b <= frz) && continue
                a_so = a
                b_so = b + n

                if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                    d2  = flip4(d, i_so, j_so, a_so, b_so, env.om)
                    sgn = double_excitation_sign_via_two_singles(d, i_so, j_so, a_so, b_so, env.om)
                    hij = double_excitation_value(i_so, j_so, a_so, b_so, env.eris1, n;
                                                  n2=n2, n3=n3, sign=sgn)
                    Nj  = compute_target_norm(d2, T_plan, C)
                    contrib = hij * xi * Nd * Nj
                    asci_keep_contrib(contrib, d2, env, E, eps_thresh, t_up, t_dn, T_plan, C) || continue
                    buf_push!(buf, d2, -eps * contrib,
                              owner_of(d2, P), acc_shards, locks)
                end
            end
        end
    end

    return nothing
end

# -----------------------------------------------------------------------------
# The ASCI expand.
#
# -----------------------------------------------------------------------------
function asci_expand_coupled!(
    acc::ExpandingAccumulator{W},
    S::Vector{Det{W}},
    x::Vector{Float64},
    norm_fac::Vector{Float64},
    env::QCEnv{W},
    T_plan::SpinFlipPlan,
    C::CoupledConsts,
    IRREPS::Vector{Int};
    chunk_size::Int = 0,
    buf_capacity::Int = 4096,
    eps_thresh::Float64 = 0.0,
    E::Float64 = 0.0,
) where {W}

    nS  = length(S)
    P   = acc.P
    n   = env.norb_tot
    n2  = n * n
    n3  = n2 * n
    frz = env.norbfrzn
    nt  = Threads.nthreads()

    if chunk_size <= 0
        chunk_size = max(1, cld(nS, 8 * nt))
    end

    chunks = collect(Iterators.partition(1:nS, chunk_size))

    # Cache references to shards/locks to avoid repeated field access
    acc_shards = acc.shards
    acc_locks  = acc.locks
    Su         = env.single_targets
    om         = env.om
    eps        = env.epsilon

    tasks = map(chunks) do chunk
        Threads.@spawn begin
            # Task-local scratch — allocated once per chunk, immune to task
            # migration because it lives in this task's closure.
            up_occ = Vector{Int}(undef, env.nup)
            dn_occ = Vector{Int}(undef, env.ndn)
            t_up   = Vector{Int}(undef, env.nup)   # target-occ scratch for the screen
            t_dn   = Vector{Int}(undef, env.ndn)
            buf    = UpdateBuffer{W}(P; capacity=buf_capacity)

            @inbounds for i in chunk
                xi = x[i]
                d  = S[i]
                Nd = norm_fac[i]

                occupied_orbitals!(up_occ, dn_occ, d, n, env.nup, env.ndn)

                # -------- (ASCI) Diagonal term intentionally NOT emitted --------

                # -------- Singles: up-spin --------
                for t in 1:env.nup
                    ii = up_occ[t]
                    (ii <= frz) && continue
                    for a in Su[ii]
                        (a <= frz) && continue
                        !isocc_fast(d, a, om) || continue

                        d2  = flip2(d, ii, a, om)
                        sgn = single_excitation_sign(d, ii, a)
                        hij = single_excitation_value_from_occ(up_occ, dn_occ, ii, a,
                                                               env.HCORE, env.eris1, n;
                                                               n2=n2, n3=n3, sign=sgn)
                        Nj  = compute_target_norm(d2, T_plan, C)
                        contrib = hij * xi * Nd * Nj
                        asci_keep_contrib(contrib, d2, env, E, eps_thresh, t_up, t_dn, T_plan, C) || continue
                        buf_push!(buf, d2, -eps * contrib,
                                  owner_of(d2, P), acc_shards, acc_locks)
                    end
                end

                # -------- Singles: down-spin --------
                for t in 1:env.ndn
                    ii_so = dn_occ[t]
                    ii    = ii_so - n
                    (ii <= frz) && continue
                    for a in Su[ii]
                        (a <= frz) && continue
                        a_so = a + n
                        !isocc_fast(d, a_so, om) || continue

                        d2  = flip2(d, ii_so, a_so, om)
                        sgn = single_excitation_sign(d, ii_so, a_so)
                        hij = single_excitation_value_from_occ(up_occ, dn_occ, ii_so, a_so,
                                                               env.HCORE, env.eris1, n;
                                                               n2=n2, n3=n3, sign=sgn)
                        Nj  = compute_target_norm(d2, T_plan, C)
                        contrib = hij * xi * Nd * Nj
                        asci_keep_contrib(contrib, d2, env, E, eps_thresh, t_up, t_dn, T_plan, C) || continue
                        buf_push!(buf, d2, -eps * contrib,
                                  owner_of(d2, P), acc_shards, acc_locks)
                    end
                end

                # -------- Doubles: up-up, down-down, up-down --------
                asci_emit_double_contributions_buf!(buf, d, xi, Nd, up_occ, dn_occ,
                                                    env, T_plan, C, IRREPS, P,
                                                    n, n2, n3, frz,
                                                    acc_shards, acc_locks, eps_thresh,
                                                    E, t_up, t_dn)
            end

            # Final flush: drain this task's buffer to the global shards.
            flush!(buf, acc_shards, acc_locks)
        end
    end

    foreach(wait, tasks)
    return nothing
end


# =============================================================================
# STEP 2 — the ASCI score, and the top-k merge that applies it
# =============================================================================

"""
    asci_score(sum_v, d, env, E, up_occ, dn_occ, T_plan, C) -> Float64

Turn the accumulated numerator for one target determinant `d` into its
first-order (Epstein-Nesbet) ASCI score.
"""
@inline function asci_score(
    sum_v::Float64,
    d::Det{W},
    env::QCEnv{W},
    E::Float64,
    up_occ::Vector{Int},
    dn_occ::Vector{Int},
    T_plan::SpinFlipPlan,
    C::CoupledConsts,
)::Float64 where {W}
    Hdd = asci_coupled_hdiag!(up_occ, dn_occ, d, env, T_plan, C)
    gap = Hdd - E
    return abs(gap) > 1e-10 ? sum_v / (env.epsilon * gap) : 0.0
end

# -----------------------------------------------------------------------------
# asci_merge_shard_topk
# -----------------------------------------------------------------------------
function asci_merge_shard_topk(
    runs::Vector{RunReader{W}}, k_local::Int,
    env::QCEnv{W}, E::Float64,
    T_plan::SpinFlipPlan, C::CoupledConsts,
) where {W}
    isempty(runs) && return Det{W}[], Float64[]

    topk = BinaryMinHeap{TopKItem{W}}()
    tiebreak = Int64(0)

    up_occ = Vector{Int}(undef, env.nup)
    dn_occ = Vector{Int}(undef, env.ndn)

    merge_heap = BinaryMinHeap{MergeHeapEntry{W}}()
    for (i, r) in enumerate(runs)
        if has_more(r)
            push!(merge_heap, MergeHeapEntry{W}(peek_det(r), i))
        end
    end

    while !isempty(merge_heap)
        top = first(merge_heap)
        cur_det = top.d
        sum_v   = 0.0

        while !isempty(merge_heap) && first(merge_heap).d == cur_det
            entry = pop!(merge_heap)
            r = runs[entry.run_idx]
            sum_v += peek_val(r)
            advance!(r)
            if has_more(r)
                push!(merge_heap, MergeHeapEntry{W}(peek_det(r), entry.run_idx))
            end
        end

        score = asci_score(sum_v, cur_det, env, E, up_occ, dn_occ, T_plan, C)
        a = abs(score)
        if score != 0.0
            tiebreak += 1
            item = TopKItem{W}(a, tiebreak, cur_det, score)
            if length(topk) < k_local
                push!(topk, item)
            elseif a > first(topk).a
                pop!(topk)
                push!(topk, item)
            end
        end
    end

    n = length(topk)
    items = Vector{TopKItem{W}}(undef, n)
    @inbounds for i in 1:n
        items[i] = pop!(topk)
    end
    reverse!(items)

    S_local = Vector{Det{W}}(undef, n)
    x_local = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        S_local[i] = items[i].d
        x_local[i] = items[i].v
    end
    return S_local, x_local
end


# =============================================================================
# STEP 3 - the ASCI candidate search (out-of-core)
# =============================================================================
function asci_search_ooc(
    S::Vector{Det{W}},
    x::Vector{Float64},
    norm_fac::Vector{Float64},
    env::QCEnv{W},
    T_plan::SpinFlipPlan,
    C::CoupledConsts,
    IRREPS::Vector{Int},
    k::Int,
    E::Float64;
    batch_size::Int,
    spill_dir::String,
    eps_thresh::Float64 = 0.0,
    P_out::Int = 0,
    verbose::Bool = false,
    gc_every::Int = 50,        # force GC every N batches to release dead bucket arrays
    compact_every::Int = 200,  # compact each shard's runs every N batches
    max_disk_gb::Float64 = 3000.0,  # abort if total spill exceeds this many GB
) where {W}

    nS = length(S)
    @assert length(x) == nS
    @assert length(norm_fac) == nS
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))

    nt    = Threads.nthreads()
    P_in  = max(64, 10 * nt)
    P_out = P_out > 0 ? P_out : max(64, 2 * nt)

    isdir(spill_dir) || mkpath(spill_dir)
    shard_paths = [joinpath(spill_dir, "shard_$(j).bin") for j in 1:P_out]
    for p in shard_paths
        isfile(p) && rm(p)
    end

    file_handles = IOStream[open(p, "w") for p in shard_paths]

    # Track which prefix of each shard file is "already compacted." The
    # lazy compaction step only re-merges runs added since this offset.
    # Initialized to 0 because we just opened fresh empty files.
    last_compacted_offset = zeros(Int64, P_out)

    expected_per_batch = 10 * batch_size
    acc = ExpandingAccumulator{W}(P_in, expected_per_batch)

    initial_buf_cap = 4096
    buf_tls = [[DrainBuffer{W}(initial_buf_cap) for _ in 1:P_out] for _ in 1:nt]

    # Pre-allocated per-shard scratch (no per-call allocations in sort_dedup)
    sort_scratch_dets             = [Vector{Det{W}}(undef, 0)   for _ in 1:P_out]
    sort_scratch_vals             = [Vector{Float64}(undef, 0)  for _ in 1:P_out]
    perm_scratch_per_shard        = [Vector{Int}(undef, 0)      for _ in 1:P_out]
    sorted_dets_scratch_per_shard = [Vector{Det{W}}(undef, 0)   for _ in 1:P_out]
    sorted_vals_scratch_per_shard = [Vector{Float64}(undef, 0)  for _ in 1:P_out]

    write_byte_bufs = [Vector{UInt8}(undef, 4 + 1024 * rec_bytes(Det{W}))
                       for _ in 1:P_out]

    n_batches = cld(nS, batch_size)
    t_total   = time()

    if verbose
        println("  asci_search_ooc:")
        @printf("    |S| = %d, k = %d, batch_size = %d, n_batches = %d\n",
                nS, k, batch_size, n_batches)
        @printf("    nthreads = %d, P_in = %d, P_out = %d\n", nt, P_in, P_out)
        @printf("    spill_dir = %s\n", spill_dir)
        flush(stdout)
    end

    try
        for b in 1:n_batches
            i_start = (b - 1) * batch_size + 1
            i_end   = min(b * batch_size, nS)

            S_batch  = collect(view(S,        i_start:i_end))
            x_batch  = collect(view(x,        i_start:i_end))
            nf_batch = collect(view(norm_fac, i_start:i_end))

            t_expand = @elapsed asci_expand_coupled!(
                acc, S_batch, x_batch, nf_batch, env, T_plan, C, IRREPS;
                eps_thresh = eps_thresh,
                E = E,
            )

            t_p1 = @elapsed drain_pass1_ooc!(acc, buf_tls, T_plan, P_out)

            t_p2 = @elapsed drain_pass2_ooc!(
                buf_tls, file_handles,
                sort_scratch_dets, sort_scratch_vals,
                write_byte_bufs,
                perm_scratch_per_shard,
                sorted_dets_scratch_per_shard,
                sorted_vals_scratch_per_shard,
            )

            if b % gc_every == 0
                GC.gc()
            end

            # Periodic LAZY compaction: K-way merge each shard's runs added
            # since the last compaction into a single new compacted run,
            # appended to the existing file. Already-compacted runs from
            # earlier intervals are left untouched.
            if b % compact_every == 0 && b < n_batches
                t_compact = @elapsed begin
                    Threads.@threads :static for j in 1:P_out
                        new_off, _n_emitted = lazy_compact_shard!(
                            shard_paths[j], file_handles[j],
                            last_compacted_offset[j], Det{W},
                        )
                        # File was closed inside lazy_compact_shard!; reopen
                        # for append to continue receiving new runs.
                        file_handles[j] = open(shard_paths[j], "a")
                        last_compacted_offset[j] = new_off
                    end
                    GC.gc()
                end
                disk_after = sum(filesize(p) for p in shard_paths if isfile(p))
                if verbose
                    @printf("    [batch %d] lazy compaction: %.1fs, disk after = %.2f GB\n",
                            b, t_compact, disk_after / (1024.0^3))
                    flush(stdout)
                end

                # Disk-budget enforcement: abort cleanly if we're approaching
                # the configured limit. 
                if disk_after / (1024.0^3) > max_disk_gb
                    error("OOC drain disk usage $(round(disk_after / 1024.0^3, digits=1)) GB " *
                          "exceeds budget $(max_disk_gb) GB at batch $b/$n_batches. " *
                          "Aborting before disk fills. Either: (1) increase max_disk_gb if " *
                          "your /scratch quota allows, (2) decrease compact_every for more " *
                          "frequent compactions, (3) reduce m to limit total emissions.")
                end
            end

            if verbose && (b == 1 || b == n_batches || b % max(1, cld(n_batches, 20)) == 0)
                @printf("    batch %d/%d: expand %.2fs, p1 %.2fs, p2 %.2fs\n",
                        b, n_batches, t_expand, t_p1, t_p2)
                flush(stdout)
            end
        end

        for fh in file_handles
            try
                isopen(fh) && close(fh)
            catch
            end
        end

        empty!(buf_tls)
        empty!(sort_scratch_dets)
        empty!(sort_scratch_vals)
        empty!(perm_scratch_per_shard)
        empty!(sorted_dets_scratch_per_shard)
        empty!(sorted_vals_scratch_per_shard)
        empty!(write_byte_bufs)
        # acc held by closure; clear shards explicitly
        for s in acc.shards
            empty!(s)
        end
        GC.gc()

        if verbose
            disk_total = sum(filesize(p) for p in shard_paths if isfile(p))
            @printf("    drain done in %.1fs.  Total spill on disk: %.2f GB\n",
                    time() - t_total, disk_total / (1024.0^3))
            flush(stdout)
        end

        # Top-k phase
        t_merge = time()

        local_S = Vector{Vector{Det{W}}}(undef, P_out)
        local_x = Vector{Vector{Float64}}(undef, P_out)

        Threads.@threads :static for j in 1:P_out
            runs = open_runs_for_shard(shard_paths[j], Det{W})
            sj, xj = asci_merge_shard_topk(runs, k, env, E, T_plan, C)
            for r in runs
                close_run!(r)
            end
            local_S[j] = sj
            local_x[j] = xj
        end

        global_heap = BinaryMinHeap{TopKItem{W}}()
        tiebreak = Int64(0)
        for j in 1:P_out
            sj = local_S[j]
            xj = local_x[j]
            for i in 1:length(xj)
                a = abs(xj[i])
                tiebreak += 1
                item = TopKItem{W}(a, tiebreak, sj[i], xj[i])
                if length(global_heap) < k
                    push!(global_heap, item)
                elseif a > first(global_heap).a
                    pop!(global_heap)
                    push!(global_heap, item)
                end
            end
        end

        n_final = length(global_heap)
        items = Vector{TopKItem{W}}(undef, n_final)
        @inbounds for i in 1:n_final
            items[i] = pop!(global_heap)
        end
        reverse!(items)

        S_new = Vector{Det{W}}(undef, n_final)
        x_new = Vector{Float64}(undef, n_final)
        @inbounds for i in 1:n_final
            S_new[i] = items[i].d
            x_new[i] = items[i].v
        end

        if verbose
            @printf("    merge+topk: %.2fs  → returning %d entries (total %.2fs)\n",
                    time() - t_merge, n_final, time() - t_total)
            flush(stdout)
        end

        return S_new, x_new

    finally
        for fh in file_handles
            try
                isopen(fh) && close(fh)
            catch
            end
        end
        for p in shard_paths
            try
                isfile(p) && rm(p)
            catch
            end
        end
    end
end


# =============================================================================
# STEP 4 — merge scored candidates with the current determinants
# =============================================================================

"""
    asci_merge_include(S_inc, v_inc, cand_dets, cand_scores, N_tdets) -> (S_new, v0)

all core determinants included
"""
function asci_merge_include(
    S_inc::Vector{Det{W}},
    v_inc::Vector{Float64},
    cand_dets::Vector{Det{W}},
    cand_scores::Vector{Float64},
    N_tdets::Int,
) where {W}
    @assert length(S_inc) == length(v_inc) "S_inc and v_inc length mismatch"
    @assert length(cand_dets) == length(cand_scores) "cand_dets and cand_scores length mismatch"

    n_inc = length(S_inc)

    if n_inc >= N_tdets
        keep = N_tdets
        if n_inc == keep
            return copy(S_inc), copy(v_inc)
        end
        idx = collect(1:n_inc)
        partialsort!(idx, 1:keep, by = i -> -abs(v_inc[i]))
        S_new = Vector{Det{W}}(undef, keep)
        v0    = Vector{Float64}(undef, keep)
        @inbounds for t in 1:keep
            i = idx[t]
            S_new[t] = S_inc[i]
            v0[t]    = v_inc[i]
        end
        return S_new, v0
    end

    incumbent = Set{Det{W}}()
    sizehint!(incumbent, n_inc)
    @inbounds for d in S_inc
        push!(incumbent, d)
    end

    gdets   = Det{W}[]
    gscores = Float64[]
    @inbounds for j in 1:length(cand_dets)
        c = cand_dets[j]
        if !(c in incumbent)
            push!(gdets, c)
            push!(gscores, cand_scores[j])
        end
    end

    n_slots = N_tdets - n_inc          # candidate budget (> 0 in this branch)
    n_gen   = length(gdets)
    n_cand  = min(n_slots, n_gen)

    S_new = Vector{Det{W}}(undef, n_inc + n_cand)
    v0    = Vector{Float64}(undef, n_inc + n_cand)

    # keep core determinants
    @inbounds for i in 1:n_inc
        S_new[i] = S_inc[i]
        v0[i]    = v_inc[i]
    end

    if n_cand == n_gen
        @inbounds for j in 1:n_gen
            S_new[n_inc + j] = gdets[j]
            v0[n_inc + j]    = gscores[j]
        end
    else
        # top n_cand genuine candidates by |score|
        idx = collect(1:n_gen)
        partialsort!(idx, 1:n_cand, by = j -> -abs(gscores[j]))
        @inbounds for t in 1:n_cand
            j = idx[t]
            S_new[n_inc + t] = gdets[j]
            v0[n_inc + t]    = gscores[j]
        end
    end

    return S_new, v0
end


"""
    asci_merge_rank(S_cur, v_cur, cand_dets, cand_scores, N_tdets)
        -> (S_new, v0)

Form the next ASCI variational space by merging the current determinants
("incumbents", carrying their true CI coefficients `v_cur`) with the scored
candidates from `asci_search_ooc`, and keeping the `N_tdets` most important.
"""
function asci_merge_rank(
    S_cur::Vector{Det{W}},
    v_cur::Vector{Float64},
    cand_dets::Vector{Det{W}},
    cand_scores::Vector{Float64},
    N_tdets::Int,
) where {W}
    @assert length(S_cur) == length(v_cur) "S_cur and v_cur length mismatch"
    @assert length(cand_dets) == length(cand_scores) "cand_dets and cand_scores length mismatch"

    n_inc = length(S_cur)

    # Incumbent membership set, used to drop re-found candidates.
    incumbent = Set{Det{W}}()
    sizehint!(incumbent, n_inc)
    @inbounds for d in S_cur
        push!(incumbent, d)
    end

    # Combined pool: incumbents (coeff = v_cur), then candidates NOT already
    # incumbents (coeff = score).  Importance is |coeff|.
    dets  = Vector{Det{W}}(undef, n_inc)
    coeff = Vector{Float64}(undef, n_inc)
    @inbounds for i in 1:n_inc
        dets[i]  = S_cur[i]
        coeff[i] = v_cur[i]
    end
    @inbounds for j in 1:length(cand_dets)
        c = cand_dets[j]
        if !(c in incumbent)
            push!(dets, c)
            push!(coeff, cand_scores[j])
        end
    end

    ntot = length(dets)
    keep = min(N_tdets, ntot)

    # Everything fits -> no selection needed.
    if keep == ntot
        return dets, coeff
    end

    # Keep the `keep` determinants of largest |coeff| (quickselect, like MACIS).
    idx = collect(1:ntot)
    partialsort!(idx, 1:keep, by = i -> -abs(coeff[i]))

    S_new = Vector{Det{W}}(undef, keep)
    v0    = Vector{Float64}(undef, keep)
    @inbounds for t in 1:keep
        i = idx[t]
        S_new[t] = dets[i]
        v0[t]    = coeff[i]
    end
    return S_new, v0
end
