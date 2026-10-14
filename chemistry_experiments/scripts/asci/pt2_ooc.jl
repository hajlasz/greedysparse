#=------------------------------------------------------------------------------
pt2_ooc.jl -- Unscreened Epstein-Nesbet PT2 with sorted disk runs.

For normalized coefficients x on the variational space S and energy E, compute

    b_i = sum_{j in S} H_ij x_j,
    E_PT2 = sum_{i not in S} |b_i|^2 / (E - H_ii).

H is expressed in the spin-flip-symmetrized basis. 

1. Expand all source states in batches, without contribution screening.
2. Map targets to representatives, sum duplicates within each batch, and
   append sorted runs to disk shards.
3. Periodically compact only the runs added since the previous compaction.
   Earlier compacted runs remain separate; duplicate targets can still occur
   in different runs.
4. Stream all runs in each shard, sum every contribution to each external
   state, then square its complete numerator and add the PT2 term. Exclude S
   and skip denominators with |E - H_ii| < gap_floor.

Compaction reduces duplicate storage. Disk usage is checked against
max_disk_gb after each compaction; the limit is measured in GiB (1024^3 bytes).

Include after spin_coupled/batched_expand_and_select_ooc.jl and asci/pt2_core.jl.
------------------------------------------------------------------------------=#

using Printf
using DataStructures: BinaryMinHeap

# -----------------------------------------------------------------------------
# Per-shard streaming PT2 reduction.
#
# Structurally identical to merge_shard_topk's K-way merge (same merge_heap,
# same "drain all runs whose head == cur_det, summing peek_val" inner loop),
# but the per-unique-det body is asci_pt2_spill's drain-phase arithmetic
# instead of top-k insertion. After the inner loop, sum_v is the COMPLETE
# value for external cur_det (summed over every run = every batch), so it is
# exactly -eps * A_a; we skip cur_det in S, skip |gap|<gap_floor, and add
# sum_v^2 * inv_eps2 / (E - H_aa). No Dict: RAM is O(#runs).
#
# Returns (E_partial, n_externals, n_gap_skipped, n_unique).
# -----------------------------------------------------------------------------
function asci_pt2_reduce_shard_compact(
    runs::Vector{RunReader{W}},
    Sset::Set{Det{W}},
    E::Float64,
    inv_eps2::Float64,
    gap_floor::Float64,
    env::QCEnv{W},
    T_plan::SpinFlipPlan,
    C::CoupledConsts,
) where {W}
    e = 0.0; ne = 0; ng = 0; nuniq = 0
    isempty(runs) && return (e, ne, ng, nuniq)

    up = Vector{Int}(undef, env.nup)
    dn = Vector{Int}(undef, env.ndn)

    merge_heap = BinaryMinHeap{MergeHeapEntry{W}}()
    for (i, r) in enumerate(runs)
        if has_more(r)
            push!(merge_heap, MergeHeapEntry{W}(peek_det(r), i))
        end
    end

    while !isempty(merge_heap)
        cur_det = first(merge_heap).d
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

        nuniq += 1
        (cur_det in Sset) && continue          # internal determinant: excluded from PT2

        Hii = asci_coupled_hdiag!(up, dn, cur_det, env, T_plan, C)
        gap = E - Hii
        if abs(gap) < gap_floor
            ng += 1
            continue
        end
        e  += (sum_v * sum_v) * inv_eps2 / gap
        ne += 1
    end

    return (e, ne, ng, nuniq)
end

"""
    asci_pt2_ooc_compact(S, x, E, env, T_plan, C, IRREPS;
                 batch_size, spill_dir, P_out=0, gap_floor=1e-10,
                 verbose=false, gc_every=50, compact_every=200,
                 max_disk_gb=3000.0)
        -> (E_pt2, n_ext, n_gap_skipped, peak_shard)

Disk-spill EN-PT2 with on-disk compaction. Reuses the validated matvec OOC
drain (expand -> hash-to-shard -> periodic lazy compaction) verbatim, and
replaces the top-k step with a streaming PT2 reduction. Result equals
asci_pt2_spill for the same (S, x, E) up to floating-point summation order.

Scaling: peak disk ~ O(unique externals) (~2.1 TB at ozone |S|=1e6, same as the
matvec); peak RAM in the reduce ~ O(#runs) per shard. If disk is tight, LOWER
compact_every (more frequent compaction). If a shard's unique-external count is
large, RAISE P_out -- externals never need to fit in RAM as a Dict, only stream
through the merge heap.
"""
function asci_pt2_ooc_compact(
    S::Vector{Det{W}},
    x::Vector{Float64},
    E::Float64,
    env::QCEnv{W},
    T_plan::SpinFlipPlan,
    C::CoupledConsts,
    IRREPS::Vector{Int};
    batch_size::Int,
    spill_dir::String,
    P_out::Int = 0,
    gap_floor::Float64 = 1e-10,
    verbose::Bool = false,
    gc_every::Int = 50,
    compact_every::Int = 200,
    max_disk_gb::Float64 = 3000.0,
) where {W}

    nS = length(S)
    @assert length(x) == nS
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))

    # norm_fac EXACTLY as asci_pt2_spill and the greedy state: invsqrt2 for
    # time-reversal-fixed determinants, 1 otherwise. Feeds expanding_matvec_
    # coupled! below identically, so on-disk val_a matches asci_pt2_spill.
    norm_fac = Vector{Float64}(undef, nS)
    @inbounds for i in 1:nS
        norm_fac[i] = is_T_fixed(S[i], T_plan) ? C.invsqrt2 : 1.0
    end

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
        println("  asci_pt2_ooc:")
        @printf("    |S| = %d, batch_size = %d, n_batches = %d\n",
                nS, batch_size, n_batches)
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

            t_expand = @elapsed expanding_matvec_coupled!(
                acc, S_batch, x_batch, nf_batch, env, T_plan, C, IRREPS,
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

            # Periodic GC to release dead bucket arrays from acc.shards' empty!()
            # and any drained DrainBuffer pages.
            if b % gc_every == 0
                GC.gc()
            end

            # Periodic LAZY compaction: K-way merge each shard's runs added
            # since the last compaction into a single new compacted run,
            # appended to the existing file. Already-compacted runs from
            # earlier intervals are left untouched.
            #
            # This bounds per-compaction work at O(compact_every batches'
            # worth of new data) instead of O(total data in shard so far),
            # giving constant per-compaction wall time even as the file
            # grows. Trade-off: shard ends up with K compacted runs (where
            # K = n_batches / compact_every) instead of 1; the final
            # K-way merge handles them all.
            #
            # Skip on the very last batch — the final merge_shard_topk pass
            # at the bottom of this function K-way merges all runs straight
            # into top-k, so compacting first would just be wasted I/O.
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
                # the configured limit. Better to error here with a clear
                # message than to crash with ENOSPC on a partial write.
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

        # Free large in-memory state before the merge phase, which spawns its
        # own working set.
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


        # ---------------------------------------------------------------
        # PT2 reduce phase  (replaces the greedy top-k selection).
        #
        # Each representative hashes to exactly ONE shard, so distinct
        # shards hold DISJOINT external sets -> threads accumulate
        # non-overlapping partial sums. Within shard j, asci_pt2_reduce_shard
        # K-way merges all runs (compacted + raw) and sums every partial
        # contribution to each external a BEFORE squaring -> complete A_a.
        # This is the same completeness invariant that makes asci_pt2_spill
        # exact; only the storage is disk-bounded.
        #
        # Accumulators indexed by the LOOP VARIABLE j (one writer per slot),
        # never Threads.threadid() (which in Julia >=1.10 can exceed
        # nthreads() via interactive threads and would index out of bounds).
        # ---------------------------------------------------------------
        t_reduce = time()

        Sset     = Set(S)
        eps_val  = env.epsilon
        inv_eps2 = 1.0 / (eps_val * eps_val)

        E_parts    = zeros(Float64, P_out)
        next_parts = zeros(Int,     P_out)
        ngap_parts = zeros(Int,     P_out)
        peak_parts = zeros(Int,     P_out)

        Threads.@threads :static for j in 1:P_out
            runs = open_runs_for_shard(shard_paths[j], Det{W})
            e, ne, ng, nu = asci_pt2_reduce_shard_compact(
                runs, Sset, E, inv_eps2, gap_floor, env, T_plan, C,
            )
            for r in runs
                close_run!(r)
            end
            E_parts[j]    = e
            next_parts[j] = ne
            ngap_parts[j] = ng
            peak_parts[j] = nu
        end

        E_pt2 = sum(E_parts)
        n_ext = sum(next_parts)
        n_gap = sum(ngap_parts)
        peak  = maximum(peak_parts)

        if verbose
            @printf("    PT2 reduce: %.2fs  externals=%d  gap-skip=%d  peak-shard=%d  (total %.2fs)\n",
                    time() - t_reduce, n_ext, n_gap, peak, time() - t_total)
            @printf("    E_PT2 = %.10f Ha\n", E_pt2)
            flush(stdout)
        end

        return (E_pt2 = E_pt2, n_ext = n_ext, n_gap_skipped = n_gap, peak_shard = peak)

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