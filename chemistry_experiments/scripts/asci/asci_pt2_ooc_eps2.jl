#=------------------------------------------------------------------------------
asci_pt2_ooc_eps2.jl -- Screened Epstein-Nesbet PT2 with sorted disk runs.

Uses the PT2 sum and full spin-flip-symmetrized diagonal in pt2_ooc.jl.
For eps2 > 0, emit an off-diagonal determinant-level term only when

    |h_ij x_j N_j N_i| >= eps2,

where N_j and N_i are the source and target normalization factors. Screening
occurs before spin-flip representatives are merged or contributions summed;
it does not test the complete coupled matrix element or final numerator.
The cutoff excludes the matvec scaling epsilon. eps2 <= 0 uses the
unscreened expansion.

1. Expand source batches and write retained contributions as sorted runs.
2. Periodically compact only runs added since the previous compaction;
   earlier compacted runs remain separate.
3. Stream all runs, sum every retained contribution for each external state,
   then add |b_i^(eps2)|^2 / (E - H_ii). Exclude the variational space and
   skip denominators with |E - H_ii| < gap_floor.

Compaction and screening reduce storage. Disk usage is checked after each
compaction against max_disk_gb, in GiB (1024^3 bytes). Sampled peak usage is
returned in bytes as peak_disk_bytes.
------------------------------------------------------------------------------=#

using Printf
using DataStructures: BinaryMinHeap

function asci_pt2_reduce_shard_compact_eps2(
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
    asci_pt2_ooc_compact_eps2(S, x, E, env, T_plan, C, IRREPS;
                 batch_size, spill_dir, P_out=0, gap_floor=1e-10,
                 verbose=false, gc_every=50, compact_every=200,
                 max_disk_gb=3000.0)
        -> (E_pt2, n_ext, n_gap_skipped, peak_shard)

Disk-spill EN-PT2 with on-disk compaction. 
"""
function asci_pt2_ooc_compact_eps2(
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
    eps2::Float64 = 0.0,
) where {W}

    nS = length(S)
    @assert length(x) == nS
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
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

    # Largest total spill seen at any checkpoint, in BYTES. Reported so the
    # eps_2 calibration can read off U = peak_disk / 32 (unique externals).
    peak_disk = 0

    if verbose
        println("  asci_pt2_ooc_eps2:")
        @printf("    eps2 = %.3e  (%s)\n", eps2,
                eps2 <= 0.0 ? "EXACT, no screening" : "screened")
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

            # Branch copied verbatim from pt2_spill.jl:142-145. At eps2 = 0
            # we take the unscreened call, so this path stays bit-identical
            # to pt2_ooc.jl.
            t_expand = @elapsed if eps2 <= 0.0
                expanding_matvec_coupled!(
                    acc, S_batch, x_batch, nf_batch, env, T_plan, C, IRREPS,
                )
            else
                asci_expand_screened!(
                    acc, S_batch, x_batch, nf_batch, env, T_plan, C, IRREPS;
                    cutoff = eps2,
                )
            end

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

            if b % compact_every == 0 && b < n_batches
                t_compact = @elapsed begin
                    Threads.@threads :static for j in 1:P_out
                        new_off, _n_emitted = lazy_compact_shard!(
                            shard_paths[j], file_handles[j],
                            last_compacted_offset[j], Det{W},
                        )
                        file_handles[j] = open(shard_paths[j], "a")
                        last_compacted_offset[j] = new_off
                    end
                    GC.gc()
                end
                disk_after = sum(filesize(p) for p in shard_paths if isfile(p))
                peak_disk  = max(peak_disk, disk_after)
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

        disk_total = sum(filesize(p) for p in shard_paths if isfile(p))
        peak_disk  = max(peak_disk, disk_total)
        if verbose
            @printf("    drain done in %.1fs.  Total spill on disk: %.2f GiB\n",
                    time() - t_total, disk_total / (1024.0^3))
            @printf("    PEAK DISK = %.2f GiB   ->  U = %.4e unique externals\n",
                    peak_disk / (1024.0^3), peak_disk / 32.0)
            flush(stdout)
        end

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
            e, ne, ng, nu = asci_pt2_reduce_shard_compact_eps2(
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

        return (E_pt2 = E_pt2, n_ext = n_ext, n_gap_skipped = n_gap, peak_shard = peak,
                peak_disk_bytes = peak_disk)

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


method_params_line(::Val{:ooc_eps2}, cfg::PT2Config; P_out, batch_size, spill_dir,
                   compact_every, max_disk_gb, eps2, kwargs...) =
    @sprintf("PT2 params (:ooc_eps2, DISK-bounded): P_out=%d, batch_size=%d, compact_every=%d, max_disk_gb=%.0f GiB, eps2=%.3e",
             P_out, batch_size, compact_every, max_disk_gb, eps2)

function launch(::Val{:ooc_eps2}, cfg::PT2Config, S, v, E_var, env, T_plan, C, IRREPS;
                P_out, batch_size, spill_dir, compact_every, max_disk_gb, eps2, kwargs...)
    return asci_pt2_ooc_compact_eps2(S, v, E_var, env, T_plan, C, IRREPS;
                                     batch_size    = batch_size,
                                     spill_dir     = spill_dir,
                                     P_out         = P_out,
                                     compact_every = compact_every,
                                     max_disk_gb   = max_disk_gb,
                                     eps2          = eps2,
                                     verbose       = true)
end
