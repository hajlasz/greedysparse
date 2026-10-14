#=------------------------------------------------------------------------------
pt2_spill.jl -- Epstein-Nesbet PT2 with disk shards and in-memory reduction.

Include after:
    spin_coupled/batched_expand_and_select_ooc.jl
    asci/pt2_core.jl
    asci/pt2_expand.jl
------------------------------------------------------------------------------=#

using Printf

# ---- write every thread-local shard buffer to its file (one writer per file) --
function _pt2_spill_write!(ios::Vector{IOStream},
                           buf_tls::Vector{Vector{DrainBuffer{W}}},
                           ::Val{W}) where {W}
    nt       = length(buf_tls)
    n_shards = length(ios)
    Threads.@threads :static for j in 1:n_shards
        tmp = IOBuffer()
        @inbounds for tid in 1:nt
            b = buf_tls[tid][j]
            for i in 1:b.len
                d = b.dets[i]
                for k in 1:W
                    write(tmp, d.words[k])       # W words, in order
                end
                write(tmp, b.vals[i])            # then the Float64 value
            end
            reset_buf!(b)                        # keep backing arrays, reset length
        end
        write(ios[j], take!(tmp))                # single block write to shard j
    end
    return nothing
end

# ---- read one shard file, summing duplicate reps → Dict rep -> val ------------
function _pt2_spill_read_merge(path::String, ::Val{W}) where {W}
    merged = Dict{Det{W},Float64}()
    io = open(path, "r")
    try
        while !eof(io)
            words = ntuple(_ -> read(io, UInt64), Val(W))   # W words, in order
            v     = read(io, Float64)
            d     = Det{W}(words)
            merged[d] = get(merged, d, 0.0) + v
        end
    finally
        close(io)
    end
    return merged
end

"""
    asci_pt2_spill(S, x, E, env, T_plan, C, IRREPS;
                   batch_size, n_shards, spill_dir, gap_floor, keep_files, verbose)
        -> (E_pt2, n_ext, n_gap_skipped, peak_shard)

Disk-spill EN-PT2.  Result equals asci_pt2_incore for the same (S, x, E) up to
floating-point summation order.  Peak RAM ≈ largest shard's unique externals.
"""
function asci_pt2_spill(
    S::Vector{Det{W}}, x::Vector{Float64}, E::Float64,
    env::QCEnv{W}, T_plan::SpinFlipPlan, C::CoupledConsts, IRREPS::Vector{Int};
    batch_size::Int,
    eps2::Float64 = 0.0,
    n_shards::Int = 0,
    spill_dir::String = mktempdir(),
    gap_floor::Float64 = 1e-10,
    keep_files::Bool = false,
    verbose::Bool = false,
) where {W}

    nS = length(S)
    @assert length(x) == nS "asci_pt2_spill: |x|=$(length(x)) ≠ |S|=$nS"
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    eps = env.epsilon

    nt       = Threads.nthreads()
    P_in     = max(64, 10 * nt)
    n_shards = n_shards > 0 ? n_shards : max(4 * nt, nt)

    norm_fac = Vector{Float64}(undef, nS)
    @inbounds for i in 1:nS
        norm_fac[i] = is_T_fixed(S[i], T_plan) ? C.invsqrt2 : 1.0
    end

    isdir(spill_dir) || mkpath(spill_dir)
    paths = [joinpath(spill_dir, "pt2_shard_$(j).bin") for j in 1:n_shards]
    ios   = [open(p, "w") for p in paths]

    acc     = ExpandingAccumulator{W}(P_in, 10 * batch_size)
    buf_tls = [[DrainBuffer{W}(4096) for _ in 1:n_shards] for _ in 1:nt]

    # ---- spill phase --------------------------------------------------------
    n_batches = cld(nS, batch_size)
    for b in 1:n_batches
        i0 = (b - 1) * batch_size + 1
        i1 = min(b * batch_size, nS)
        Sb  = collect(view(S,        i0:i1))
        xb  = collect(view(x,        i0:i1))
        nfb = collect(view(norm_fac, i0:i1))

        if eps2 <= 0.0
            expanding_matvec_coupled!(acc, Sb, xb, nfb, env, T_plan, C, IRREPS)
        else
            asci_expand_screened!(acc, Sb, xb, nfb, env, T_plan, C, IRREPS; cutoff = eps2)
        end
        drain_pass1_ooc!(acc, buf_tls, T_plan, n_shards)     # empties acc, buckets by hash(rep)
        _pt2_spill_write!(ios, buf_tls, Val(W))          # append buckets to files, reset buffers
    end
    for io in ios
        close(io)
    end

    # drain phase
    Sset     = Set(S)
    inv_eps2 = 1.0 / (eps * eps)
    E_parts    = zeros(Float64, n_shards)
    next_parts = zeros(Int, n_shards)
    ngap_parts = zeros(Int, n_shards)
    peak_parts = zeros(Int, n_shards)

    Threads.@threads :static for j in 1:n_shards
        up = Vector{Int}(undef, env.nup)
        dn = Vector{Int}(undef, env.ndn)
        merged = _pt2_spill_read_merge(paths[j], Val(W))
        peak_parts[j] = length(merged)
        e = 0.0; ne = 0; ng = 0
        for (d, val) in merged
            (d in Sset) && continue
            Hii = asci_coupled_hdiag!(up, dn, d, env, T_plan, C)
            gap = E - Hii
            if abs(gap) < gap_floor
                ng += 1
                continue
            end
            e += (val * val) * inv_eps2 / gap
            ne += 1
        end
        E_parts[j]    = e
        next_parts[j] = ne
        ngap_parts[j] = ng
        keep_files || rm(paths[j]; force = true)
    end
    E_pt2 = sum(E_parts)
    n_ext = sum(next_parts)
    n_gap = sum(ngap_parts)
    peak  = maximum(peak_parts)

    if verbose
        @printf("  asci_pt2_spill: |S|=%d batches=%d shards=%d externals=%d gap-skip=%d peak-shard=%d E_PT2=%.10f Ha\n",
                nS, n_batches, n_shards, n_ext, n_gap, peak, E_pt2)
    end
    return (E_pt2 = E_pt2, n_ext = n_ext, n_gap_skipped = n_gap, peak_shard = peak)
end