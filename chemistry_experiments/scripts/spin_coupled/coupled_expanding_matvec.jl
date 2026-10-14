#=------------------------------------------------------------------------------
coupled_expanding_matvec.jl

Expanding (unrestricted) matvec in the spin-coupled basis.

------------------------------------------------------------------------------=#

# -------------------------------------------------------------------------
# Sharded accumulator
# -------------------------------------------------------------------------

"""
Sharded accumulator for expanding matvec.
Uses P shards to reduce lock contention during parallel accumulation.
Compatible with the UpdateBuffer from helper_functions/expanding_matvec.jl.
"""
struct ExpandingAccumulator{W}
    shards::Vector{RobinDict{Det{W},Float64}}
    locks::Vector{Threads.SpinLock}
    P::Int
end

function ExpandingAccumulator{W}(P::Int, expected_size::Int) where {W}
    shards = [RobinDict{Det{W},Float64}() for _ in 1:P]
    sz_per = max(16, expected_size ÷ P)
    for s in shards
        sizehint!(s, sz_per)
    end
    locks = [Threads.SpinLock() for _ in 1:P]
    return ExpandingAccumulator{W}(shards, locks, P)
end

function clear!(acc::ExpandingAccumulator{W}) where {W}
    for s in acc.shards
        empty!(s)
    end
end

# -------------------------------------------------------------------------
# Target norm factor
# -------------------------------------------------------------------------

"""
Compute norm factor for a target determinant (may not be representative).

The norm factor only depends on whether the EQUIVALENCE CLASS is T-fixed,
which is invariant under T, so we can evaluate it on either d or T(d).
"""
@inline function compute_target_norm(d::Det{W}, T_plan::SpinFlipPlan, C::CoupledConsts) where {W}
    td = T_det(d, T_plan)
    if td.words == d.words
        return C.sqrt2  # T-fixed: inloop factor is √2
    else
        return 1.0
    end
end

# -------------------------------------------------------------------------
# Buffered emission helpers (doubles)
# -------------------------------------------------------------------------

"""
    emit_double_contributions_buf!(buf, d, xi, Nd, up_occ, dn_occ,
                                   env, T_plan, C, IRREPS, P,
                                   n, n2, n3, frz, acc_shards, locks)

Emit all double-excitation contributions from determinant `d` through the
thread-local buffer `buf`.  Handles up-up, down-down, and up-down doubles.
"""
function emit_double_contributions_buf!(
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
                    buf_push!(buf, d2, -eps * hij * xi * Nd * Nj,
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
                    buf_push!(buf, d2, -eps * hij * xi * Nd * Nj,
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
                    buf_push!(buf, d2, -eps * hij * xi * Nd * Nj,
                              owner_of(d2, P), acc_shards, locks)
                end
            end
        end
    end

    return nothing
end

# -------------------------------------------------------------------------
# Expanding matvec (buffered, @spawn + chunks)
# -------------------------------------------------------------------------

"""
    expanding_matvec_coupled!(acc, S, x, norm_fac, env, T_plan, C, IRREPS;
                              chunk_size=0, buf_capacity=4096)

Compute y = (I - ε H) * x where x is supported on S, accumulating into the
sharded accumulator `acc`.  Uses thread-local UpdateBuffers to batch
shard updates, eliminating per-emit locking.

Keyword arguments:
  - `chunk_size`: source dets per task. Default: cld(|S|, 8 * nthreads()),
    giving ~8 chunks per thread for load balance.
  - `buf_capacity`: per-task buffer size (entries). Default 4096.
"""
function expanding_matvec_coupled!(
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
            buf    = UpdateBuffer{W}(P; capacity=buf_capacity)

            @inbounds for i in chunk
                xi = x[i]
                d  = S[i]
                Nd = norm_fac[i]

                occupied_orbitals!(up_occ, dn_occ, d, n, env.nup, env.ndn)

                # -------- Diagonal: (shift - ε H_ii) x_i --------
                hii = diagonal_value_from_occ(up_occ, dn_occ,
                                              env.HCORE_DIAG, env.J, env.K,
                                              n, env.nup, env.ndn)
                buf_push!(buf, d, diag_A(env, hii) * xi,
                          owner_of(d, P), acc_shards, acc_locks)

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
                        buf_push!(buf, d2, -eps * hij * xi * Nd * Nj,
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
                        buf_push!(buf, d2, -eps * hij * xi * Nd * Nj,
                                  owner_of(d2, P), acc_shards, acc_locks)
                    end
                end

                # -------- Doubles: up-up, down-down, up-down --------
                emit_double_contributions_buf!(buf, d, xi, Nd, up_occ, dn_occ,
                                               env, T_plan, C, IRREPS, P,
                                               n, n2, n3, frz,
                                               acc_shards, acc_locks)
            end

            # Final flush: drain this task's buffer to the global shards.
            flush!(buf, acc_shards, acc_locks)
        end
    end

    foreach(wait, tasks)
    return nothing
end

# -------------------------------------------------------------------------
# Post-processing: merging shards and selecting top-k
# -------------------------------------------------------------------------

"""
    merge_T_partners!(acc, T_plan) -> Dict{Det{W},Float64}

Merge contributions from d and T(d) into representative keys.
Returns a new dictionary keyed by representatives only.
"""
function merge_T_partners!(acc::ExpandingAccumulator{W}, T_plan::SpinFlipPlan) where {W}
    merged = Dict{Det{W},Float64}()

    total_size = sum(length(s) for s in acc.shards)
    sizehint!(merged, total_size)

    for shard in acc.shards
        for (d, val) in shard
            d_rep = rep_det(d, T_plan)
            merged[d_rep] = get(merged, d_rep, 0.0) + val
        end
    end

    return merged
end

"""
    select_top_k(merged, k) -> (S_new, x_new)

Select the k determinants with largest-magnitude values.
Returns the new support set S and corresponding signed coefficients.
"""
function select_top_k(merged::Dict{Det{W},Float64}, k::Int) where {W}
    n = length(merged)

    if n <= k
        S_new = collect(keys(merged))
        x_new = [merged[d] for d in S_new]
        return S_new, x_new
    end

    entries = [(d, abs(v), v) for (d, v) in merged]
    partialsort!(entries, 1:k, by = e -> -e[2])

    S_new = Vector{Det{W}}(undef, k)
    x_new = Vector{Float64}(undef, k)

    @inbounds for i in 1:k
        S_new[i] = entries[i][1]
        x_new[i] = entries[i][3]
    end

    return S_new, x_new
end

"""
Build a DualIndexedGreedyState from a given S (already representatives).
"""
function build_dual_indexed_state_from_S(
    S::Vector{Det{W}},
    env::QCEnv{W},
    T_plan::SpinFlipPlan,
    C::CoupledConsts,
) where {W}

    nS = length(S)

    det2idx = RobinDict{Det{W},Int}()
    sizehint!(det2idx, nS)
    for i in 1:nS
        det2idx[S[i]] = i
    end

    gs = DualIndexedGreedyState{W}(
        S,
        det2idx,
        RobinDict{Det{W},Int}(),
        Int[],
        Int[],
        Vector{Float64}(undef, nS),
        Vector{Float64}(undef, nS),
        Vector{Float64}(undef, nS),
    )

    build_dual_index!(gs, T_plan)
    build_hdiag_and_norms!(gs, env, T_plan, C)

    return gs
end