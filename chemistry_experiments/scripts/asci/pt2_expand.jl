#=------------------------------------------------------------------------------
pt2_expand.jl -- Screen individual contributions to the PT2 numerator.

used in asci_pt2_ooc_eps2.jl

Include after spin_coupled/coupled_expanding_matvec.jl
------------------------------------------------------------------------------=#

using Printf

function asci_emit_double_screened!(
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
    cutoff::Float64,
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
                    if abs(hij * xi * Nd * Nj) >= cutoff
                        buf_push!(buf, d2, -eps * hij * xi * Nd * Nj,
                                  owner_of(d2, P), acc_shards, locks)
                    end
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
                    if abs(hij * xi * Nd * Nj) >= cutoff
                        buf_push!(buf, d2, -eps * hij * xi * Nd * Nj,
                                  owner_of(d2, P), acc_shards, locks)
                    end
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
                    if abs(hij * xi * Nd * Nj) >= cutoff
                        buf_push!(buf, d2, -eps * hij * xi * Nd * Nj,
                                  owner_of(d2, P), acc_shards, locks)
                    end
                end
            end
        end
    end

    return nothing
end

function asci_expand_screened!(
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
    cutoff::Float64 = 0.0,
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
                        if abs(hij * xi * Nd * Nj) >= cutoff
                            buf_push!(buf, d2, -eps * hij * xi * Nd * Nj,
                                      owner_of(d2, P), acc_shards, acc_locks)
                        end
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
                        if abs(hij * xi * Nd * Nj) >= cutoff
                            buf_push!(buf, d2, -eps * hij * xi * Nd * Nj,
                                      owner_of(d2, P), acc_shards, acc_locks)
                        end
                    end
                end

                # -------- Doubles: up-up, down-down, up-down --------
                asci_emit_double_screened!(buf, d, xi, Nd, up_occ, dn_occ,
                                               env, T_plan, C, IRREPS, P,
                                               n, n2, n3, frz, cutoff,
                                               acc_shards, acc_locks)
            end

            # Final flush: drain this task's buffer to the global shards.
            flush!(buf, acc_shards, acc_locks)
        end
    end

    foreach(wait, tasks)
    return nothing
end