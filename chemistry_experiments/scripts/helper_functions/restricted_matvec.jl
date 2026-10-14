# restricted_matvec.jl
# Matrix-free y = (shift * I - eps*H) x with x,y supported on a FIXED bag S.
# Enumerates single+double neighbors of each row and keeps only those in S
# (Bloom pre-filter + det2idx). mul_A! is the serial reference kept for
# testing; mul_A_threaded! is the parallelized version.

using LinearAlgebra
using DataStructures
using DataStructures: RobinDict
using Base.Threads

# Off-diagonal row contribution:
# y[i] += -eps * sum_{j in S, j!=i} H_ij * x[j]
# This enumerates all single+double neighbors of d and only keeps those in S.
function add_offdiag_row_contrib!(
    x::AbstractVector{Float64},
    gs::GreedyState{W},
    env::QCEnv{W},
    d::Det{W},                         # current determinant (row)
    IRREPS::Vector{Int},                # irreps vector used to build buckets
    up_occ::Vector{Int},                # scratch length nup
    dn_occ::Vector{Int},                 # scratch length ndn (spin-orb indices n+1..2n)
    up_tmp::Vector{Int},                # scratch length nup
    dn_tmp::Vector{Int},                 # scratch length nd (s spin-orb indices n+1..2n)
    bf::BloomFilter
) where {W}

    n   = env.norb_tot
    frz = env.norbfrzn
    Su  = env.single_targets
    Dss = env.doub_same
    Dod = env.doub_diff
    epsilon = env.epsilon

    yi = 0.0

    # fill occupied lists once per parent row
    occupied_orbitals!(up_occ, dn_occ, d, n, env.nup, env.ndn)

    # --------------------------
    # Singles: spin up
    # --------------------------
    @inbounds for t in 1:env.nup
        i = up_occ[t]
        (i <= frz) && continue
        i_so = i
        targets = Su[i]
        for u in 1:length(targets)
            a = targets[u]
            (a <= frz) && continue
            a_so = a
            if !isocc_fast(d, a_so, env.om)
                d2 = flip2(d, i_so, a_so, env.om)
                if bloom_maybe_contains(bf, d2)
                    j = get(gs.det2idx, d2, 0)
                    if j != 0
                        sgn = single_excitation_sign(d, i_so, a_so)
                        hij = single_excitation_value_from_occ(
                            up_occ, dn_occ, i_so, a_so,
                            env.HCORE, env.eris1, n; sign=sgn
                        )
                        yi += (-epsilon) * hij * x[j]
                    end
                end
            end
        end
    end

    # --------------------------
    # Singles: spin-down
    # --------------------------
    @inbounds for t in 1:env.ndn
        i_so = dn_occ[t]
        i = i_so - n
        (i <= frz) && continue
        targets = Su[i]
        for u in 1:length(targets)
            a = targets[u]
            (a <= frz) && continue
            a_so = a + n
            if !isocc_fast(d, a_so, env.om)
                d2 = flip2(d, i_so, a_so, env.om)
                if bloom_maybe_contains(bf, d2)
                    j = get(gs.det2idx, d2, 0)
                    if j != 0
                        sgn = single_excitation_sign(d, i_so, a_so)
                        hij = single_excitation_value_from_occ(
                            up_occ, dn_occ, i_so, a_so,
                            env.HCORE, env.eris1, n; sign=sgn
                        )
                        yi += (-epsilon) * hij * x[j]
                    end
                end
            end
        end
    end

    # --------------------------
    # Doubles: up-up
    # --------------------------
    @inbounds for p in 1:(env.nup - 1)
        i = up_occ[p]
        (i <= frz) && continue
        i_so = i
        iri  = IRREPS[i]
        for q in (p + 1):env.nup
            j = up_occ[q]
            (j <= frz) && continue
            j_so = j
            key  = iri ⊻ IRREPS[j]
            cand = Dss[key + 1]      # (a,b) spatial a<b
            for u in 1:length(cand)
                a, b = cand[u]
                (a <= frz || b <= frz) && continue
                a_so = a
                b_so = b
                if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                    d2 = flip4(d, i_so, j_so, a_so, b_so, env.om)
                    if bloom_maybe_contains(bf, d2)
                        jj = get(gs.det2idx, d2, 0)
                        if jj != 0
                            sgn = double_excitation_sign_via_two_singles(d, i_so, j_so, a_so, b_so, env.om)
                            hij = double_excitation_value(i_so, j_so, a_so, b_so, env.eris1, n; sign=sgn)
                            yi += (-epsilon) * hij * x[jj]
                        end
                    end
                end
            end
        end
    end

    # --------------------------
    # Doubles: down-down
    # --------------------------
    @inbounds for p in 1:(env.ndn - 1)
        i_so = dn_occ[p]
        i    = i_so - n
        (i <= frz) && continue
        iri  = IRREPS[i]

        for q in (p + 1):env.ndn
            j_so = dn_occ[q]
            j    = j_so - n
            (j <= frz) && continue

            key  = iri ⊻ IRREPS[j]
            cand = Dss[key + 1]      # (a,b) spatial a<b

            for u in 1:length(cand)
                a, b = cand[u]
                (a <= frz || b <= frz) && continue
                a_so = a + n
                b_so = b + n

                if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                    d2 = flip4(d, i_so, j_so, a_so, b_so, env.om)
                    if bloom_maybe_contains(bf, d2)
                        jj = get(gs.det2idx, d2, 0)
                        if jj != 0
                            sgn = double_excitation_sign_via_two_singles(d, i_so, j_so, a_so, b_so, env.om)
                            hij = double_excitation_value(i_so, j_so, a_so, b_so, env.eris1, n; sign=sgn)
                            yi += (-epsilon) * hij * x[jj]
                        end
                    end
                end
            end
        end
    end

    # --------------------------
    # Doubles: up-down
    # interpret cand (a,b) as: a is up target, b is down target
    # --------------------------
    @inbounds for p in 1:env.nup
        i = up_occ[p]
        (i <= frz) && continue
        i_so = i
        iri  = IRREPS[i]
        for q in 1:env.ndn
            j_so = dn_occ[q]
            j    = j_so - n
            (j <= frz) && continue
            key  = iri ⊻ IRREPS[j]
            cand = Dod[key + 1]      # (a,b) spatial (a up, b down), may include a==b
            for u in 1:length(cand)
                a, b = cand[u]
                (a <= frz || b <= frz) && continue
                a_so = a
                b_so = b + n
                if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                    d2 = flip4(d, i_so, j_so, a_so, b_so, env.om)
                    if bloom_maybe_contains(bf, d2)
                        jj = get(gs.det2idx, d2, 0)
                        if jj != 0
                            sgn = double_excitation_sign_via_two_singles(d, i_so, j_so, a_so, b_so, env.om)
                            hij = double_excitation_value(i_so, j_so, a_so, b_so, env.eris1, n; sign=sgn)
                            yi += (-epsilon) * hij * x[jj]
                        end
                    end
                end
            end
        end
    end
    return yi
end

function mul_A!(
    y::AbstractVector{Float64},
    x::AbstractVector{Float64},
    gs::GreedyState{W},
    env::QCEnv{W},
    IRREPS::Vector{Int},
    up_occ::Vector{Int},
    dn_occ::Vector{Int},
    up_tmp::Vector{Int},
    dn_tmp::Vector{Int},
    bf::BloomFilter
) where {W}

    nS = length(gs.S)
    @boundscheck begin
        length(y) == nS || throw(DimensionMismatch("length(y)=$(length(y)) ≠ |S|=$nS"))
        length(x) == nS || throw(DimensionMismatch("length(x)=$(length(x)) ≠ |S|=$nS"))
    end

    @inbounds for i in 1:nS
        d = gs.S[i]

        # diagonal term
        y[i] = diag_A(env, gs.hdiag[i]) * x[i]

        # off-diagonal term
        y[i] += add_offdiag_row_contrib!(x, gs, env, d, IRREPS, up_occ, dn_occ, up_tmp, dn_tmp, bf)
    end

    return y
end

function mul_A_threaded!(
    y::AbstractVector{Float64},
    x::AbstractVector{Float64},
    gs::GreedyState{W},
    env::QCEnv{W},
    IRREPS::Vector{Int},
    scratch::Vector{MatvecScratch},
    bf::BloomFilter
) where {W}

    nS = length(gs.S)
    @boundscheck begin
        length(y) == nS || throw(DimensionMismatch("length(y)=$(length(y)) ≠ |S|=$nS"))
        length(x) == nS || throw(DimensionMismatch("length(x)=$(length(x)) ≠ |S|=$nS"))
        length(scratch) >= Threads.maxthreadid() || throw(DimensionMismatch(
            "scratch length=$(length(scratch)) < maxthreadid()=$(Threads.maxthreadid())"
        ))
    end

    Threads.@threads :static for i in 1:nS
        tid = Threads.threadid()
        s = @inbounds scratch[tid]

        up_occ = getfield(s, :up_occ)
        dn_occ = getfield(s, :dn_occ)
        up_tmp = getfield(s, :up_tmp)
        dn_tmp = getfield(s, :dn_tmp)

        @inbounds begin
            d  = gs.S[i]
            yi = diag_A(env, gs.hdiag[i]) * x[i]
            yi += add_offdiag_row_contrib!(x, gs, env, d, IRREPS,
                                           up_occ, dn_occ, up_tmp, dn_tmp, bf)
            y[i] = yi
        end
    end

    return y
end
