#=------------------------------------------------------------------------------
sparse_hamiltonian_csc.jl -- Sparse assembly of A[S,S] = shift*I - epsilon*H[S,S].

build_sparse_A_csc constructs the restricted operator in the
spin-flip-symmetrized basis as a compressed sparse column (CSC) matrix.
ASCI reuses this matrix throughout each eigensolve; it must fit in RAM.
------------------------------------------------------------------------------=#

using SparseArrays
using Base.Threads

# Row-index / colptr element type for the CSC. 
const CSCIndex = Int

# Contiguous split of 1:nS into nt ranges 
function _col_chunks(nS::Int, nt::Int)
    chunks = Vector{UnitRange{Int}}(undef, nt)
    base = nS ÷ nt
    rem  = nS % nt
    start = 1
    @inbounds for c in 1:nt
        len = base + (c <= rem ? 1 : 0)
        chunks[c] = start:(start + len - 1)
        start += len
    end
    return chunks
end

# Per-thread column accumulator
mutable struct ColAccumulator
    acc_val::Vector{Float64}
    seen::Vector{Bool}
    touched::Vector{Int}
end
ColAccumulator(nS::Int) = ColAccumulator(zeros(Float64, nS), falses(nS), Int[])

@inline function _accum!(acc::ColAccumulator, j::Int, v::Float64)
    @inbounds if !acc.seen[j]
        acc.seen[j] = true
        push!(acc.touched, j)
    end
    @inbounds acc.acc_val[j] += v
    return nothing
end

@inline function _reset!(acc::ColAccumulator)
    @inbounds for j in acc.touched
        acc.acc_val[j] = 0.0
        acc.seen[j] = false
    end
    empty!(acc.touched)
    return nothing
end


function accumulate_column!(
    acc::ColAccumulator,
    gs::DualIndexedGreedyState{W},
    env::QCEnv{W},
    d::Det{W},
    col::Int,
    IRREPS::Vector{Int},
    up_occ::Vector{Int},
    dn_occ::Vector{Int},
    bf::BloomFilter,
) where {W}

    n   = env.norb_tot
    n2  = n * n
    n3  = n2 * n
    frz = env.norbfrzn
    Su  = env.single_targets
    Dss = env.doub_same
    Dod = env.doub_diff
    epsilon = env.epsilon

    Nd = gs.norm_fac[col]

    # diagonal A[col,col] = shift - eps*H_cc
    _accum!(acc, col, diag_A(env, gs.hdiag[col]))

    occupied_orbitals!(up_occ, dn_occ, d, n, env.nup, env.ndn)

    # --------------------------
    # Singles: spin up
    # --------------------------
    @inbounds for t in 1:env.nup
        i_so = up_occ[t]
        i = i_so
        (i <= frz) && continue
        targets = Su[i]
        for u in 1:length(targets)
            a = @inbounds targets[u]
            (a <= frz) && continue
            a_so = a
            if !isocc_fast(d, a_so, env.om)
                d2 = flip2(d, i_so, a_so, env.om)
                if bloom_maybe_contains(bf, d2)
                    jdx = dual_lookup(d2, gs)
                    if jdx != 0
                        sgn = single_excitation_sign(d, i_so, a_so)
                        hidxjdx = single_excitation_value_from_occ(
                            up_occ, dn_occ, i_so, a_so,
                            env.HCORE, env.eris1, n; n2=n2, n3=n3, sign=sgn
                        )
                        Nj = gs.inloop_norm[jdx]
                        _accum!(acc, jdx, (-epsilon) * hidxjdx * Nd * Nj)
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
            a = @inbounds targets[u]
            (a <= frz) && continue
            a_so = a + n
            if !isocc_fast(d, a_so, env.om)
                d2 = flip2(d, i_so, a_so, env.om)
                if bloom_maybe_contains(bf, d2)
                    jdx = dual_lookup(d2, gs)
                    if jdx != 0
                        sgn = single_excitation_sign(d, i_so, a_so)
                        hidxjdx = single_excitation_value_from_occ(
                            up_occ, dn_occ, i_so, a_so,
                            env.HCORE, env.eris1, n; n2=n2, n3=n3, sign=sgn
                        )
                        Nj = gs.inloop_norm[jdx]
                        _accum!(acc, jdx, (-epsilon) * hidxjdx * Nd * Nj)
                    end
                end
            end
        end
    end

    # --------------------------
    # Doubles: up-up
    # --------------------------
    @inbounds for p in 1:(env.nup - 1)
        i_so = up_occ[p]
        i = i_so
        (i <= frz) && continue
        iri  = IRREPS[i]
        for q in (p + 1):env.nup
            j_so = up_occ[q]
            j = j_so
            (j <= frz) && continue
            key  = iri ⊻ IRREPS[j]
            cand = Dss[key + 1]
            for u in 1:length(cand)
                a, b = @inbounds cand[u]
                (a <= frz || b <= frz) && continue
                a_so = a
                b_so = b
                if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                    d2 = flip4(d, i_so, j_so, a_so, b_so, env.om)
                    if bloom_maybe_contains(bf, d2)
                        jdx = dual_lookup(d2, gs)
                        if jdx != 0
                            sgn = double_excitation_sign_via_two_singles(d, i_so, j_so, a_so, b_so, env.om)
                            hidxjdx = double_excitation_value(i_so, j_so, a_so, b_so, env.eris1, n; n2=n2, n3=n3, sign=sgn)
                            Nj = gs.inloop_norm[jdx]
                            _accum!(acc, jdx, (-epsilon) * hidxjdx * Nd * Nj)
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
            cand = Dss[key + 1]
            for u in 1:length(cand)
                a, b = @inbounds cand[u]
                (a <= frz || b <= frz) && continue
                a_so = a + n
                b_so = b + n
                if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                    d2 = flip4(d, i_so, j_so, a_so, b_so, env.om)
                    if bloom_maybe_contains(bf, d2)
                        jdx = dual_lookup(d2, gs)
                        if jdx != 0
                            sgn = double_excitation_sign_via_two_singles(d, i_so, j_so, a_so, b_so, env.om)
                            hidxjdx = double_excitation_value(i_so, j_so, a_so, b_so, env.eris1, n; n2=n2, n3=n3, sign=sgn)
                            Nj = gs.inloop_norm[jdx]
                            _accum!(acc, jdx, (-epsilon) * hidxjdx * Nd * Nj)
                        end
                    end
                end
            end
        end
    end

    # --------------------------
    # Doubles: up-down  (a is up target, b is down target)
    # --------------------------
    @inbounds for p in 1:env.nup
        i_so = up_occ[p]
        i = i_so
        (i <= frz) && continue
        iri  = IRREPS[i]
        for q in 1:env.ndn
            j_so = dn_occ[q]
            j    = j_so - n
            (j <= frz) && continue
            key  = iri ⊻ IRREPS[j]
            cand = Dod[key + 1]
            for u in 1:length(cand)
                a, b = @inbounds cand[u]
                (a <= frz || b <= frz) && continue
                a_so = a
                b_so = b + n
                if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                    d2 = flip4(d, i_so, j_so, a_so, b_so, env.om)
                    if bloom_maybe_contains(bf, d2)
                        jdx = dual_lookup(d2, gs)
                        if jdx != 0
                            sgn = double_excitation_sign_via_two_singles(d, i_so, j_so, a_so, b_so, env.om)
                            hidxjdx = double_excitation_value(i_so, j_so, a_so, b_so, env.eris1, n; n2=n2, n3=n3, sign=sgn)
                            Nj = gs.inloop_norm[jdx]
                            _accum!(acc, jdx, (-epsilon) * hidxjdx * Nd * Nj)
                        end
                    end
                end
            end
        end
    end

    return nothing
end

"""
    build_sparse_A_csc(gs, env, IRREPS, bf) -> SparseMatrixCSC{Float64,CSCIndex}

Two-pass CSC assembly of A = I - eps*H restricted to gs.S. Threaded by disjoint column ranges
"""
function build_sparse_A_csc(
    gs::DualIndexedGreedyState{W},
    env::QCEnv{W},
    IRREPS::Vector{Int},
    bf::BloomFilter,
) where {W}

    nS = length(gs.S)
    nt = max(1, Threads.nthreads())
    chunks = _col_chunks(nS, nt)

    colcount = Vector{Int}(undef, nS)

    # ---------- Pass 1: count distinct nonzeros per column ----------
    @sync for c in 1:nt
        Threads.@spawn begin
            acc = ColAccumulator(nS)
            up_occ = Vector{Int}(undef, env.nup)
            dn_occ = Vector{Int}(undef, env.ndn)
            @inbounds for col in chunks[c]
                accumulate_column!(acc, gs, env, gs.S[col], col, IRREPS, up_occ, dn_occ, bf)
                colcount[col] = length(acc.touched)
                _reset!(acc)
            end
        end
    end

    # ---------- colptr by prefix sum ----------
    colptr = Vector{Int}(undef, nS + 1)
    colptr[1] = 1
    @inbounds for j in 1:nS
        colptr[j + 1] = colptr[j] + colcount[j]
    end
    nnz_total = colptr[nS + 1] - 1

    rowval = Vector{CSCIndex}(undef, nnz_total)
    nzval  = Vector{Float64}(undef, nnz_total)

    # ---------- Pass 2: fill each column's block directly ----------
    @sync for c in 1:nt
        Threads.@spawn begin
            acc = ColAccumulator(nS)
            up_occ = Vector{Int}(undef, env.nup)
            dn_occ = Vector{Int}(undef, env.ndn)
            @inbounds for col in chunks[c]
                accumulate_column!(acc, gs, env, gs.S[col], col, IRREPS, up_occ, dn_occ, bf)
                sort!(acc.touched)                       # canonical CSC: sorted row indices
                pos = colptr[col]
                for r in acc.touched
                    rowval[pos] = CSCIndex(r)
                    nzval[pos]  = acc.acc_val[r]
                    pos += 1
                end
                _reset!(acc)
            end
        end
    end

    return SparseMatrixCSC{Float64,CSCIndex}(nS, nS, colptr, rowval, nzval)
end