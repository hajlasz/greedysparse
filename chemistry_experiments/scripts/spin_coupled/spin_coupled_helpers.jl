using Arpack
using Combinatorics
using DelimitedFiles
using LinearAlgebra
using Random
using SparseArrays
using StatsBase
using JSON
using DataStructures
using Base.Threads

# -------------------------------------------------------------------------
# Spin-flip / time reversal for Det{W}: swap the up/down halves (1..n) <-> (n+1..2n)
# T denotes the time-reversal operator on determinants.
# -------------------------------------------------------------------------

struct SpinFlipPlan
    n::Int           # spatial orbitals
    W::Int           # number of UInt64 words in Det{W}
    wsh::Int         # n >>> 6
    bsh::Int         # n & 63
    lastmask::UInt64 # mask for last word to keep only valid (2n) bits
end

struct CoupledConsts
    sqrt2::Float64
    invsqrt2::Float64
end

@inline CoupledConsts() = CoupledConsts(sqrt(2.0), 1/sqrt(2.0))

@inline norm_factor(d::Det{W}, plan::SpinFlipPlan, C::CoupledConsts) where {W} =
    ifelse(is_T_fixed(d, plan), 1/C.sqrt2, 1.0)

"""
Compute norm factor for a target determinant (may not be representative).
"""
@inline function compute_target_norm(d::Det{W}, T_plan::SpinFlipPlan, C::CoupledConsts) where {W}
    td = T_det(d, T_plan)
    if td.words == d.words
        return C.sqrt2  # T-fixed: inloop factor is √2
    else
        return 1.0
    end
end

# for states generated in loop, additional factor of two if |j> = T|j> because
# those states only get generated once but contribute twice in the sum over j 
@inline inloop_norm_factor(d::Det{W}, plan::SpinFlipPlan, C::CoupledConsts) where {W} =
    ifelse(is_T_fixed(d, plan), 2/C.sqrt2, 1.0)

@inline function _mask_lastword(n::Int)::UInt64
    nso = 2n
    r = nso & 0x3f                      # mod 64
    return r == 0 ? typemax(UInt64) : (UInt64(1) << r) - 1
end

@inline SpinFlipPlan(n::Int, ::Val{W}) where {W} =
    SpinFlipPlan(n, W, n >>> 6, n & 0x3f, _mask_lastword(n))

# Compare packed bitstrings "as integers": most-significant word first
@inline function _words_lt(a::NTuple{W,UInt64}, b::NTuple{W,UInt64})::Bool where {W}
    @inbounds for k in W:-1:1
        ak = a[k]; bk = b[k]
        ak == bk && continue
        return ak < bk
    end
    return false
end

# Core: compute T(d).words without allocating anything except the returned tuple.
@inline function T_words(words::NTuple{W,UInt64}, plan::SpinFlipPlan)::NTuple{W,UInt64} where {W}
    wsh = plan.wsh
    bsh = plan.bsh
    lastmask = plan.lastmask

    # rotate: (words << n) | (words >> n) within 2n bits
    return ntuple(k -> begin
        # left shift part into word k
        vL = UInt64(0)
        kk = k - wsh
        if kk > 0
            if bsh == 0
                @inbounds vL = words[kk]
            else
                @inbounds vL = words[kk] << bsh
                if kk > 1
                    @inbounds vL |= words[kk-1] >>> (64 - bsh)
                end
            end
        end

        # right shift part into word k
        vR = UInt64(0)
        kk2 = k + wsh
        if kk2 <= W
            if bsh == 0
                @inbounds vR = words[kk2]
            else
                @inbounds vR = words[kk2] >>> bsh
                if kk2 < W
                    @inbounds vR |= words[kk2+1] << (64 - bsh)
                end
            end
        end

        x = vL | vR
        (k == W) ? (x & lastmask) : x
    end, W)
end

@inline function T_det(d::Det{W}, plan::SpinFlipPlan)::Det{W} where {W}
    return Det{W}(T_words(d.words, plan))
end

# Representative: binary minimum of d and T(d)
# Hot-loop friendly: only constructs T(d) if needed, and returns d unchanged if it wins.
@inline function rep_det(d::Det{W}, plan::SpinFlipPlan)::Det{W} where {W}
    td_words = T_words(d.words, plan)
    return _words_lt(td_words, d.words) ? Det{W}(td_words) : d
end

# fast check for fixed point (T(d) == d) without constructing a Det
@inline function is_T_fixed(d::Det{W}, plan::SpinFlipPlan)::Bool where {W}
    td_words = T_words(d.words, plan)
    return td_words == d.words
end



# ------------------------------------------------------------------------------
# Excitation Helpers with Spin Coupling
# ------------------------------------------------------------------------------

# check if a T|i> and |i> differ by a double excitation
# if so, return (true, p, q, sgn) where p->q are spatial orbitals involved
# and sgn is the fermionic sign
# else return (false, 0, 0, 0)
@inline function spinflip_double_params_from_occ(
    up_occ::Vector{Int},      # spatial 1..n, sorted
    dn_occ::Vector{Int},      # spin-orbital n+1..2n, sorted
    n::Int, nup::Int, ndn::Int
)::Tuple{Bool,Int,Int,Int}

    # This is written for nup = ndn systems 
    nup == ndn || return (false, 0, 0, 0)

    # Two-pointer diff between up_occ (spatial) and dn_occ mapped to spatial by -n.
    a = 1
    b = 1
    p = 0  # in alpha not beta
    q = 0  # in beta not alpha
    ndiff_a = 0
    ndiff_b = 0

    @inbounds while a <= nup && b <= ndn
        ua = up_occ[a]
        db = dn_occ[b] - n  # spatial

        if ua == db
            a += 1; b += 1
        elseif ua < db
            ndiff_a += 1
            ndiff_a > 1 && return (false, 0, 0, 0)
            p = ua
            a += 1
        else
            ndiff_b += 1
            ndiff_b > 1 && return (false, 0, 0, 0)
            q = db
            b += 1
        end
    end

    @inbounds while a <= nup
        ndiff_a += 1
        ndiff_a > 1 && return (false, 0, 0, 0)
        p = up_occ[a]
        a += 1
    end

    @inbounds while b <= ndn
        ndiff_b += 1
        ndiff_b > 1 && return (false, 0, 0, 0)
        q = dn_occ[b] - n
        b += 1
    end

    (ndiff_a == 1 && ndiff_b == 1) || return (false, 0, 0, 0)

    # Fermionic sign for the opposite-spin double:
    # alpha single: p -> q within up_occ (spatial)
    lo = (p < q) ? p : q
    hi = (p < q) ? q : p
    # count occupied alpha strictly between lo and hi
    i_lo = searchsortedfirst(up_occ, lo + 1)
    i_hi = searchsortedfirst(up_occ, hi)
    cntα = i_hi - i_lo

    # beta single: (q+n) -> (p+n) within dn_occ (spin-orbital indices)
    qp = q + n
    pp = p + n
    lo2 = (qp < pp) ? qp : pp
    hi2 = (qp < pp) ? pp : qp
    j_lo = searchsortedfirst(dn_occ, lo2 + 1)
    j_hi = searchsortedfirst(dn_occ, hi2)
    cntβ = j_hi - j_lo

    sgn = isodd(cntα + cntβ) ? -1 : 1
    return (true, p, q, sgn)
end

# check if |i> and |j> differ by a single excitation
# Return (ok, i_so, a_so, sgn) for a *single* excitation taking d1 -> d2.
@inline function single_excitation_params_from_dets(
    d1::Det{W},
    d2::Det{W},
    om::OrbMasks,
    n::Int
)::Tuple{Bool,Int,Int,Int} where {W}

    # Find the two differing spin-orbitals via XOR
    p = 0
    q = 0
    ndiff = 0

    @inbounds for w in 1:W
        x = d1.words[w] ⊻ d2.words[w]
        while x != 0
            b = trailing_zeros(x)                  # 0..63
            so = ((w - 1) << 6) + b + 1           # 1-based spin-orb index
            ndiff += 1
            if ndiff == 1
                p = so
            elseif ndiff == 2
                q = so
            else
                return (false, 0, 0, 0)           # more than 2 bits differ
            end
            x &= x - 1
        end
    end

    ndiff == 2 || return (false, 0, 0, 0)

    # Must be same-spin (both <= n or both > n)
    (p <= n) == (q <= n) || return (false, 0, 0, 0)

    # Identify which is removed (occupied in d1) vs added (unoccupied in d1)
    p_occ = @inbounds (d1.words[om.word[p]] & om.mask[p]) != 0
    q_occ = @inbounds (d1.words[om.word[q]] & om.mask[q]) != 0

    # For a valid single excitation: exactly one of {p,q} occupied in d1
    (p_occ ⊻ q_occ) || return (false, 0, 0, 0)

    i_so = p_occ ? p : q
    a_so = p_occ ? q : p

    sgn = single_excitation_sign(d1, i_so, a_so)  # from excitation_helpers.jl :contentReference[oaicite:0]{index=0}
    return (true, i_so, a_so, sgn)
end


"""
    DualIndexedGreedyState{W}

Extended greedy state with dual indexing for O(1) lookups without rep_det.

Fields:
- `S`: Vector of representative determinants
- `det2idx`: Maps rep(d) → index in S
- `Tdet2idx`: Maps T(rep(d)) → same index in S (for non-T-fixed points)
- `frontier`, `next_frontier`: For bag expansion in unsparsified case 
- `hdiag`: Diagonal Hamiltonian elements ⟨d|H|d⟩
- `norm_fac`: Precomputed N_d = 1/√2 if T-fixed, else 1
- `inloop_norm`: Precomputed 2/√2 if T-fixed, else 1 (for neighbor contributions)

inloop norm is used when d is generated in the matvec loop, to account for
the fact that T-fixed determinants only appear once but contribute twice.
"""
mutable struct DualIndexedGreedyState{W}
    S::Vector{Det{W}}                    
    det2idx::RobinDict{Det{W},Int}
    Tdet2idx::RobinDict{Det{W},Int}      
    frontier::Vector{Int}
    next_frontier::Vector{Int}
    hdiag::Vector{Float64}
    norm_fac::Vector{Float64}             
    inloop_norm::Vector{Float64}          
end

# Constructor that initializes with a single determinant (like HF)
function DualIndexedGreedyState{W}(d_init::Det{W}) where {W}
    S = Det{W}[d_init]
    det2idx = RobinDict{Det{W},Int}(d_init => 1)
    Tdet2idx = RobinDict{Det{W},Int}()
    frontier = Int[1]
    next_frontier = Int[]
    hdiag = Vector{Float64}(undef, 1)
    norm_fac = Vector{Float64}(undef, 1)
    inloop_norm = Vector{Float64}(undef, 1)
    
    return DualIndexedGreedyState{W}(
        S, det2idx, Tdet2idx, frontier, next_frontier,
        hdiag, norm_fac, inloop_norm
    )
end

"""
Insert a determinant into the dual-indexed state during bag expansion.
Assumes d is already a representative (call rep_det before this).
"""
@inline function insert_det!(gs::DualIndexedGreedyState{W}, d::Det{W}) where {W}
    haskey(gs.det2idx, d) && return false  # Already present
    
    push!(gs.S, d)
    idx = length(gs.S)
    gs.det2idx[d] = idx
    push!(gs.next_frontier, idx)
    
    return true  # Newly inserted
end

"""
    build_dual_index!(gs::DualIndexedGreedyState{W}, T_plan::SpinFlipPlan)

Populate Tdet2idx so that both d and T(d) map to the same index.
Call this AFTER bag expansion is complete, BEFORE building Bloom filter.
"""
function build_dual_index!(gs::DualIndexedGreedyState{W}, T_plan::SpinFlipPlan) where {W}
    nS = length(gs.S)
    empty!(gs.Tdet2idx)
    sizehint!(gs.Tdet2idx, nS)
    
    # Phase 1: Parallel compute which indices are non-T-fixed
    is_non_fixed = Vector{Bool}(undef, nS)
    T_dets = Vector{Det{W}}(undef, nS)
    
    Threads.@threads for i in 1:nS
        @inbounds begin
            d = gs.S[i]
            td = T_det(d, T_plan)
            T_dets[i] = td
            is_non_fixed[i] = (td.words != d.words)
        end
    end
    
    # Phase 2: Serial insertion (hash tables aren't thread-safe for insertion)
    @inbounds for i in 1:nS
        if is_non_fixed[i]
            gs.Tdet2idx[T_dets[i]] = i
        end
    end
    
    return nothing
end

"""
    build_hdiag_and_norms!(gs::DualIndexedGreedyState{W}, env::QCEnv{W}, 
                           T_plan::SpinFlipPlan, C::CoupledConsts)

Build diagonal elements (Slater basis, NOT spin-coupled) and precompute norm factors.

The spin-coupling contribution ⟨d|H|T(d)⟩ is NOT included in the diagonal because
it will be picked up during the off-diagonal neighbor enumeration when T(d) is 
generated as a double-excitation neighbor (and dual_lookup maps it back to the same index).
"""
function build_hdiag_and_norms!(
    gs::DualIndexedGreedyState{W},
    env::QCEnv{W},
    T_plan::SpinFlipPlan,
    C::CoupledConsts  
) where {W}
    
    nS = length(gs.S)
    resize!(gs.hdiag, nS)
    resize!(gs.norm_fac, nS)
    resize!(gs.inloop_norm, nS)
    
    n = env.norb_tot
    
    Threads.@threads for i in 1:nS
        # Task-local scratch arrays
        up_occ = Vector{Int}(undef, env.nup)
        dn_occ = Vector{Int}(undef, env.ndn)
        
        @inbounds begin
            d = gs.S[i]
            
            td_words = T_words(d.words, T_plan)
            is_fixed = (td_words == d.words)
            
            # determinant d normalization factors
            # norm_fac = 1/√2 if T-fixed, else 1
            # inloop_norm = √2 if T-fixed, else 1
            # inloop norm factor is doubled bc d and T(d) both contribute but only one is generated
            gs.norm_fac[i] = is_fixed ? C.invsqrt2 : 1.0
            gs.inloop_norm[i] = is_fixed ? C.sqrt2 : 1.0
            
            occupied_orbitals!(up_occ, dn_occ, d, n, env.nup, env.ndn)
            
            # <d|H|d> without spin-coupling term
            gs.hdiag[i] = diagonal_value_from_occ(up_occ, dn_occ, env.HCORE_DIAG, env.J, 
                                env.K, n, env.nup, env.ndn)
        end
    end
    
    return nothing
end


"""
    build_dual_bloom(gs::DualIndexedGreedyState{W}, T_plan::SpinFlipPlan)

Build a Bloom filter containing both d and T(d) for all d ∈ S.
"""
function build_dual_bloom(gs::DualIndexedGreedyState{W}, T_plan::SpinFlipPlan) where {W}
    n_items = 2 * length(gs.S)
    bf = BloomFilter(n_items; bits_per_item=12, k=3)
    
    for i in eachindex(gs.S)
        @inbounds begin
            d = gs.S[i]
            bloom_add!(bf, d)
            
            td = T_det(d, T_plan)
            if td.words != d.words
                bloom_add!(bf, td)
            end
        end
    end
    
    return bf
end

"""
    dual_lookup(d::Det{W}, gs::DualIndexedGreedyState{W})::Int

Look up a determinant in the dual index. Returns 0 if not found.
"""
@inline function dual_lookup(d::Det{W}, gs::DualIndexedGreedyState{W})::Int where {W}
    # Try primary index (d is a representative)
    j = get(gs.det2idx, d, 0)
    j != 0 && return j
    
    # Try T-partner index (d = T(rep) for some rep in S)
    return get(gs.Tdet2idx, d, 0)
end

"""
    dual_indexed_offdiag_contrib!(...)

Compute off-diagonal contribution to matvec for row i
in spin-coupled basis 
"""
function dual_indexed_offdiag_contrib!(
    x::AbstractVector{Float64},
    gs::DualIndexedGreedyState{W},
    env::QCEnv{W},
    d::Det{W},
    row_idx::Int,
    IRREPS::Vector{Int},
    up_occ::Vector{Int},
    dn_occ::Vector{Int},
    bf::BloomFilter
) where {W}

    n   = env.norb_tot
    n2  = n * n
    n3  = n2 * n
    frz = env.norbfrzn
    Su  = env.single_targets
    Dss = env.doub_same
    Dod = env.doub_diff
    epsilon = env.epsilon

    # Precomputed norm factor for this row
    Nd = gs.norm_fac[row_idx]
    
    yi = 0.0

    # Fill occupied orbitals once per row
    occupied_orbitals!(up_occ, dn_occ, d, n, env.nup, env.ndn)

    # --------------------------
    # Singles: spin up
    # --------------------------
    @inbounds for t in 1:env.nup
        i_so = up_occ[t]
        i = i_so
        (i <= frz) && continue
        targets = Su[i] # allowed single excitations from i, spatial orbitals
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
                        yi += (-epsilon) * hidxjdx * x[jdx] * Nd * Nj
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
        targets = Su[i] # allowed single excitations from i, spatial orbitals
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
                        yi += (-epsilon) * hidxjdx * x[jdx] * Nd * Nj
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
            cand = Dss[key + 1] # allowed double excitations, spatial orbitals, a<b
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
                            yi += (-epsilon) * hidxjdx * x[jdx] * Nd * Nj
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
            cand = Dss[key + 1] # allowed double excitations, spatial orbitals, a<b
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
                            yi += (-epsilon) * hidxjdx * x[jdx] * Nd * Nj
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
        i_so = up_occ[p]
        i = i_so
        (i <= frz) && continue
        iri  = IRREPS[i]
        for q in 1:env.ndn
            j_so = dn_occ[q]
            j    = j_so - n
            (j <= frz) && continue
            key  = iri ⊻ IRREPS[j]
            cand = Dod[key + 1] # allowed double excitations, spatial orbitals, (a up, b down), may include a==b
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
                            yi += (-epsilon) * hidxjdx * x[jdx] * Nd * Nj
                        end
                    end
                end
            end
        end
    end
    return yi
end

"""
    dual_indexed_mul_A_threaded!(y, x, gs, env, IRREPS, scratch, bf)

Threaded matrix-vector multiplication using dual indexing.
Computes y = (I - ε H) x restricted to the bag S.
"""
function dual_indexed_mul_A_threaded!(
    y::AbstractVector{Float64},
    x::AbstractVector{Float64},
    gs::DualIndexedGreedyState{W},
    env::QCEnv{W},
    IRREPS::Vector{Int},
    scratch::Vector{MatvecScratch},
    bf::BloomFilter
) where {W}

    nS = length(gs.S)
    
    @boundscheck begin
        length(y) == nS || throw(DimensionMismatch("length(y)=$(length(y)) ≠ |S|=$nS"))
        length(x) == nS || throw(DimensionMismatch("length(x)=$(length(x)) ≠ |S|=$nS"))
    end

    Threads.@threads for i in 1:nS
        tid = Threads.threadid()
        s = @inbounds scratch[tid]

        up_occ = getfield(s, :up_occ)
        dn_occ = getfield(s, :dn_occ)

        @inbounds begin
            d = gs.S[i]
            
            # Diagonal contribution: (shift - ε H_ii) x_i
            yi = diag_A(env, gs.hdiag[i]) * x[i]
            
            # Off-diagonal contribution (using dual indexing)
            yi += dual_indexed_offdiag_contrib!(
                x, gs, env, d, i, IRREPS, up_occ, dn_occ, bf
            )
            
            y[i] = yi
        end
    end
    
    return y
end


# ------------------------------------------------------------------------------
# Unsparsified Bag Expansion with Spin Coupling
# ------------------------------------------------------------------------------
"""
    coupled_expand_phase1!(gs, env, IRREPS, norbfrzn, max_dets,
                           up_occ, dn_occ, T_plan)
Expand the bag in the unsparsified greedy algorithm in spin-coupled basis.
"""
function coupled_expand_phase1!(
    gs::DualIndexedGreedyState{W},
    env::QCEnv{W},
    IRREPS::Vector{Int},      # must match the irreps used to build doub_* buckets
    norbfrzn::Int,            # frozen spatial orbitals are 1:norbfrzn
    max_dets::Int,
    up_occ::Vector{Int},      # scratch (length nup), spatial indices 1..n
    dn_occ::Vector{Int},       # scratch (length ndn), spin-orb indices (n+1..2n)
    T_plan::SpinFlipPlan
) where {W}

    empty!(gs.next_frontier)

    n   = env.norb_tot
    frz = norbfrzn

    Su  = env.single_targets
    Dss = env.doub_same
    Dod = env.doub_diff

    @inbounds for idx in gs.frontier
        d = gs.S[idx]

        # fill occ lists once per parent
        occupied_orbitals!(up_occ, dn_occ, d, n, env.nup, env.ndn)

        # --------------------------
        # Singles: up-spin
        # --------------------------
        for t in 1:env.nup
            i = up_occ[t]            # spatial
            (i <= frz) && continue   # never remove from frozen
            i_so = i                 # up spin-orb

            targets = Su[i]
            for u in 1:length(targets)
                a = targets[u]       # spatial
                (a <= frz) && continue
                a_so = a             # up spin-orb

                if !isocc_fast(d, a_so, env.om)
                    d2 = flip2(d,i_so, a_so, env.om)
                    d2 = rep_det(d2, T_plan) # get spin-coupled representative
                    insert_det!(gs, d2)
                    (length(gs.S) >= max_dets) && return
                end
            end
        end

        # --------------------------
        # Singles: down-spin
        # --------------------------
        for t in 1:env.ndn
            i_so = dn_occ[t]         # down spin-orb (n+1..2n)
            i = i_so - n             # spatial
            (i <= frz) && continue
            targets = Su[i]

            for u in 1:length(targets)
                a = targets[u]       # spatial
                (a <= frz) && continue
                a_so = a + n         # down spin-orb

                if !isocc_fast(d, a_so, env.om)
                    d2 = flip2(d, i_so, a_so, env.om)
                    d2 = rep_det(d2, T_plan) # get spin-coupled representative
                    insert_det!(gs, d2)
                    (length(gs.S) >= max_dets) && return
                end
            end
        end

        # --------------------------
        # Doubles: up-up (same spin)
        # --------------------------
        for p in 1:(env.nup - 1)
            i = up_occ[p]            # spatial
            (i <= frz) && continue
            i_so = i
            iri = IRREPS[i]

            for q in (p + 1):env.nup
                j = up_occ[q]        # spatial
                (j <= frz) && continue
                j_so = j

                key  = iri ⊻ IRREPS[j]
                cand = Dss[key + 1]  # (a,b) spatial, a<b

                for u in 1:length(cand)
                    a, b = cand[u]
                    (a <= frz || b <= frz) && continue

                    a_so = a
                    b_so = b
                    if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                        d2 = flip4(d, i_so, j_so, a_so, b_so, env.om)
                        d2 = rep_det(d2, T_plan) # get spin-coupled representative
                        insert_det!(gs, d2)
                        (length(gs.S) >= max_dets) && return
                    end
                end
            end
        end

        # --------------------------
        # Doubles: down-down (same spin)
        # --------------------------
        for p in 1:(env.ndn - 1)
            i_so = dn_occ[p]
            i = i_so - n
            (i <= frz) && continue
            iri = IRREPS[i]

            for q in (p + 1):env.ndn
                j_so = dn_occ[q]
                j = j_so - n
                (j <= frz) && continue

                key  = iri ⊻ IRREPS[j]
                cand = Dss[key + 1]

                for u in 1:length(cand)
                    a, b = cand[u]
                    (a <= frz || b <= frz) && continue

                    a_so = a + n
                    b_so = b + n
                    if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                        d2 = flip4(d, i_so, j_so, a_so, b_so, env.om)
                        d2 = rep_det(d2, T_plan)
                        insert_det!(gs, d2)
                        (length(gs.S) >= max_dets) && return
                    end
                end
            end
        end

        # --------------------------
        # Doubles: up-down (opposite spin)
        # interpret candidate (a,b) as: a is up target, b is down target
        # --------------------------
        for p in 1:env.nup
            i = up_occ[p]            # spatial
            (i <= frz) && continue
            i_so = i
            iri = IRREPS[i]

            for q in 1:env.ndn
                j_so = dn_occ[q]
                j = j_so - n
                (j <= frz) && continue

                key  = iri ⊻ IRREPS[j]
                cand = Dod[key + 1]  # (a,b) spatial (a up, b down), includes a==b

                for u in 1:length(cand)
                    a, b = cand[u]
                    (a <= frz || b <= frz) && continue

                    a_so = a
                    b_so = b + n
                    if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                        d2 = flip4(d, i_so, j_so, a_so, b_so, env.om)
                        d2 = rep_det(d2, T_plan)
                        insert_det!(gs, d2)
                        (length(gs.S) >= max_dets) && return
                    end
                end
            end
        end
    end

    return nothing
end