# hamiltonian.jl
# Excitation parities, one-/two-electron matrix elements <d1|H|d2>, the
# diagonal H_ii, and build_hdiag! (caches H_ii over a GreedyState).
# build_hdiag! takes a GreedyState, so this file must load AFTER bag.jl.

using LinearAlgebra
using DataStructures
using DataStructures: RobinDict
using Base.Threads

# ------------------------------------------------------------------------------
# Get parities of excitations
# ------------------------------------------------------------------------------
# Parity of a single excitation i->a starting from determinant d1
@inline function single_excitation_sign(d1::Det{W}, i::Int, a::Int) where {W}
    return isodd(_count_occ_between(d1, i, a)) ? -1 : 1
end
@inline function single_sign_from_occ(occ::Vector{Int}, pos_i::Int, a::Int)
    @inbounds i = occ[pos_i]
    t = searchsortedfirst(occ, a)  # insertion position for a (a assumed unoccupied)
    between = (a > i) ? (t - pos_i - 1) : (pos_i - t)
    return isodd(between) ? -1 : 1
end

@inline function double_excitation_sign_via_two_singles(
    d1::Det{W},
    i_so::Int, j_so::Int,
    a_so::Int, b_so::Int,
    om::OrbMasks
) where {W}
    # do i->a first, then j->b on the intermediate determinant
    s1  = single_excitation_sign(d1, i_so, a_so)
    dtmp = flip2(d1, i_so, a_so, om)             
    s2  = single_excitation_sign(dtmp, j_so, b_so)
    return s1 * s2
end
@inline function double_sign_same_spin_from_occ!(occ::Vector{Int}, tmp::Vector{Int},
                                                pos_i::Int, pos_j::Int, a::Int, b::Int)
    s1 = single_sign_from_occ(occ, pos_i, a)
    apply_single_excitation_sorted!(tmp, occ, pos_i, a)

    @inbounds j = occ[pos_j]
    pj = 0
    @inbounds for k in 1:length(tmp)
        if tmp[k] == j
            pj = k
            break
        end
    end

    s2 = single_sign_from_occ(tmp, pj, b)
    return s1 * s2
end
@inline function double_sign_opposite_spin_from_occ(up_occ::Vector{Int}, dn_occ::Vector{Int},
                                                    pos_i_up::Int, pos_j_dn::Int,
                                                    a_up::Int, b_dn::Int)
    return single_sign_from_occ(up_occ, pos_i_up, a_up) * single_sign_from_occ(dn_occ, pos_j_dn, b_dn)
end

# ------------------------------------------------------------------------------
# Compute matrix elements of the Hamiltonian
# ------------------------------------------------------------------------------
@inline function diagonal_value!(up_occ::Vector{Int}, dn_occ::Vector{Int},
                                      d::Det{W},
                                      hdiag::Vector{Float64},
                                      J::Matrix{Float64},
                                      K::Matrix{Float64},
                                      n::Int, nup::Int, ndn::Int) where {W}

    # fill occupied lists (allocation-free)
    occupied_orbitals!(up_occ, dn_occ, d, n, nup, ndn)

    h = 0.0

    # -------------------------
    # one-body term
    # -------------------------
    @inbounds for a in 1:nup
        i = up_occ[a]            # spatial 1..n
        h += hdiag[i]
    end
    @inbounds for a in 1:ndn
        i = dn_occ[a] - n        # map (n+1..2n) -> spatial (1..n)
        h += hdiag[i]
    end

    # -------------------------
    # two-body: αα = sum_{a<b} (J - K)
    # -------------------------
    @inbounds for a in 1:(nup-1)
        i = up_occ[a]
        for b in (a+1):nup
            j = up_occ[b]
            h += (J[i,j] - K[i,j])
        end
    end

    # -------------------------
    # two-body: ββ = sum_{a<b} (J - K)
    # (convert to spatial indices)
    # -------------------------
    @inbounds for a in 1:(ndn-1)
        i = dn_occ[a] - n
        for b in (a+1):ndn
            j = dn_occ[b] - n
            h += (J[i,j] - K[i,j])
        end
    end

    # -------------------------
    # two-body: αβ = sum J
    # -------------------------
    @inbounds for a in 1:nup
        i = up_occ[a]
        for b in 1:ndn
            j = dn_occ[b] - n
            h += J[i,j]
        end
    end

    return h
end

@inline function diagonal_value_from_occ(
    up_occ::Vector{Int}, dn_occ::Vector{Int},
    hdiag::Vector{Float64}, J::Matrix{Float64}, K::Matrix{Float64},
    n::Int, nup::Int, ndn::Int
)
    h = 0.0

    @inbounds for a in 1:nup
        i = up_occ[a]
        h += hdiag[i]
    end
    @inbounds for a in 1:ndn
        i = dn_occ[a] - n
        h += hdiag[i]
    end

    @inbounds for a in 1:(nup-1)
        i = up_occ[a]
        for b in (a+1):nup
            j = up_occ[b]
            h += (J[i,j] - K[i,j])
        end
    end

    @inbounds for a in 1:(ndn-1)
        i = dn_occ[a] - n
        for b in (a+1):ndn
            j = dn_occ[b] - n
            h += (J[i,j] - K[i,j])
        end
    end

    @inbounds for a in 1:nup
        i = up_occ[a]
        for b in 1:ndn
            j = dn_occ[b] - n
            h += J[i,j]
        end
    end

    return h
end


function build_hdiag!(gs::GreedyState{W}, env::QCEnv{W}) where {W}
    nS = length(gs.S)
    resize!(gs.hdiag, nS)

    up_occ = Vector{Int}(undef, env.nup)
    dn_occ = Vector{Int}(undef, env.ndn)

    @inbounds for i in 1:nS
        d = gs.S[i]
        # diagonal_value! should return H_ii (or write into something and return it)
        gs.hdiag[i] = diagonal_value!(up_occ, dn_occ, d,
                                      env.HCORE_DIAG, env.J, env.K,
                                      env.norb_tot, env.nup, env.ndn)
    end
    return nothing
end


# Compute ⟨d1|H|d2⟩ for a known double excitation (i_so,j_so)->(a_so,b_so).
# orbitals in 1..2n notation (1..n for spin-up, n+1..2n for spin-down); 
# sign is the excitation parity (±1).
# Assumes physicists' notation: eris[p,q,r,s] = ⟨pq|rs⟩
@inline function double_excitation_value(
    i_so::Int, j_so::Int,
    a_so::Int, b_so::Int,
    eris1::Vector{Float64},
    n::Int;
    n2::Int= n * n,
    n3::Int= n2 * n,
    sign::Int = 1)
    # spins
    si_up = is_up_so(i_so, n)
    sj_up = is_up_so(j_so, n)
    sa_up = is_up_so(a_so, n)
    sb_up = is_up_so(b_so, n)

    # spatial orbital numbers (1..n)
    i = spatial(i_so, n); j = spatial(j_so, n)
    a = spatial(a_so, n); b = spatial(b_so, n)

    if si_up == sj_up
        # same-spin double: antisymmetrized integral
        val = eri(eris1,n,n2, n3, a,b,i,j) - eri(eris1,n,n2, n3, a,b,j,i)
        return sign * val
    else
        # opposite-spin double:
        if sa_up == si_up
            val = eri(eris1,n,n2, n3, a,b,i,j)
        else
            # swap a,b so that a matches i-spin
            val = eri(eris1,n,n2, n3, b,a,i,j)
        end
        return sign * val
    end
end

# Compute ⟨d1|H|d2⟩ for a known single excitation i_so -> a_so (same spin).
# `up_occ` contains occupied spin-up spatial orbitals (1..n) in d1.
# `dn_occ` contains occupied spin-down *spin-orbital indices* (n+1..2n) in d1,
# as produced by occupied_orbitals!(...). These are mapped to spatial internally.
# sign is excitation parity (\pm 1)
# This is good for a single excitation but too slow for batch single excitations.
@inline function single_excitation_value_from_occ(
    up_occ::Vector{Int},
    dn_occ::Vector{Int},
    i_so::Int,
    a_so::Int,
    hcore::Matrix{Float64},
    eris1::Vector{Float64},
    n::Int;
    n2::Int = n * n,
    n3::Int = n2 * n,
    sign::Int = 1)
    

    i = spatial(i_so, n)
    a = spatial(a_so, n)

    nu = length(up_occ)
    nd = length(dn_occ)
    iup = is_up_so(i_so, n)

    # one-electron part
    h = @inbounds hcore[a,i]

    if iup 
        # same-spin (up)
        @inbounds for t in 1:nu 
            k = up_occ[t]
            if k != i
                J = eri(eris1,n, n2, n3,a,k,i,k)
                K = eri(eris1,n, n2, n3,a,k,k,i)
                h += (J - K)
            end
        end
        # opposite-spin (down)
        @inbounds for t in 1:nd 
            k = dn_occ[t] - n           # map spin-orb idx -> spatial
            h += eri(eris1,n, n2, n3,a,k,i,k)
        end
    else
        # same-spin (down)
        @inbounds for t in 1:nd 
            k = dn_occ[t] - n           # map spin-orb idx -> spatial
            if k != i
                J = eri(eris1,n, n2, n3,a,k,i,k)
                K = eri(eris1,n, n2, n3,a,k,k,i)
                h += (J - K)
            end
        end
        # opposite-spin (up) 
        @inbounds for t in 1:nu
            k = up_occ[t]
            h += eri(eris1,n, n2, n3,a,k,i,k)
        end
    end

    return sign * h
end

# In-place, O(n) single excitation on a SORTED occupancy list: remove occ[pos_i]
# and insert `a`, writing the resulting sorted list into `tmp`. Branches on a vs i
# so only the span between the deleted and inserted slots is shifted. Used to build
# the intermediate occupancy for double-excitation parities (double_sign_same_spin_from_occ!).
# Preconditions (unchecked): occ sorted ascending, a ∉ occ, a ≠ i.
@inline function apply_single_excitation_sorted!(tmp::Vector{Int}, occ::Vector{Int}, pos_i::Int, a::Int)
    n = length(occ)

    @inbounds for k in 1:n
        tmp[k] = occ[k]
    end
    @inbounds i = tmp[pos_i]

    if a > i
        # delete i (shift left)
        @inbounds for k in pos_i:(n-1)
            tmp[k] = tmp[k+1]
        end
        n1 = n - 1
        ins = searchsortedfirst(view(tmp, 1:n1), a)
        @inbounds for k in n:-1:(ins+1)
            tmp[k] = tmp[k-1]
        end
        @inbounds tmp[ins] = a
    else
        # delete i (shift right-to-left)
        @inbounds for k in pos_i:-1:2
            tmp[k] = tmp[k-1]
        end
        n1 = n - 1
        ins = searchsortedfirst(view(tmp, 2:n), a) + 1  # shift because we used tmp[2:n]
        @inbounds for k in 1:(ins-2)
            tmp[k] = tmp[k+1]
        end
        @inbounds tmp[ins-1] = a
    end
    return nothing
end