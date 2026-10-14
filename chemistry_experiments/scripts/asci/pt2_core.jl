#=------------------------------------------------------------------------------
pt2_core.jl -- Coupled diagonals and in-memory Epstein-Nesbet PT2.

NOT FOR OZONE

Include after the determinant helpers and spin_coupled/coupled_expanding_matvec.jl.
------------------------------------------------------------------------------=#

using Printf

# =============================================================================
# ⟨d|H|T(d)⟩  —  the time-reversal coupling term (Sz = 0)
# =============================================================================

@inline function asci_tcoupling_term(
    d::Det{W}, td::Det{W},
    up_occ::Vector{Int}, dn_occ::Vector{Int},
    env::QCEnv{W}, n::Int, n2::Int, n3::Int,
)::Float64 where {W}
    om = env.om

    p_up = 0; n_upopen = 0          # spatial orbital that is up-occupied, down-empty  (1,0)
    p_dn = 0; n_dnopen = 0          # spatial orbital that is up-empty,   down-occupied (0,1)

    @inbounds for t in 1:env.nup
        p = up_occ[t]                       # spatial (1..n), up-occupied
        if !isocc_fast(d, p + n, om)        # down empty ⇒ (1,0) open shell
            p_up = p; n_upopen += 1
        end
    end
    @inbounds for t in 1:env.ndn
        p = dn_occ[t] - n                   # spatial (1..n), down-occupied
        if !isocc_fast(d, p, om)            # up empty ⇒ (0,1) open shell
            p_dn = p; n_dnopen += 1
        end
    end

    # Only exactly one open shell of each type gives a (double-excitation) coupling.
    (n_upopen == 1 && n_dnopen == 1) || return 0.0

    i_so = p_up          # removed up   (spin-orbital index = spatial, since up)
    j_so = p_dn + n      # removed down (spin-orbital index)
    a_so = p_dn          # added   up
    b_so = p_up + n      # added   down (spin-orbital index)

    @assert flip4(d, i_so, j_so, a_so, b_so, om) == td "asci_tcoupling_term: R/A extraction does not reproduce T(d)"

    sgn = double_excitation_sign_via_two_singles(d, i_so, j_so, a_so, b_so, om)
    return double_excitation_value(i_so, j_so, a_so, b_so, env.eris1, n;
                                   n2=n2, n3=n3, sign=sgn)
end

# =============================================================================
# H_ii^cpl  —  the coupled diagonal (Slater + time-reversal term)
# =============================================================================

"""
    asci_coupled_hdiag!(up_occ, dn_occ, d, env, T_plan, C) -> Float64

Coupled diagonal ⟨Φ_d|H|Φ_d⟩ for representative determinant `d`.
`up_occ`/`dn_occ` are caller-provided scratch (lengths env.nup / env.ndn);
they are overwritten.  This is the version to use in hot loops.
"""
@inline function asci_coupled_hdiag!(
    up_occ::Vector{Int}, dn_occ::Vector{Int},
    d::Det{W}, env::QCEnv{W}, T_plan::SpinFlipPlan, C::CoupledConsts,
)::Float64 where {W}
    n  = env.norb_tot
    n2 = n * n
    n3 = n2 * n

    occupied_orbitals!(up_occ, dn_occ, d, n, env.nup, env.ndn)
    Hs = diagonal_value_from_occ(up_occ, dn_occ, env.HCORE_DIAG, env.J, env.K,
                                 n, env.nup, env.ndn)

    td = T_det(d, T_plan)
    (td == d) && return Hs                       # T-fixed ⇒ no coupling term
    return Hs + asci_tcoupling_term(d, td, up_occ, dn_occ, env, n, n2, n3)
end

"""
    asci_coupled_hdiag(d, env, T_plan, C) -> Float64

Allocating convenience wrapper (builds its own scratch).  Use in tests /
non-hot code; prefer the `!` version inside loops.
"""
function asci_coupled_hdiag(d::Det{W}, env::QCEnv{W},
                            T_plan::SpinFlipPlan, C::CoupledConsts) where {W}
    up = Vector{Int}(undef, env.nup)
    dn = Vector{Int}(undef, env.ndn)
    return asci_coupled_hdiag!(up, dn, d, env, T_plan, C)
end

# =============================================================================
# In-core Epstein–Nesbet PT2  (STAGE 1 — Neon-scale ONLY)
# =============================================================================

"""
    asci_pt2_incore(S, x, E, env, T_plan, C, IRREPS;
                    P, expected_ext, allowed, gap_floor, verbose)
        -> (E_pt2, n_ext, n_gap_skipped)

Second-order Epstein–Nesbet correction to the variational state (S, x) with
variational energy E, computed in core.  `x` must be the coupled-basis
eigenvector (normalized), `S` the representative determinants.
"""
function asci_pt2_incore(
    S::Vector{Det{W}}, x::Vector{Float64}, E::Float64,
    env::QCEnv{W}, T_plan::SpinFlipPlan, C::CoupledConsts, IRREPS::Vector{Int};
    P::Int = max(8, Threads.nthreads()),
    expected_ext::Int = 8 * length(S),
    allowed::Union{Nothing,Set{Det{W}}} = nothing,
    gap_floor::Float64 = 1e-10,
    verbose::Bool = false,
) where {W}

    @assert length(x) == length(S) "asci_pt2_incore: |x|=$(length(x)) ≠ |S|=$(length(S))"
    eps = env.epsilon

    # --- source norm factors N_d for the coupled matvec (= gs.norm_fac) ---
    norm_fac = Vector{Float64}(undef, length(S))
    @inbounds for i in eachindex(S)
        norm_fac[i] = is_T_fixed(S[i], T_plan) ? C.invsqrt2 : 1.0
    end

    # --- numerator: val_i = −ε·b_i accumulated at every target i ---
    acc = ExpandingAccumulator{W}(P, expected_ext)
    expanding_matvec_coupled!(acc, S, x, norm_fac, env, T_plan, C, IRREPS)
    merged = merge_T_partners!(acc, T_plan)      # Dict: rep -> val_i (= −ε·b_i for externals)

    # --- internal set for exclusion (S are already representatives) ---
    Sset = Set(S)

    # --- E_PT2 = Σ_{i∉S} b_i^2 / (E − H_ii^cpl),  with b_i^2 = val_i^2 / ε^2 ---
    inv_eps2 = 1.0 / (eps * eps)
    up = Vector{Int}(undef, env.nup)
    dn = Vector{Int}(undef, env.ndn)

    E_pt2 = 0.0
    n_ext = 0
    n_gap = 0
    for (d, val) in merged
        (d in Sset) && continue                                  # internal → skip
        (allowed !== nothing && !(d in allowed)) && continue     # optional restriction
        Hii = asci_coupled_hdiag!(up, dn, d, env, T_plan, C)     # coupled diagonal (with T-term)
        gap = E - Hii
        if abs(gap) < gap_floor
            n_gap += 1                                           # intruder/degeneracy guard
            continue
        end
        E_pt2 += (val * val) * inv_eps2 / gap
        n_ext += 1
    end

    if verbose
        @printf("  asci_pt2_incore: |S|=%d  externals=%d  gap-skipped=%d  E_PT2=%.10f Ha\n",
                length(S), n_ext, n_gap, E_pt2)
    end
    return (E_pt2 = E_pt2, n_ext = n_ext, n_gap_skipped = n_gap)
end