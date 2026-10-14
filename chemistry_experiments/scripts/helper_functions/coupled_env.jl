# coupled_env.jl
# ============================================================================
# One-time construction of the quantum-chemistry environment used by EVERY
# method that works in the spin-coupled basis (greedy, TPM, ASCI, PT2).
#
# QCEnv describes the plain Slater-determinant basis. The
# spin-coupled basis groups each determinant d with its spin-flipped partner
# T(d) into a single basis function. 
#
#   T_plan  the precomputed bit-permutation that maps d -> T(d)
#   C       the normalisation constants (1 and 1/sqrt(2)) that appear because a
#           determinant which is its OWN spin-flip needs a different prefactor
#           from one that pairs with a distinct partner
#   HF_rep  the Hartree-Fock determinant mapped to its representative. For a
#           closed-shell molecule HF is its own spin-flip, so this is just HF
#
# ============================================================================

using DelimitedFiles

"""
    build_coupled_env(norb_tot, nelec_tot, norbfrzn, epsilon,
                      eri_file, hcore_file, symm_file)
        -> (env, T_plan, C, IRREPS, HF_rep)

Read the integral/symmetry files and build the environment plus the
spin-coupling machinery. Assumes a closed-shell system (equal numbers of
spin-up and spin-down electrons)

Arguments
  norb_tot   total number of spatial orbitals, INCLUDING frozen ones
  nelec_tot  total number of electrons,        INCLUDING frozen ones
  norbfrzn   number of frozen orbitals (the lowest `norbfrzn` are never excited)
  epsilon    the shift parameter e in A = s*I - e*H
  eri_file   two-electron integrals, comma-separated, in row-major order
  hcore_file one-electron integrals, comma-separated
  symm_file  one integer per line: the symmetry label (irrep) of each orbital
  shift     the identity coefficient s in A = s*I - e*H; 0.0 = -e*H, 1.0 = I - e*H
"""
function build_coupled_env(norb_tot::Int, nelec_tot::Int, norbfrzn::Int,
                           epsilon::Float64, eri_file::String,
                           hcore_file::String, symm_file::String;
                           shift::Float64 = parse(Float64, get(ENV, "GS_SHIFT", "0.0")))

    # W = how many 64-bit integers it takes to store one determinant as a
    # bitstring. 
    W   = cld(2 * norb_tot, 64)
    nup = nelec_tot ÷ 2
    ndn = nelec_tot ÷ 2
    @assert 2 * (nelec_tot ÷ 2) == nelec_tot "coupled driver assumes closed-shell (nup == ndn)"

    # Orbital symmetry labels. `irreps` is a 1-based relabelling used only by
    # the single-excitation lookup table, which indexes an array by irrep.
    IRREPS = vec(readdlm(symm_file, Int))
    irreps, _, _ = normalize_irreps_to_1based(IRREPS)

    # One- and two-electron integrals over real spatial molecular orbitals.
    # ERIS is stored on disk as a flat list and reshaped into a 4-index array.
    ERIS  = reshape(readdlm(eri_file, ',', Float64),
                    norb_tot, norb_tot, norb_tot, norb_tot)
    HCORE = Array{Float64}(readdlm(hcore_file, ',', Float64))

    # HCORE_DIAG, J and K are combinations of the integrals that the diagonal
    # matrix element <d|H|d> needs. 
    HCORE_DIAG, J, K = precompute_hdiag_JK(HCORE, ERIS)

    HF = get_ground_state(norb_tot, nup, ndn)

    # Symmetry-filtered excitation targets, precomputed once 
    single_targets, _ = get_single_excitation_targets_dense(irreps, norbfrzn, maximum(irreps))
    doub_same = build_pairs_same_spin_by_xor_buckets(IRREPS, norbfrzn)
    doub_diff = build_pairs_opposite_spin_by_xor_buckets(IRREPS, norbfrzn)

    om    = build_orbmasks(2 * norb_tot)
    eris1 = vec(ERIS)   

    env = QCEnv{W}(
        norb_tot, nup, ndn, norbfrzn, epsilon, shift,
        ERIS, eris1, HCORE, IRREPS, HF,
        HCORE_DIAG, J, K,
        single_targets, doub_same, doub_diff, om,
    )
    T_plan = SpinFlipPlan(env.norb_tot, Val(W))
    C      = CoupledConsts()
    HF_rep = rep_det(HF, T_plan)

    return env, T_plan, C, IRREPS, HF_rep
end
