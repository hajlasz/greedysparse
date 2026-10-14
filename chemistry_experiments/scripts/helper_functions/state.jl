# state.jl
# Contains structs to define the system initially and to 
# keep track of the state of the system while running.

using DataStructures
using DataStructures: RobinDict
using Base.Threads

# ------------------------------------------------------------------------------
# ImmutableStructs to store precomputed values, parameters
# ------------------------------------------------------------------------------
struct QCEnv{W}
    norb_tot::Int # total number of spatial orbitals (including frozen)
    nup::Int  # number of spinup electrons
    ndn::Int  # number of spindown electrons
    norbfrzn::Int # number of frozen spatial orbitals
    epsilon::Float64   # scaling parameter ε in A = σI - εH
    shift::Float64     # identity coefficient σ in A = σI - εH; 0.0 = -εH, 1.0 = I - εH
    ERIS::Array{Float64,4}
    eris1::Vector{Float64}   # linearized ERIS for faster access
    HCORE::Matrix{Float64} 
    IRREPS::Vector{Int}    # IRREPS[i] = irrep of spatial orbital i, used to determine allowed excitations in expand_phase1!
    HF::Det{W}             # Hartree-Fock determinant 
    HCORE_DIAG::Vector{Float64} # HCORE_DIAG[i] = hcore[i,i], not multiplied by 1 or -epsilon
    J::Matrix{Float64}      # J[i,j] = eris[i,j,i,j]
    K::Matrix{Float64}      # K[i,j] = eris[i,j,j,i]
    single_targets::Vector{Vector{Int}}      # symmetry allowed single excitation targets from each spatial orbital
    doub_same::Vector{Vector{NTuple{2,Int}}} # symmetry allowed double excitation targets (same spin) from each irrep key
    doub_diff::Vector{Vector{NTuple{2,Int}}} # symmetry allowed double excitation targets (opposite spin) from each irrep key
    om::OrbMasks # orbital masks for fast occupation checks
end

@inline diag_A(env::QCEnv, Hii::Float64) = env.shift - env.epsilon * Hii
@inline energy_from_eig(env::QCEnv, λ)   = (env.shift - real(λ)) / env.epsilon
@inline rescale_for_shift(env::QCEnv, x::Float64, E_est::Float64) =
    x * (env.epsilon * abs(E_est)) / abs(env.shift - env.epsilon * E_est)

# Which end of A(S,S)'s spectrum the eigensolve targets.
#   :algebraic  follow sign(epsilon) -- always the ground state of H(S,S),
#               for any bag, any shift.  This is the variational choice.
#   :magnitude  ask for the largest-|lambda| eigenpair,
#               Equivalent only when the magnitude ordering happens to agree;
#               on a truncated bag it need not if A is not SPD.
# Set with GS_EIG_SELECT=algebraic|magnitude.  Validated at load, not at the
# first eigensolve, so a typo fails in seconds rather than after the expand.
function _parse_eig_select()
    s = Symbol(get(ENV, "GS_EIG_SELECT", "algebraic"))
    s in (:algebraic, :magnitude) || error(
        "GS_EIG_SELECT must be \"algebraic\" or \"magnitude\", got \"$s\"")
    return s
end
const EIG_SELECT = Ref(_parse_eig_select())

@inline which_target(env::QCEnv) =
    EIG_SELECT[] === :magnitude ? :LM : (env.epsilon > 0 ? :LR : :SR)

# eris1 accessors functions for QCEnv
@inline function eri(eris1::Vector{Float64}, n::Int, a::Int, b::Int, i::Int, j::Int)
    # eris[a,b,i,j] in column-major:
    idx = a + (b-1)*n + (i-1)*n*n + (j-1)*n*n*n
    @inbounds return eris1[idx]
end
@inline function eri(eris1::Vector{Float64}, n::Int, n2::Int, n3::Int,
                     a::Int, b::Int, i::Int, j::Int)::Float64
    idx = a + (b-1)*n + (i-1)*n2 + (j-1)*n3
    @inbounds return eris1[idx]
end

# ------------------------------------------------------------------------------
# Mutable Structs to keep track of the state of my system 
# ------------------------------------------------------------------------------
mutable struct GreedyState{W}
    S::Vector{Det{W}}                  # bag of determinants (indexed 1..|S|)
    det2idx::RobinDict{Det{W},Int}     # det -> index in S
    hdiag::Vector{Float64}             # cached H_{ii} for each det in S (same order as S), this is without the 1 or -epsilon factor 
end

# Build gs (S, det2idx) for the restricted matvec. hdiag is left empty here and
# filled later by build_hdiag!.
function make_gs_from_S(S::Vector{Det{W}}) where {W}
    det2idx = RobinDict{Det{W},Int}()
    sizehint!(det2idx, length(S))
    @inbounds for i in 1:length(S)
        det2idx[S[i]] = i
    end
    return GreedyState{W}(S, det2idx, Float64[])
end

# ------------------------------------------------------------------------------
# Precalculations for QCEnv: HDIAG, J, K, single and double excitation dicts
# ------------------------------------------------------------------------------
# hdiag[i] = hcore[i,i]
# J[i,j]   = eris[i,j,i,j]
# K[i,j]   = eris[i,j,j,i]
function precompute_hdiag_JK(hcore::AbstractMatrix{<:Real},
                             eris::Array{<:Real,4})
    n = size(hcore, 1)
    hdiag = Vector{Float64}(undef, n)
    J = Matrix{Float64}(undef, n, n)
    K = Matrix{Float64}(undef, n, n)

    @inbounds for i in 1:n
        hdiag[i] = hcore[i,i]
    end
    @inbounds for i in 1:n
        for j in 1:n
            J[i,j] = eris[i,j,i,j]
            K[i,j] = eris[i,j,j,i]
        end
    end

    return hdiag, J, K
end

#### LOOKUPS OF POSSIBLE JUMPS AND EXCITATION TARGETS

# normalized irreps are only used to build the single excitation lookup 
function normalize_irreps_to_1based(irreps::Vector{Int})
    labs = sort!(unique(irreps))
    mapping = Dict(labs[k] => k for k in eachindex(labs))  # old -> 1..K
    irr2 = similar(irreps)
    @inbounds for i in eachindex(irreps)
        irr2[i] = mapping[irreps[i]]
    end
    return irr2, length(labs), mapping
end

# targets[i] = which orbitals we can move to for a single excitation starting from orbital i
# takes in normalized irreps (1-based)
function get_single_excitation_targets_dense(irreps::Vector{Int},
                                             norbfrzn::Int,
                                             nirrep::Int)
    norb = length(irreps)

    orbs_in_irrep = [Int[] for _ in 1:nirrep]
    for j in (norbfrzn + 1):norb
        push!(orbs_in_irrep[irreps[j]], j)
    end

    targets = Vector{Vector{Int}}(undef, norb)

    for i in 1:norbfrzn
        targets[i] = Int[]
    end

    for i in (norbfrzn + 1):norb
        lst = orbs_in_irrep[irreps[i]]
        targets[i] = length(lst) == 1 ? Int[] : [j for j in lst if j != i]
    end

    return targets, orbs_in_irrep
end

# smallest power-of-two > max irrep label
nbuckets_from_irreps(irreps::AbstractVector{<:Integer}) =
    1 << (ndigits(maximum(irreps); base=2))

# possible double excitation going from i,j with same spin 
# to find possible orbitals, check buckets[irreps[i]⊻irreps[j] + 1]
function build_pairs_same_spin_by_xor_buckets(irreps::Vector{Int},
                                              norbfrzn::Int)
    n = length(irreps)
    lo = norbfrzn + 1
    nb = nbuckets_from_irreps(irreps)          # e.g. 8 for 0..7
    buckets = [NTuple{2,Int}[] for _ in 1:nb]  # bucket[key+1]

    @inbounds for a in lo:n-1
        ira = irreps[a]
        for b in (a+1):n
            key = ira ⊻ irreps[b]              # 0..nb-1
            push!(buckets[key + 1], (a,b))
        end
    end

    return buckets
end

# possible double excitation going from i,j with different spin 
# to find possible orbitals, check buckets[irreps[i]⊻irreps[j] + 1]
function build_pairs_opposite_spin_by_xor_buckets(irreps::Vector{Int},
                                                  norbfrzn::Int)
    n = length(irreps)
    lo = norbfrzn + 1
    nb = nbuckets_from_irreps(irreps)
    buckets = [NTuple{2,Int}[] for _ in 1:nb]  # bucket[key+1]

    @inbounds for a in lo:n
        ira = irreps[a]
        for b in lo:n
            key = ira ⊻ irreps[b]
            push!(buckets[key + 1], (a,b))     # NOTE: includes a==b
        end
    end

    return buckets
end

# ------------------------------------------------------------------------------
# Threading
# ------------------------------------------------------------------------------
struct MatvecScratch
    up_occ::Vector{Int}
    dn_occ::Vector{Int}
    up_tmp::Vector{Int}
    dn_tmp::Vector{Int}
end

function make_matvec_scratch(env::QCEnv)
    nt = Threads.maxthreadid()   
    scr = Vector{MatvecScratch}(undef, nt)
    for t in 1:nt
        scr[t] = MatvecScratch(
            Vector{Int}(undef, env.nup),
            Vector{Int}(undef, env.ndn),
            Vector{Int}(undef, env.nup),
            Vector{Int}(undef, env.ndn),
        )
    end
    return scr
end