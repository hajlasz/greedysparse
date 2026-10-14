# Neon_sparsified_experiment.jl
# ============================================================================
# Shared driver for eigenvalue methods on the Neon (Slater deterimnant basis)
#
#   :greedy  Sparsified Greedy Bagging Method       
#   :tpm     Truncated Power Method  
#
# Assumes these files are already included (in dependency order):
#   bitops, bloom, env, bag, hamiltonian, restricted_matvec, expanding_matvec
# ============================================================================

using Arpack, DelimitedFiles, LinearAlgebra, JSON, LinearMaps
using DataStructures: RobinDict
using Base.Threads

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
struct ExperimentConfig
    norb_tot::Int # total number of spatial orbitals (including frozen)
    nelec_tot::Int # total number of electrons (assumed even, closed-shell)
    norbfrzn::Int # number of frozen orbitals (starting from the lowest)
    epsilon::Float64 # scaling parameter ε in A = σI - εH
    shift::Float64   # identity coefficient σ
    eri_file::String # file containing two-electron integrals
    hcore_file::String # file containing one-electron integrals
    symm_file::String # file containing symmetry labels for each orbital
    k_keep::Int          # truncation parameter m: keep top-k of |Av|
    outfile::String     # where to save the iteration history (JSON)
    max_iters::Int       # hard cap on iterations
    tol::Float64         # |E_t - E_{t-1}| threshold for the convergence streak
    eval_true::Float64   # reference |E_0| for reporting relative error
end

# ----------------------------------------------------------------------------
# One-time setup: integrals, symmetry, excitation lookups, env, HF
# ----------------------------------------------------------------------------
function build_env(cfg::ExperimentConfig)
    W   = cld(2 * cfg.norb_tot, 64)
    nup = cfg.nelec_tot ÷ 2
    ndn = cfg.nelec_tot ÷ 2
    @assert 2 * (cfg.nelec_tot ÷ 2) == cfg.nelec_tot "driver assumes closed-shell (nup == ndn)"

    IRREPS = vec(readdlm(cfg.symm_file, Int))
    irreps, _, _ = normalize_irreps_to_1based(IRREPS)

    ERIS  = reshape(readdlm(cfg.eri_file, ',', Float64),
                    cfg.norb_tot, cfg.norb_tot, cfg.norb_tot, cfg.norb_tot)
    HCORE = Array{Float64}(readdlm(cfg.hcore_file, ',', Float64))
    HCORE_DIAG, J, K = precompute_hdiag_JK(HCORE, ERIS)

    HF = get_ground_state(cfg.norb_tot, nup, ndn)

    single_targets, _ = get_single_excitation_targets_dense(irreps, cfg.norbfrzn, maximum(irreps))
    doub_same = build_pairs_same_spin_by_xor_buckets(IRREPS, cfg.norbfrzn)
    doub_diff = build_pairs_opposite_spin_by_xor_buckets(IRREPS, cfg.norbfrzn)

    om    = build_orbmasks(2 * cfg.norb_tot)
    eris1 = vec(ERIS)

    env = QCEnv{W}(
        cfg.norb_tot, nup, ndn, cfg.norbfrzn, cfg.epsilon, cfg.shift,
        ERIS, eris1, HCORE, IRREPS, HF,
        HCORE_DIAG, J, K,
        single_targets, doub_same, doub_diff, om,
    )
    return env, IRREPS, HF
end

# ----------------------------------------------------------------------------
# Sharded accumulator + per-thread scratch for the expanding matvec.
# ----------------------------------------------------------------------------
struct ExpandWorkspace{W}
    acc_shards::Vector{RobinDict{Det{W},Float64}}
    locks::Vector{Threads.SpinLock}
    buf_tls::Vector{UpdateBuffer{W}}
    scratch_tls::Vector{MatvecScratch}
    P::Int
end

function ExpandWorkspace(env::QCEnv{W}) where {W}
    nt = Threads.maxthreadid()
    P  = max(8 * nt, 1)
    acc_shards  = [RobinDict{Det{W},Float64}() for _ in 1:P]
    locks       = [Threads.SpinLock() for _ in 1:P]
    buf_tls     = [UpdateBuffer{W}(P; capacity = 4096) for _ in 1:nt]
    scratch_tls = make_matvec_scratch(env)
    return ExpandWorkspace{W}(acc_shards, locks, buf_tls, scratch_tls, P)
end

# ----------------------------------------------------------------------------
# Helpers shared by both methods.
# ----------------------------------------------------------------------------

# Build the restricted bag state (S, det2idx, cached H_ii) and its Bloom filter,
# then return a matvec-able LinearMap for A_{S,S} plus the state itself.
function restricted_operator(S::Vector{Det{W}}, env::QCEnv{W},
                             IRREPS::Vector{Int},
                             scratch::Vector{MatvecScratch}) where {W}
    gs = make_gs_from_S(S)
    build_hdiag!(gs, env)
    bf = build_bloom(gs)
    nS = length(S)
    Aop = LinearMap{Float64}(
        (y, x) -> mul_A_threaded!(y, x, gs, env, IRREPS, scratch, bf),
        nS, nS; ismutating = true,
    )
    return Aop, gs
end

# Atomic write (tmp + rename) so a killed job never leaves a half-written file.
function save_progress(outfile::AbstractString, data::Dict)
    tmp = outfile * ".tmp"
    open(tmp, "w") do io
        JSON.print(io, data)
    end
    mv(tmp, outfile; force = true)
    return nothing
end

# ----------------------------------------------------------------------------
# Sparsification schemes of the expanded Av, returns the selected dets, vals
# How the support S (= support ψ(Av)) is chosen from the shards:
#   :greedy / :tpm -> deterministic top-k
# ----------------------------------------------------------------------------

sparsify(::Val{:greedy},  ws, cfg, rng) = topk_from_shards(ws.acc_shards, cfg.k_keep)
sparsify(::Val{:tpm},     ws, cfg, rng) = topk_from_shards(ws.acc_shards, cfg.k_keep)


# ----------------------------------------------------------------------------
# Given the top-k (dets, vals) of the expanded Av, return (S, v, E_est) where
# v is the iterate carried into the next expansion.
# ----------------------------------------------------------------------------
function estimate_step(::Val{:greedy}, dets_top, vals_top,
                       env, IRREPS, scratch)
    S = dets_top                                   # S <- support(ψ(Av))
    Aop, _ = restricted_operator(S, env, IRREPS, scratch)
    λ, vecs = eigs(Aop; nev = 1, which = which_target(env))
    v = real(vecs[:, 1])
    return S, v, energy_from_eig(env, λ[1]), λ
end

function estimate_step(::Val{:tpm}, dets_top, vals_top,
                       env, IRREPS, scratch)
    S = dets_top
    v = copy(vals_top)
    v ./= norm(v)                                  # x_t = ψ(Av)/|ψ(Av)|
    Aop, _ = restricted_operator(S, env, IRREPS, scratch)
    w = similar(v)
    mul!(w, Aop, v)                                
    λ = dot(v, w) / dot(v, v)                      
    return S, v, energy_from_eig(env, λ), λ
end

# ----------------------------------------------------------------------------
# Main loop 
# ----------------------------------------------------------------------------
function run_experiment(method::Symbol, cfg::ExperimentConfig)
    methodval = Val(method)

    env, IRREPS, HF = build_env(cfg)
    ws = ExpandWorkspace(env)

    # Initial state: S = {HF}, v = [1]
    S = [HF]
    v = Float64[1.0]

    eigvals    = Float64[]
    bagsizes   = Int[]
    rel_errors = Float64[]
    iter_times = Float64[]

    patience      = 3
    stable_streak = 0
    iter          = 0

    while iter < cfg.max_iters
        iter += 1
        t0 = time()
        println("ITERATION $iter   |S| = $(length(S))"); flush(stdout)

        # y = (I - εH) v, accumulated into sharded dicts
        t_expand = @elapsed mulA_expand!(ws.acc_shards, ws.locks, ws.buf_tls,
                                         S, v, env, IRREPS, ws.scratch_tls)


        dets_top, vals_top = sparsify(methodval, ws, cfg, Random.GLOBAL_RNG)                             
        if isempty(dets_top)
            println("STOP: Av produced no entries.")
            break
        end

        # method-specific estimate and/or v update
        t_est = @elapsed begin
            S, v, E_est, λ = estimate_step(methodval, dets_top, vals_top,
                                           env, IRREPS, ws.scratch_tls)
        end

        nS  = length(S)
        rel = abs(abs(E_est) - abs(cfg.eval_true)) / abs(cfg.eval_true)
        push!(eigvals, E_est); push!(bagsizes, nS); push!(rel_errors, rel)

        # end if converged for long enough, else report progress and continue
        if length(eigvals) >= 2
            chg = abs(eigvals[end] - eigvals[end-1])
            stable_streak = chg < cfg.tol ? stable_streak + 1 : 0
            println("  E = $E_est   relerr = $rel   ΔE = $chg   streak = $stable_streak/$patience")
        else
            println("  E = $E_est   relerr = $rel   (first iteration)")
        end

        push!(iter_times, time() - t0)
        println("  expand = $(round(t_expand, digits=3)) s   estimate = $(round(t_est, digits=3)) s   |S| = $nS")
        println(); flush(stdout)

        stop_reason = stable_streak >= patience ? "tol_streak" :
                      (iter >= cfg.max_iters    ? "max_iters"  : "running")

        save_progress(cfg.outfile, Dict(
            "method"      => String(method),
            "iteration"   => iter,
            "eigvals"     => eigvals,
            "bagsizes"    => bagsizes,
            "rel_errors"  => rel_errors,
            "iter_times"  => iter_times,
            "k_keep"      => cfg.k_keep,
            "max_iters"   => cfg.max_iters,
            "tol"         => cfg.tol,
            "stop_reason" => stop_reason,
        ))

        if stable_streak >= patience
            println("STOP: |ΔE| < $(cfg.tol) for $patience consecutive iterations.")
            break
        end
    end

    return eigvals, bagsizes, rel_errors, iter_times
end

# ----------------------------------------------------------------------------
# Arg parsing shared by run_greedy.jl / run_tpm.jl.
# ----------------------------------------------------------------------------
function parse_config(args)
    return ExperimentConfig(
        parse(Int,     args[1]),   # norb_tot
        parse(Int,     args[2]),   # nelec_tot
        parse(Int,     args[3]),   # norbfrzn
        parse(Float64, args[4]),   # epsilon
        parse(Float64, get(ENV, "GS_SHIFT", "0.0")),  # shift
        args[5],                   # eri_file
        args[6],                   # hcore_file
        args[7],                   # symm_file
        parse(Int,     args[8]),   # k_keep
        args[9],                   # outfile
        parse(Int,     args[10]),  # max_iters
        parse(Float64, args[11]),  # tol
        parse(Float64, args[12]),  # eval_true
    )
end
