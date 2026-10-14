# asci_experiment.jl
# ============================================================================
# Shared driver for ASCI (Adaptive Sampling Configuration Interaction) in the
# SPIN-COUPLED basis with the OUT-OF-CORE (OOC) search.
# ============================================================================

using Printf

# ----------------------------------------------------------------------------
# Configuration
#
# ----------------------------------------------------------------------------
struct ASCIConfig
    norb_tot::Int      # total number of spatial orbitals (including frozen)
    nelec_tot::Int     # total number of electrons (assumed even, closed-shell)
    norbfrzn::Int      # number of frozen orbitals (starting from the lowest)
    epsilon::Float64   # the shift parameter e in A = I - e*H
    eri_file::String   # file containing two-electron integrals
    hcore_file::String # file containing one-electron integrals
    symm_file::String  # file containing symmetry labels for each orbital
    N_cdets::Int       # core size: how many determinants may generate candidates
    N_tdets::Int       # target size: how many determinants get diagonalized
    batch_size::Int    # how many generators to expand per batch in the search
    q_max::Int         # maximum number of ASCI iterations
    outfile::String    # where to save the iteration history (JSON)
    spill_dir::String  # scratch directory for the on-disk candidate shards
end

function parse_asci_config(args)
    return ASCIConfig(
        parse(Int,     args[1]),          # norb_tot
        parse(Int,     args[2]),          # nelec_tot
        parse(Int,     args[3]),          # norbfrzn
        parse(Float64, args[4]),          # epsilon
        args[5],                          # eri_file
        args[6],                          # hcore_file
        args[7],                          # symm_file
        Int(parse(Float64, args[8])),     # N_cdets     (accepts 5e5-style)
        Int(parse(Float64, args[9])),     # N_tdets     (accepts 2e6-style)
        Int(parse(Float64, args[10])),    # batch_size
        parse(Int,     args[11]),         # q_max
        args[12],                         # outfile
        args[13],                         # spill_dir
    )
end

build_coupled_env(cfg::ASCIConfig) =
    build_coupled_env(cfg.norb_tot, cfg.nelec_tot, cfg.norbfrzn, cfg.epsilon,
                      cfg.eri_file, cfg.hcore_file, cfg.symm_file)

# ----------------------------------------------------------------------------
# Method-specific pieces
# ----------------------------------------------------------------------------

# One banner line describing the method's parameters.
method_params_line(::Val{:asci}, cfg::ASCIConfig; merge_mode, eps_thresh, econv, kwargs...) =
    @sprintf("ASCI params: N_cdets=%d, N_tdets=%d, batch_size=%d, q_max=%d  (OOC search, sparse-CSC eigensolve)",
             cfg.N_cdets, cfg.N_tdets, cfg.batch_size, cfg.q_max)

method_params_line(::Val{M}, cfg::ASCIConfig; kwargs...) where {M} =
    error("Unknown method :$M -- see 'HOW TO ADD A NEW ASCI-SHAPED METHOD' at the top of asci_experiment.jl")

# The call into the method's iteration loop.
function launch(::Val{:asci}, cfg::ASCIConfig, env, T_plan, C, IRREPS, HF_rep,
                ckpt_file; merge_mode, eps_thresh, econv, arpack_tol,
                max_disk_gb, compact_every, resume)
    return run_asci_coupled_ooc(
        env, T_plan, C, IRREPS;
        N_cdets         = cfg.N_cdets,
        N_tdets         = cfg.N_tdets,
        q               = cfg.q_max,
        batch_size      = cfg.batch_size,
        spill_dir       = cfg.spill_dir,
        merge_mode      = merge_mode,
        eps_thresh      = eps_thresh,
        econv           = econv,
        starting_det    = HF_rep,
        outfile         = cfg.outfile,
        arpack_tol      = arpack_tol,
        compact_every   = compact_every,
        max_disk_gb     = max_disk_gb,
        checkpoint_file = ckpt_file,
        resume          = resume,
    )
end

launch(::Val{M}, cfg::ASCIConfig, env, T_plan, C, IRREPS, HF_rep, ckpt_file;
       kwargs...) where {M} =
    error("Unknown method :$M -- see 'HOW TO ADD A NEW ASCI-SHAPED METHOD' at the top of asci_experiment.jl")

# ----------------------------------------------------------------------------
# Keyword arguments:
#   merge_mode    :include (all core determinants always kept)
#                 or :rank (rank the core determinants by their partial score and keep the top N_cdets)
#   eps_thresh    screening threshold in the search; 0.0 keeps every candidate
#                 (exact). A small positive value discards candidates whose
#                 partial score is negligible, which bounds the candidate bag.
#   econv         stop early once |dE| between iterations falls below this
#                 (in Hartree); 0.0 means always run all q_max iterations
#   arpack_tol    eigensolver residual tolerance. Note that the energy error is
#                 roughly arpack_tol/epsilon, because E_0 = (1 - lambda)/epsilon
#                 divides by epsilon and so magnifies the eigenvalue error.
#   compact_every how often (in batches) to compact the on-disk shards
#   max_disk_gb   abort cleanly rather than filling the scratch filesystem
#   resume        restore state from the canonical checkpoint and continue
# ----------------------------------------------------------------------------
function run_asci_experiment(method::Symbol, cfg::ASCIConfig;
                             merge_mode::Symbol    = :include,
                             eps_thresh::Float64   = 0.0,
                             econv::Float64        = 0.0,
                             arpack_tol::Float64   = 1e-7,
                             compact_every::Int    = 200,
                             max_disk_gb::Float64  = 3000.0,
                             resume::Bool          = false)

    merge_mode === :include || merge_mode === :rank ||
        error("merge_mode must be :include or :rank, got :$merge_mode")
    cfg.N_cdets <= cfg.N_tdets ||
        error("need N_cdets <= N_tdets (got N_cdets=$(cfg.N_cdets), N_tdets=$(cfg.N_tdets))")

    env, T_plan, C, IRREPS, HF_rep = build_coupled_env(cfg)

    ckpt_file = checkpoint_path_for(cfg.outfile, env.shift)

    W = cld(2 * cfg.norb_tot, 64)   # for the banner; same formula as build_coupled_env
    @printf("\nSystem: n=%d, ne=%d, frozen=%d, W=%d, ε=%.2e\n",
            cfg.norb_tot, cfg.nelec_tot, cfg.norbfrzn, W, cfg.epsilon)
    println(method_params_line(Val(method), cfg;
                               merge_mode = merge_mode, eps_thresh = eps_thresh,
                               econv = econv))
    @printf("Merge rule:      :%s\n", merge_mode)
    @printf("eps_thresh:      %.2e   econv: %.2e\n", eps_thresh, econv)
    @printf("Spill dir:       %s\n", cfg.spill_dir)
    @printf("Output JSON:     %s\n", cfg.outfile)
    @printf("Checkpoint file: %s\n", ckpt_file)
    @printf("Resume mode:     %s\n", resume ? "TRUE" : "FALSE (fresh start)")
    flush(stdout)

    return launch(Val(method), cfg, env, T_plan, C, IRREPS, HF_rep, ckpt_file;
                  merge_mode = merge_mode, eps_thresh = eps_thresh, econv = econv,
                  arpack_tol = arpack_tol, max_disk_gb = max_disk_gb,
                  compact_every = compact_every, resume = resume)
end
