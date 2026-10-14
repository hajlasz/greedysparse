# coupled_ooc_experiment.jl
# ============================================================================
# Shared driver for eigenvalue methods in the SPIN-COUPLED basis with the
# OUT-OF-CORE (OOC) drain
#
#   :greedy  Submatrix Selection Method
#   :tpm     Truncated Power Method 
#
# ============================================================================

using DelimitedFiles
using Printf

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
struct CoupledOOCConfig
    norb_tot::Int      # total number of spatial orbitals (including frozen)
    nelec_tot::Int     # total number of electrons (assumed even, closed-shell)
    norbfrzn::Int      # number of frozen orbitals (starting from the lowest)
    epsilon::Float64   # the shift parameter ε in A = I - εH
    eri_file::String   # file containing two-electron integrals
    hcore_file::String # file containing one-electron integrals
    symm_file::String  # file containing symmetry labels for each orbital
    m::Int             # truncation parameter: keep the top-m of |Av| each iteration
    batch_size::Int    # how many source determinants to expand per drain batch
    q_max::Int         # maximum number of iterations
    outfile::String    # where to save the iteration history (JSON)
    spill_dir::String  # scratch directory for the on-disk drain (>1 TB at m = 1M)
end

function parse_coupled_ooc_config(args)
    return CoupledOOCConfig(
        parse(Int,     args[1]),          # norb_tot
        parse(Int,     args[2]),          # nelec_tot
        parse(Int,     args[3]),          # norbfrzn
        parse(Float64, args[4]),          # epsilon
        args[5],                          # eri_file
        args[6],                          # hcore_file
        args[7],                          # symm_file
        Int(parse(Float64, args[8])),     # m           (accepts 1e6-style)
        Int(parse(Float64, args[9])),     # batch_size  (accepts 1e6-style)
        parse(Int,     args[10]),         # q_max
        args[11],                         # outfile
        args[12],                         # spill_dir
    )
end

# ----------------------------------------------------------------------------
# One-time setup.
#
# ----------------------------------------------------------------------------
build_coupled_env(cfg::CoupledOOCConfig) =
    build_coupled_env(cfg.norb_tot, cfg.nelec_tot, cfg.norbfrzn, cfg.epsilon,
                      cfg.eri_file, cfg.hcore_file, cfg.symm_file)


# One banner line describing the method's parameters.
method_params_line(::Val{:greedy}, cfg::CoupledOOCConfig; arpack_tol, tol, patience) =
    @sprintf("Greedy params: m=%d, batch_size=%d, q_max=%d  (nev=1, λ₁ only, OOC)",
             cfg.m, cfg.batch_size, cfg.q_max)

method_params_line(::Val{:tpm}, cfg::CoupledOOCConfig; arpack_tol, tol, patience) =
    @sprintf("TPM params: m=%d, batch_size=%d, q_max=%d, tol=%.1e, patience=%d  (OOC)",
             cfg.m, cfg.batch_size, cfg.q_max, tol, patience)

method_params_line(::Val{M}, cfg::CoupledOOCConfig; kwargs...) where {M} =
    error("Unknown method :$M — see 'HOW TO ADD A NEW METHOD' at the top of coupled_ooc_experiment.jl")

# The call into the method's iteration loop.
function launch(::Val{:greedy}, cfg::CoupledOOCConfig, env, T_plan, C, IRREPS, HF_rep,
                ckpt_file; arpack_tol, tol, patience, resume)
    # tol/patience are TPM's convergence controls and are not used here;
    # the greedy loop runs to q_max with no convergence-based early stop.
    return run_greedy_coupled_ooc(
        env, T_plan, C, IRREPS;
        m               = cfg.m,
        q               = cfg.q_max,
        batch_size      = cfg.batch_size,
        spill_dir       = cfg.spill_dir,
        starting_det    = HF_rep,
        outfile         = cfg.outfile,
        arpack_tol      = arpack_tol,
        checkpoint_file = ckpt_file,
        resume          = resume,
    )
end

function launch(::Val{:tpm}, cfg::CoupledOOCConfig, env, T_plan, C, IRREPS, HF_rep,
                ckpt_file; arpack_tol, tol, patience, resume)
    # arpack_tol is the greedy method's eigensolver tolerance and is not
    # used here; TPM has no eigensolve.
    return run_tpm_coupled_ooc(
        env, T_plan, C, IRREPS;
        m               = cfg.m,
        q               = cfg.q_max,
        batch_size      = cfg.batch_size,
        spill_dir       = cfg.spill_dir,
        starting_det    = HF_rep,
        outfile         = cfg.outfile,
        tol             = tol,
        patience        = patience,
        checkpoint_file = ckpt_file,
        resume          = resume,
    )
end

launch(::Val{M}, cfg::CoupledOOCConfig, env, T_plan, C, IRREPS, HF_rep, ckpt_file;
       kwargs...) where {M} =
    error("Unknown method :$M — see 'HOW TO ADD A NEW METHOD' at the top of coupled_ooc_experiment.jl")

# ----------------------------------------------------------------------------
# Entry point shared by run_greedy_ooc.jl / run_tpm_ooc.jl.
#
# Keyword arguments:
#   resume      whether to restore state from the canonical checkpoint (both)
#   arpack_tol  eigensolver residual tolerance                    (:greedy)
#   tol         convergence threshold on |ΔE| between iterations  (:tpm)
#   patience    stop after this many consecutive |ΔE| < tol       (:tpm)
# ----------------------------------------------------------------------------
function run_coupled_ooc_experiment(method::Symbol, cfg::CoupledOOCConfig;
                                    resume::Bool     = false,
                                    arpack_tol::Float64 = 1e-7,
                                    tol::Float64        = 1e-7,
                                    patience::Int       = 3)
    env, T_plan, C, IRREPS, HF_rep = build_coupled_env(cfg)

    ckpt_file = checkpoint_path_for(cfg.outfile, env.shift)

    W = cld(2 * cfg.norb_tot, 64)   # for the banner; same formula as build_coupled_env
    @printf("\nSystem: n=%d, ne=%d, frozen=%d, W=%d, ε=%.2e\n",
            cfg.norb_tot, cfg.nelec_tot, cfg.norbfrzn, W, cfg.epsilon)
    println(method_params_line(Val(method), cfg;
                               arpack_tol = arpack_tol, tol = tol, patience = patience))
    @printf("Spill dir:       %s\n", cfg.spill_dir)
    @printf("Output JSON:     %s\n", cfg.outfile)
    @printf("Checkpoint file: %s\n", ckpt_file)
    @printf("Resume mode:     %s\n", resume ? "TRUE" : "FALSE (fresh start)")
    flush(stdout)

    return launch(Val(method), cfg, env, T_plan, C, IRREPS, HF_rep, ckpt_file;
                  arpack_tol = arpack_tol, tol = tol, patience = patience,
                  resume = resume)
end