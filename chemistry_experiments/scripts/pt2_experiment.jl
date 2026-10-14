# pt2_experiment.jl
# ============================================================================
# Shared driver for the Epstein-Nesbet PT2 correction.
#
#   :spill    (asci/pt2_spill.jl)
#   :ooc      (asci/pt2_ooc.jl)
#
# ----------------------------------------------------------------------------
# WHICH VARIANT TO USE
# ----------------------------------------------------------------------------
#   :spill   faster, but uses more RAM. No compaction but disk space can be
#            controlled with eps2 screening.
#
#   :ooc     uses less RAM, has compaction, no eps2 screening
#
# ============================================================================

using Printf

struct PT2Config
    norb_tot::Int       # total number of spatial orbitals (including frozen)
    nelec_tot::Int      # total number of electrons (assumed even, closed-shell)
    norbfrzn::Int       # number of frozen orbitals (starting from the lowest)
    epsilon::Float64    # the shift parameter e in A = I - e*H
    eri_file::String    # file containing two-electron integrals
    hcore_file::String  # file containing one-electron integrals
    symm_file::String   # file containing symmetry labels for each orbital
    ckpt_file::String   # converged variational checkpoint to correct
end

"""
    parse_pt2_config(args) -> PT2Config

Parse the 8 common positional arguments (args[1:8]). Trailing arguments are
ignored here -- the entry points handle them.
"""
function parse_pt2_config(args)
    return PT2Config(
        parse(Int,     args[1]),   # norb_tot
        parse(Int,     args[2]),   # nelec_tot
        parse(Int,     args[3]),   # norbfrzn
        parse(Float64, args[4]),   # epsilon
        args[5],                   # eri_file
        args[6],                   # hcore_file
        args[7],                   # symm_file
        args[8],                   # ckpt_file
    )
end

build_coupled_env(cfg::PT2Config) =
    build_coupled_env(cfg.norb_tot, cfg.nelec_tot, cfg.norbfrzn, cfg.epsilon,
                      cfg.eri_file, cfg.hcore_file, cfg.symm_file)

# ----------------------------------------------------------------------------
# Method-specific pieces, selected by Val dispatch.
# ----------------------------------------------------------------------------

method_params_line(::Val{:spill}, cfg::PT2Config; n_shards, batch_size, spill_dir,
                   eps2, kwargs...) =
    @sprintf("PT2 params (:spill, RAM-bounded): n_shards=%d, batch_size=%d, eps2=%.1e",
             n_shards, batch_size, eps2)

method_params_line(::Val{:ooc}, cfg::PT2Config; P_out, batch_size, spill_dir,
                   compact_every, max_disk_gb, kwargs...) =
    @sprintf("PT2 params (:ooc, DISK-bounded, exact): P_out=%d, batch_size=%d, compact_every=%d, max_disk_gb=%.0f",
             P_out, batch_size, compact_every, max_disk_gb)

method_params_line(::Val{M}, cfg::PT2Config; kwargs...) where {M} =
    error("Unknown PT2 variant :$M -- see 'HOW TO ADD A NEW PT2 VARIANT' at the top of pt2_experiment.jl")

function launch(::Val{:spill}, cfg::PT2Config, S, v, E_var, env, T_plan, C, IRREPS;
                n_shards, batch_size, spill_dir, eps2, kwargs...)
    return asci_pt2_spill(S, v, E_var, env, T_plan, C, IRREPS;
                          batch_size = batch_size,
                          eps2       = eps2,
                          n_shards   = n_shards,
                          spill_dir  = spill_dir,
                          verbose    = true)
end

function launch(::Val{:ooc}, cfg::PT2Config, S, v, E_var, env, T_plan, C, IRREPS;
                P_out, batch_size, spill_dir, compact_every, max_disk_gb, kwargs...)
    return asci_pt2_ooc_compact(S, v, E_var, env, T_plan, C, IRREPS;
                                batch_size    = batch_size,
                                spill_dir     = spill_dir,
                                P_out         = P_out,
                                compact_every = compact_every,
                                max_disk_gb   = max_disk_gb,
                                verbose       = true)
end

launch(::Val{M}, cfg::PT2Config, S, v, E_var, env, T_plan, C, IRREPS; kwargs...) where {M} =
    error("Unknown PT2 variant :$M -- see 'HOW TO ADD A NEW PT2 VARIANT' at the top of pt2_experiment.jl")

# ----------------------------------------------------------------------------
# Entry point shared by run_asci_pt2.jl / run_asci_pt2_ooc.jl.
# ----------------------------------------------------------------------------
function run_pt2_experiment(variant::Symbol, cfg::PT2Config; kwargs...)
    env, T_plan, C, IRREPS, _ = build_coupled_env(cfg)
    W = cld(2 * cfg.norb_tot, 64)

    @printf("\nSystem: n=%d, ne=%d, frozen=%d, W=%d, ε=%.2e\n",
            cfg.norb_tot, cfg.nelec_tot, cfg.norbfrzn, W, cfg.epsilon)

    # ---- load the converged variational state --------------------------------
    loaded = load_checkpoint(cfg.ckpt_file, Det{W})
    loaded === nothing && error("could not load checkpoint: $(cfg.ckpt_file)")
    iter, S, v, eigval_history, _, _ = loaded
    length(v) == length(S) ||
        error("checkpoint is inconsistent: |v| = $(length(v)) but |S| = $(length(S))")
    isempty(eigval_history) &&
        error("checkpoint has an empty eigval_history, so there is no E_var to correct")
    E_var = eigval_history[end]

    println(method_params_line(Val(variant), cfg; kwargs...))
    @printf("Checkpoint:      %s\n", cfg.ckpt_file)
    @printf("  iteration %d   |S| = %d   E_var = %.10f Ha\n", iter, length(S), E_var)
    flush(stdout)

    t = @elapsed res = launch(Val(variant), cfg, S, v, E_var, env, T_plan, C, IRREPS;
                              kwargs...)

    println("\n═══════════════════════════════════════════════════════════")
    @printf("  |S|              = %d\n", length(S))
    @printf("  externals summed = %d\n", res.n_ext)
    @printf("  gap-skipped      = %d\n", res.n_gap_skipped)
    @printf("  peak shard       = %d\n", res.peak_shard)
    @printf("  wall time        = %.1f s\n", t)
    println("  ─────────────────────────────────────────")
    @printf("  E_var            = %.10f Ha\n", E_var)
    @printf("  E_PT2            = %.10f Ha\n", res.E_pt2)
    @printf("  E_var + E_PT2    = %.10f Ha\n", E_var + res.E_pt2)
    println("═══════════════════════════════════════════════════════════")
    flush(stdout)

    return (E_var = E_var, E_pt2 = res.E_pt2, E_total = E_var + res.E_pt2,
            n_ext = res.n_ext, peak_shard = res.peak_shard)
end
