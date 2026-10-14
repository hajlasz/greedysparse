#=------------------------------------------------------------------------------
run_tpm_coupled_ooc.jl

Truncated Power Method in spin-coupled basis with out-of-core
sharded drain and checkpoint resume support.
------------------------------------------------------------------------------=#

using Printf
using JSON
using LinearAlgebra

"""
    run_tpm_coupled_ooc(env, T_plan, C, IRREPS; m, q, batch_size,
                        spill_dir, starting_det=HF_rep,
                        outfile=nothing, tol=1e-7, patience=3,
                        checkpoint_file=nothing, resume=false)

Truncated Power Method with OOC drain and checkpoint resume.

If `checkpoint_file` is set and `resume=true`, the function attempts to
load (S, v, iter, history) and continue. After every iteration the
checkpoint is overwritten atomically.

Note on patience streak across resume: when resuming, we initialize
stable_streak from the loaded history's |ΔE| trail. This means the
patience counter is preserved correctly across restarts.
"""
function run_tpm_coupled_ooc(
    env::QCEnv{W},
    T_plan::SpinFlipPlan,
    C::CoupledConsts,
    IRREPS::Vector{Int};
    m::Int,
    q::Int,
    batch_size::Int,
    spill_dir::String,
    starting_det::Det{W} = rep_det(get_ground_state(env.norb_tot, env.nup, env.ndn), T_plan),
    outfile::Union{Nothing,String} = nothing,
    tol::Float64 = 1e-7,
    patience::Int = 3,
    checkpoint_file::Union{Nothing,String} = nothing,
    resume::Bool = false,
) where {W}

    # ---------- Initialize state, possibly from checkpoint ----------
    eigval_history  = Float64[]
    bagsize_history = Int[]
    iter_times      = Float64[]
    iter_start      = 1

    S = Det{W}[starting_det]
    v = Float64[1.0]

    resumed = false
    if resume && checkpoint_file !== nothing
        ckpt = load_checkpoint(checkpoint_file, Det{W})
        if ckpt !== nothing
            iter_completed, S_ck, v_ck, hist_ck, bag_ck, t_ck = ckpt
            S = S_ck
            v = v_ck
            eigval_history  = hist_ck
            bagsize_history = bag_ck
            iter_times      = t_ck
            iter_start = iter_completed + 1
            resumed = true
        end
    end

    if !resumed && isempty(bagsize_history)
        push!(bagsize_history, 1)
    end

    gs = build_dual_indexed_state_from_S(S, env, T_plan, C)

    eigval = isempty(eigval_history) ? gs.hdiag[1] : eigval_history[end]
    tol = rescale_for_shift(env, tol, eigval)

    # Recompute the patience streak from the loaded history
    stable_streak = 0
    if length(eigval_history) >= 2
        for i in (length(eigval_history) - patience + 1):length(eigval_history)
            i < 2 && continue
            chg = abs(eigval_history[i] - eigval_history[i-1])
            if chg < tol
                stable_streak += 1
            else
                stable_streak = 0
            end
        end
    end

    println("═══════════════════════════════════════════════════════════")
    println("  Truncated Power Method (spin-coupled, OOC drain)")
    @printf("  m = %d, q_max = %d, batch_size = %d, ε = %.2e\n",
            m, q, batch_size, env.epsilon)
    @printf("  spill_dir = %s\n", spill_dir)
    if checkpoint_file !== nothing
        @printf("  checkpoint_file = %s\n", checkpoint_file)
    end
    println("  drain: OOC-EXACT (no pruning, on-disk sharded)")
    @printf("  convergence: tol = %.1e on |ΔE|, patience = %d\n", tol, patience)
    if resumed
        @printf("  ▶ RESUMED from checkpoint at iteration %d\n", iter_start - 1)
        @printf("  ▶ Energy at resume: %.10f Ha,  |S| = %d,  streak = %d/%d\n",
                eigval, length(gs.S), stable_streak, patience)
    else
        @printf("  Starting fresh. HF diag: %.10f\n", eigval)
    end
    println("═══════════════════════════════════════════════════════════")
    println()
    flush(stdout)

    isdir(spill_dir) || mkpath(spill_dir)

    for t in iter_start:q
        t_iter_start = time()

        println("--- Iteration $t ---")
        @printf("  |S| = %d, |v| = %d\n", length(gs.S), length(v))
        flush(stdout)

        # ============= Expand + sparsify =============
        t_expand = @elapsed begin
            S_new, x_new = batched_expand_merge_select_ooc(
                gs.S, v, gs.norm_fac,
                env, T_plan, C, IRREPS, m;
                batch_size = batch_size,
                spill_dir = spill_dir,
                verbose = true,
            )
        end

        nS_new = length(S_new)
        if nS_new == 0
            println("  STOP: A v produced no entries.")
            break
        end

        x_norm = norm(x_new)
        if x_norm == 0.0 || !isfinite(x_norm)
            @warn "x_new has zero / non-finite norm at iteration $t; stopping."
            break
        end
        x_new ./= x_norm

        @printf("  expand+select (OOC): %.2fs, |S_new| = %d\n", t_expand, nS_new)
        flush(stdout)

        # ============= Build new state =============
        t_build = @elapsed begin
            gs = build_dual_indexed_state_from_S(S_new, env, T_plan, C)
            bf = build_dual_bloom(gs, T_plan)
            scratch = make_matvec_scratch(env)
        end
        @printf("  build state: %.2fs\n", t_build)
        flush(stdout)

        # ============= Rayleigh quotient =============
        nS = length(gs.S)
        w = Vector{Float64}(undef, nS)
        t_rq = @elapsed dual_indexed_mul_A_threaded!(w, x_new, gs, env, IRREPS, scratch, bf)
        λA = dot(x_new, w)
        E  = energy_from_eig(env, λA)

        v      = x_new
        eigval = E

        push!(eigval_history,  E)
        push!(bagsize_history, nS_new)

        t_iter = time() - t_iter_start
        push!(iter_times, t_iter)

        @printf("  Rayleigh quotient: %.2fs,  λ(A) = %.10f\n", t_rq, λA)
        @printf("  Energy estimate:   %.10f Ha\n", E)
        @printf("  Iteration total:   %.2fs\n", t_iter)

        # Convergence check
        if length(eigval_history) >= 2
            chg = abs(eigval_history[end] - eigval_history[end-1])
            if chg < tol
                stable_streak += 1
            else
                stable_streak = 0
            end
            @printf("  |ΔE| = %.3e  (tol %.1e)  streak = %d/%d\n",
                    chg, tol, stable_streak, patience)
        else
            println("  |ΔE| = n/a (first iteration)")
        end

        # ============= Save progress =============
        if outfile !== nothing
            data = Dict(
                "method"              => "TPM_coupled_ooc",
                "iteration"           => t,
                "eigval_history"      => eigval_history,
                "bagsize_history"     => bagsize_history,
                "iter_times"          => iter_times,
                "current_eigval"      => E,
                "current_bagsize"     => nS_new,
                "m"                   => m,
                "q_max"               => q,
                "batch_size"          => batch_size,
                "spill_dir"           => spill_dir,
                "tol"                 => tol,
                "patience"            => patience,
                "epsilon"             => env.epsilon,
                "shift"           => env.shift,
                "eig_select"      => String(EIG_SELECT[]),
                "which"           => String(which_target(env)),
                "timestamp"           => time(),
            )
            tmp = outfile * ".tmp"
            open(tmp, "w") do io
                JSON.print(io, data)
            end
            mv(tmp, outfile; force = true)
        end

        if checkpoint_file !== nothing
            t_ckpt = @elapsed save_checkpoint(
                checkpoint_file, t,
                gs.S, v,
                eigval_history, bagsize_history, iter_times,
            )
            @printf("  checkpoint: %.2fs (%.1f MB)\n",
                    t_ckpt, filesize(checkpoint_file) / 1024^2)
        end

        println()
        flush(stdout)

        if stable_streak >= patience
            println("STOP: energy estimate stabilized for $patience iterations (|ΔE| < $tol).")
            break
        end
    end

    println("═══════════════════════════════════════════════════════════")
    @printf("  Final energy: %.10f\n", eigval)
    @printf("  Final |S|:    %d\n", length(gs.S))
    @printf("  Iterations:   %d (run from %d to %d)\n",
            length(eigval_history), iter_start, iter_start + length(eigval_history) - 1)
    println("═══════════════════════════════════════════════════════════")
    flush(stdout)

    return gs, v, eigval, eigval_history, bagsize_history, iter_times
end