#=------------------------------------------------------------------------------
run_greedy_coupled_ooc.jl

The greedy method in the spin-coupled basis,
using the out-of-core sharded drain.

Checkpointing: when `checkpoint_file` is provided and `resume=true`, the
function will attempt to restore (S, v, iteration, history) from disk and
continue from the next iteration. After every iteration, it writes the
current state back atomically (temp file + rename).
------------------------------------------------------------------------------=#

using Printf
using JSON
using LinearAlgebra
using LinearMaps

"""
    run_greedy_coupled_ooc(env, T_plan, C, IRREPS; m, q, batch_size,
                               spill_dir, starting_det=HF_rep,
                               outfile=nothing, arpack_tol=1e-7,
                               checkpoint_file=nothing, resume=false)

The greedy method in the spin-coupled basis with OOC drain and optional checkpoint
resume.

If `checkpoint_file !== nothing` and `resume == true`, attempt to load
state from that file and continue from where the previous run stopped.
On every successful iteration, the checkpoint is overwritten with the
current state.
"""
function run_greedy_coupled_ooc(
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
    arpack_tol::Float64 = 1e-7,
    checkpoint_file::Union{Nothing,String} = nothing,
    resume::Bool = false,
) where {W}

    # ---------- Initialize state, possibly from checkpoint ----------
    eigval_history  = Float64[]
    bagsize_history = Int[]
    iter_times      = Float64[]
    iter_start      = 1     # first iteration to run

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

    if !resumed
        # Fresh start: |S|=1 sets bagsize_history[1] for consistency
        if isempty(bagsize_history)
            push!(bagsize_history, 1)
        end
    end

    gs = build_dual_indexed_state_from_S(S, env, T_plan, C)
    scratch = make_matvec_scratch(env)

    eigval = isempty(eigval_history) ? gs.hdiag[1] : eigval_history[end]

    println("═══════════════════════════════════════════════════════════")
    println("  Greedy (spin-coupled, OOC drain)")
    @printf("  m = %d, q = %d, batch_size = %d, ε = %.2e\n",
            m, q, batch_size, env.epsilon)
    @printf("  ARPACK tol = %.1e → %.1e at σ=%.1f  (energy accuracy ≲ %.1e Ha)\n",
            arpack_tol, rescale_for_shift(env, arpack_tol, eigval), env.shift,
            arpack_tol * abs(eigval))
    @printf("  spill_dir = %s\n", spill_dir)
    if checkpoint_file !== nothing
        @printf("  checkpoint_file = %s\n", checkpoint_file)
    end
    println("  drain: OOC-EXACT (no pruning, on-disk sharded)")
    if resumed
        @printf("  ▶ RESUMED from checkpoint at iteration %d\n", iter_start - 1)
        @printf("  ▶ Energy at resume: %.10f Ha,  |S| = %d\n", eigval, length(gs.S))
    else
        @printf("  Starting fresh. Initial energy estimate: %.10f\n", eigval)
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

        # ============= y = Ax/||Ax||, sparsify, update S =============
        t_expand = @elapsed begin
            S_new, x_new = batched_expand_merge_select_ooc(
                gs.S, v, gs.norm_fac,
                env, T_plan, C, IRREPS, m;
                batch_size = batch_size,
                spill_dir = spill_dir,
                verbose = true,
            )
        end

        x_norm = norm(x_new)
        if x_norm > 0
            x_new ./= x_norm
        end

        nS_new = length(S_new)
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

        push!(bagsize_history, nS_new)

        # ============= Restricted eigenproblem =============
        nS = length(gs.S)
        Aop = LinearMap{Float64}(
            (y, x) -> dual_indexed_mul_A_threaded!(y, x, gs, env, IRREPS, scratch, bf),
            nS, nS; ismutating = true
        )

        v0 = x_new
        if length(v0) != nS
            v0 = zeros(Float64, nS)
            v0[1] = 1.0
        end

        t_arpack = @elapsed begin
            λ, evecs, nconv, niter, nmult, resid = eigs(
                Aop; nev = 1, which = :LR, v0 = v0,
                tol = rescale_for_shift(env, arpack_tol, eigval), maxiter = 300,
            )
        end

        v = real(vec(evecs))
        eigval = energy_from_eig(env, λ[1])
        push!(eigval_history, eigval)

        t_iter = time() - t_iter_start
        push!(iter_times, t_iter)

        @printf("  ARPACK: %.2fs (niter=%d, nmult=%d, nconv=%d)\n",
                t_arpack, niter, nmult, nconv)
        @printf("  Energy estimate: %.10f\n", eigval)
        @printf("  Iteration total: %.2fs\n", t_iter)

        # ============= Save progress =============
        if outfile !== nothing
            data = Dict(
                "method"          => "greedy_coupled_ooc",
                "iteration"       => t,
                "eigval_history"  => eigval_history,
                "bagsize_history" => bagsize_history,
                "iter_times"      => iter_times,
                "current_eigval"  => eigval,
                "current_bagsize" => nS_new,
                "m"               => m,
                "q_max"           => q,
                "batch_size"      => batch_size,
                "spill_dir"       => spill_dir,
                "epsilon"         => env.epsilon,
                "shift"           => env.shift,
                "eig_select"      => String(EIG_SELECT[]),
                "which"           => String(which_target(env)),
                "arpack_tol_effective" => rescale_for_shift(env, arpack_tol, eigval),
                "timestamp"       => time(),
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
    end

    println("═══════════════════════════════════════════════════════════")
    @printf("  Final energy: %.10f\n", eigval)
    @printf("  Final |S|:    %d\n", length(gs.S))
    @printf("  Iterations:   %d (run from %d to %d)\n",
            length(eigval_history), iter_start, iter_start + length(eigval_history) - 1)
    println("═══════════════════════════════════════════════════════════")
    flush(stdout)

    return gs, v, eigval, eigval_history, bagsize_history
end