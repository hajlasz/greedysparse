#=------------------------------------------------------------------------------
run_asci_coupled_ooc.jl

ASCI iteration in the spin-flip-symmetrized basis, with an out-of-core search
and an explicit sparse eigensolve. A fresh run starts from the Hartree-Fock
state

Starting from space S, coefficients v, and variational energy E, each step is:

  1. Select core determinants G: up to N_cdets states of S with largest |v_i|.
  2. Search their allowed single and double excitations in batches on disk.
     For candidates outside G, the unscreened ranking score is

         s_i = sum_{j in G} H_ij v_j / (E - H_ii),

     where H and its full diagonal H_ii are in the coupled basis. Retain up
     to N_tdets candidates by |s_i|. Optional eps_thresh screening and the
     near-zero-denominator guard are implemented in asci_search.jl.
  3. Merge into a space of at most N_tdets states. The default :include rule
     retains G and fills remaining slots by |s_i|; :rank compares generator
     |v_i| and candidate |s_i| together. Candidates may be states of S outside G.
  4. Assemble A_SS = sigma*I - epsilon*H_SS with build_sparse_A_csc and solve
     the restricted eigenproblem with ARPACK, using the normalized merged
     coefficients as the initial guess and the configured eigenvalue target.
     Recover E = (sigma - lambda)/epsilon; the eigenvector is the next iterate.

The sparse matrix is reused throughout each eigensolve and must fit in RAM.
N_cdets <= N_tdets is required. The loop runs for at most q iterations, with
optional early stopping when the energy change falls below econv.

When enabled, checkpoints save the state after each iteration and JSON
progress is written atomically. Resuming continues from the next iteration.
The saved checkpoint can also be read by the PT2 drivers.
------------------------------------------------------------------------------=#

using Printf
using JSON
using LinearAlgebra
using LinearMaps

"""
    _asci_build_and_solve(gs, env, IRREPS, bf, v0, arpack_tol)

Build A = I - epsilon*H, diagonalize it, and return ONLY scalars plus the
eigenvector: `(lambda, v, nnzA, t_build, t_arpack)`.

WHY THIS IS A SEPARATE FUNCTION.  At |S| = 2e7 the matrix is ~427 GiB and it
MUST NOT survive into the next iteration's search.  Job 15263821 died that way.

The obvious fix -- `A_sparse = nothing; GC.gc()` in the loop body -- was tried
in job 15474431 and DID NOT WORK: the live heap printed immediately after that
collection was still 434.1 GiB, i.e. the matrix was still reachable.  Clearing
one variable is not enough inside a large function: the GC frame, temporaries
introduced by macro expansion (`@elapsed` bodies), and the closure Arpack builds
over the operator can each keep the object rooted until the enclosing call
returns.

Giving the build and the solve their own function removes the ambiguity.  On
return no frame holds a reference, so the collection at the call site is
guaranteed to reclaim the matrix.  This is the standard Julia idiom for bounding
the lifetime of a large temporary; it does not rely on inference details.

`resid` and `evecs` are deliberately not returned, and `v` is a fresh copy, so
ARPACK's Krylov workspace dies with the call as well.
"""
function _asci_build_and_solve(gs, env, IRREPS, bf, v0::Vector{Float64},
                               arpack_tol::Float64)
    t_build = @elapsed A = build_sparse_A_csc(gs, env, IRREPS, bf)
    nnzA = nnz(A)

    local lam, v
    t_arpack = @elapsed begin
        lamvec, evecs, nconv, niter, nmult, resid = eigs(
            A; nev = 1, which = which_target(env), v0 = v0,
            tol = arpack_tol, maxiter = 300,
        )
        lam = real(lamvec[1])
        v   = copy(real(vec(evecs)))
        @printf("  ARPACK: niter=%d, nmult=%d, nconv=%d\n", niter, nmult, nconv)
        flush(stdout)
    end

    return lam, v, nnzA, t_build, t_arpack
end

"""
    run_asci_coupled_ooc(env, T_plan, C, IRREPS; N_cdets, N_tdets, q, batch_size,
                         spill_dir, merge_mode=:include, eps_thresh=0.0,
                         econv=0.0, starting_det=HF_rep, outfile=nothing,
                         arpack_tol=1e-7, compact_every=200, max_disk_gb=3000.0,
                         checkpoint_file=nothing, resume=false)

ASCI in the spin-coupled basis with OOC search and optional checkpoint resume.

Required keywords
  N_cdets   :: Int     core/generator size (top-|v| determinants that generate
                       singles+doubles each iteration).
  N_tdets   :: Int     target space size (rank of the matrix diagonalized).
  q         :: Int     maximum number of ASCI iterations.
  batch_size:: Int     generators processed per batch in the OOC search.
  spill_dir :: String  scratch directory for the on-disk candidate shards.

Optional keywords
  merge_mode :: Symbol  :include (MACIS, default) or :rank (Tubman-2016).
  eps_thresh :: Float64 partial-score screen in the search (0.0 = keep all).
  econv      :: Float64 stop early when |ΔE| < econv Ha (0.0 = run all q iters).
  arpack_tol :: Float64 ARPACK residual tolerance (energy accuracy ≈ tol/ε).
  compact_every, max_disk_gb : passed through to asci_search_ooc (disk control).

If `checkpoint_file !== nothing` and `resume == true`, state is restored and the
run continues from the next iteration; the checkpoint is overwritten each step.
"""
function run_asci_coupled_ooc(
    env::QCEnv{W},
    T_plan::SpinFlipPlan,
    C::CoupledConsts,
    IRREPS::Vector{Int};
    N_cdets::Int,
    N_tdets::Int,
    q::Int,
    batch_size::Int,
    spill_dir::String,
    merge_mode::Symbol = :include,
    eps_thresh::Float64 = 0.0,
    econv::Float64 = 0.0,
    starting_det::Det{W} = rep_det(get_ground_state(env.norb_tot, env.nup, env.ndn), T_plan),
    outfile::Union{Nothing,String} = nothing,
    arpack_tol::Float64 = 1e-7,
    compact_every::Int = 200,
    max_disk_gb::Float64 = 3000.0,
    checkpoint_file::Union{Nothing,String} = nothing,
    resume::Bool = false,
) where {W}

    merge_mode === :include || merge_mode === :rank ||
        error("merge_mode must be :include or :rank, got :$merge_mode")
    N_cdets <= N_tdets ||
        error("need N_cdets <= N_tdets (got N_cdets=$N_cdets, N_tdets=$N_tdets)")
    merge_fn = merge_mode === :include ? asci_merge_include : asci_merge_rank

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
        if isempty(bagsize_history)
            push!(bagsize_history, 1)
        end
    end

    gs = build_dual_indexed_state_from_S(S, env, T_plan, C)
    scratch = make_matvec_scratch(env)

    # Current variational energy E.  Fresh start: the 1-determinant space {HF}
    # has energy ⟨HF|H|HF⟩ = gs.hdiag[1], i.e. E_HF -- the correct E_0 for the
    # first search denominator (E - H_ii).
    eigval = isempty(eigval_history) ? gs.hdiag[1] : eigval_history[end]

    println("═══════════════════════════════════════════════════════════")
    println("  ASCI (spin-coupled, OOC search)")
    @printf("  N_cdets = %d, N_tdets = %d, q = %d, ε = %.2e\n",
            N_cdets, N_tdets, q, env.epsilon)
    @printf("  merge rule = :%s   eps_thresh = %.2e\n", merge_mode, eps_thresh)
    @printf("  ARPACK tol = %.1e → %.1e at σ=%.1f  (energy accuracy ≲ %.1e Ha)\n",
        arpack_tol, rescale_for_shift(env, arpack_tol, eigval), env.shift,
        arpack_tol * abs(eigval))
    if econv > 0
        @printf("  convergence: stop when |ΔE| < %.1e Ha\n", econv)
    end
    @printf("  spill_dir = %s\n", spill_dir)
    if checkpoint_file !== nothing
        @printf("  checkpoint_file = %s\n", checkpoint_file)
    end
    println("  search: OOC (on-disk sharded, screened by eps_thresh)")
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
        @printf("  |S| = %d, |v| = %d,  E = %.10f\n", length(gs.S), length(v), eigval)
        flush(stdout)

        # ===================== ASCI search + merge (Step 5 swap) ==============
        # The ONLY block that differs from the greedy driver.
        t_expand = @elapsed begin
            # (1) generators: the top-N_cdets determinants of S by |coefficient|.
            #     gs.S, v, gs.norm_fac are mutually index-aligned.
            ngen = min(N_cdets, length(gs.S))
            gidx = collect(1:length(v))
            if ngen < length(gidx)
                partialsort!(gidx, 1:ngen, by = i -> -abs(v[i]))
            end
            gen_idx  = @view gidx[1:ngen]
            gen_dets = gs.S[gen_idx]
            gen_v    = v[gen_idx]
            gen_nf   = gs.norm_fac[gen_idx]

            # (2) search: singles+doubles of the generators, scored by the ASCI
            #     metric with denominator (E - H_ii); keep the top N_tdets.
            cand_dets, cand_scores = asci_search_ooc(
                gen_dets, gen_v, gen_nf,
                env, T_plan, C, IRREPS,
                N_tdets, eigval;
                batch_size    = batch_size,
                spill_dir     = spill_dir,
                eps_thresh    = eps_thresh,
                compact_every = compact_every,
                max_disk_gb   = max_disk_gb,
                verbose       = true,
            )

            # (3) merge generators (true coeffs) with candidates (scores).
            S_new, x_new = merge_fn(gen_dets, gen_v, cand_dets, cand_scores, N_tdets)
        end

        x_norm = norm(x_new)
        if x_norm > 0
            x_new ./= x_norm
        end

        nS_new = length(S_new)
        @printf("  search+merge (OOC): %.2fs,  generators = %d,  |S_new| = %d\n",
                t_expand, ngen, nS_new)
        flush(stdout)

        # ============= Build new state (identical to greedy driver) =============
        t_build = @elapsed begin
            gs = build_dual_indexed_state_from_S(S_new, env, T_plan, C)
            bf = build_dual_bloom(gs, T_plan)
            scratch = make_matvec_scratch(env)
        end
        @printf("  build state: %.2fs\n", t_build)
        flush(stdout)

        push!(bagsize_history, nS_new)

        # ============= Restricted eigenproblem (SPARSE-CSC variant) =============
        # Build A = I - εH explicitly ONCE per iteration via the memory-lean
        # two-pass CSC assembly, then ARPACK does cheap sparse multiplies.
        # build_sparse_A_csc is validated (test_sparse_A.jl) to reproduce
        # dual_indexed_mul_A_threaded! (and build_sparse_A) to round-off, so the
        # eigenvalue and E = (1-λ)/ε recovery below are unchanged -- same operator.
        nS = length(gs.S)

        v0 = x_new
        if length(v0) != nS
            v0 = zeros(Float64, nS)
            v0[1] = 1.0
        end

        # Build + diagonalize in their own scope so the matrix is provably
        # unreachable on return.  See the docstring on _asci_build_and_solve.
        λ1, v, nnzA, t_buildH, t_arpack = _asci_build_and_solve(
            gs, env, IRREPS, bf, v0, rescale_for_shift(env, arpack_tol, eigval),
        )
        v0 = nothing

        @printf("  build sparse A: %.2fs  (nnz=%d, %.0f nnz/row, %.2f GiB)\n",
                t_buildH, nnzA, nnzA / nS,
                (nnzA * 16 + (nS + 1) * 8) / 1024^3)
        flush(stdout)

        # Reclaim the matrix NOW, before the next search allocates.  Two passes:
        # the first drops the CSC wrapper, the second its colptr/rowval/nzval.
        # malloc_trim then returns the freed pages to the OS -- glibc otherwise
        # retains a ~450 GiB arena, and SLURM enforces its limit on RSS, not on
        # Julia's notion of live bytes.
        GC.gc(); GC.gc()
        @static if Sys.islinux()
            try
                ccall(:malloc_trim, Cint, (Csize_t,), 0)
            catch
            end
        end
        @printf("  freed A: live heap %.1f GiB, RSS %.1f GiB\n",
                Base.gc_live_bytes() / 1024^3,
                parse(Float64, split(read("/proc/self/statm", String))[2]) *
                    4096 / 1024^3)
        flush(stdout)

        eigval = energy_from_eig(env, λ1)
        push!(eigval_history, eigval)

        t_iter = time() - t_iter_start
        push!(iter_times, t_iter)

        @printf("  ARPACK: %.2fs\n", t_arpack)
        @printf("  Energy estimate: %.10f\n", eigval)
        @printf("  Iteration total: %.2fs\n", t_iter)

        # ============= Save progress (identical to greedy driver) ===============
        if outfile !== nothing
            data = Dict(
                "method"          => "ASCI_coupled_ooc",
                "merge_mode"      => String(merge_mode),
                "iteration"       => t,
                "eigval_history"  => eigval_history,
                "bagsize_history" => bagsize_history,
                "iter_times"      => iter_times,
                "current_eigval"  => eigval,
                "current_bagsize" => nS_new,
                "N_cdets"         => N_cdets,
                "N_tdets"         => N_tdets,
                "eps_thresh"      => eps_thresh,
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

        # ============= Optional convergence stop ================================
        if econv > 0 && length(eigval_history) >= 2
            dE = abs(eigval_history[end] - eigval_history[end-1])
            if dE < econv
                @printf("  ✓ Converged: |ΔE| = %.2e < %.2e Ha\n", dE, econv)
                println()
                flush(stdout)
                break
            end
        end

        println()
        flush(stdout)
    end

    println("═══════════════════════════════════════════════════════════")
    @printf("  Final energy: %.10f\n", eigval)
    @printf("  Final |S|:    %d\n", length(gs.S))
    @printf("  Iterations:   %d (run from %d)\n",
            length(eigval_history), iter_start)
    println("═══════════════════════════════════════════════════════════")
    flush(stdout)

    return gs, v, eigval, eigval_history, bagsize_history
end