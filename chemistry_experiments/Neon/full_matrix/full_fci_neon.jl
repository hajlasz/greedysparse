#=============================================================================
full_fci_neon.jl

Builds the FULL FCI Hamiltonian for Neon as an explicit SparseMatrixCSC and
computes lambda_1, lambda_2, lambda_n together with the true leading
eigenvector.
=============================================================================#

using Arpack, Combinatorics, DelimitedFiles, LinearAlgebra, Random,
      SparseArrays, JSON, LinearMaps, DataStructures
using DataStructures: RobinDict
using Base.Threads

const HERE = @__DIR__                                            # .../Neon/full_matrix
const NEON = normpath(joinpath(HERE, ".."))                      # .../Neon
const HELP = normpath(joinpath(HERE, "..", "..", "scripts", "helper_functions"))

# DO NOT REORDER
include(joinpath(HELP, "bitops.jl"))
include(joinpath(HELP, "bloom.jl"))
include(joinpath(HELP, "state.jl"))
include(joinpath(HELP, "hamiltonian.jl"))
include(joinpath(HELP, "restricted_matvec.jl"))
include(joinpath(HERE, "progress.jl"))

# ---------------------------------------------------------------------------
# Environment. Identical to build_env in Neon_sparsified_experiment.jl, except
# that epsilon is irrelevant here and is set to 1.0: this script assembles H
# itself, not A = I - eps*H, so the reported eigenvalues are energies.
# ---------------------------------------------------------------------------
function build_env_plain(norb_tot::Int, nelec_tot::Int, norbfrzn::Int,
                         eri_file::String, hcore_file::String, symm_file::String)
    W   = cld(2 * norb_tot, 64)
    nup = nelec_tot ÷ 2
    ndn = nelec_tot ÷ 2
    @assert 2 * nup == nelec_tot "closed shell assumed"

    IRREPS = vec(readdlm(symm_file, Int))
    irreps, _, _ = normalize_irreps_to_1based(IRREPS)

    ERIS  = reshape(readdlm(eri_file, ',', Float64),
                    norb_tot, norb_tot, norb_tot, norb_tot)
    HCORE = Array{Float64}(readdlm(hcore_file, ',', Float64))
    HCORE_DIAG, J, K = precompute_hdiag_JK(HCORE, ERIS)

    HF = get_ground_state(norb_tot, nup, ndn)

    single_targets, _ = get_single_excitation_targets_dense(irreps, norbfrzn, maximum(irreps))
    doub_same = build_pairs_same_spin_by_xor_buckets(IRREPS, norbfrzn)
    doub_diff = build_pairs_opposite_spin_by_xor_buckets(IRREPS, norbfrzn)

    # epsilon = 1.0, shift = 0.0: this script wants exactly -H, so that the
    # matvec check against the assembled matrix is a check on H itself.
    env = QCEnv{W}(norb_tot, nup, ndn, norbfrzn, 1.0, 0.0,
                   ERIS, vec(ERIS), HCORE, IRREPS, HF,
                   HCORE_DIAG, J, K,
                   single_targets, doub_same, doub_diff,
                   build_orbmasks(2 * norb_tot))
    return env, IRREPS
end

# ---------------------------------------------------------------------------
# Full determinant space of one symmetry block.
#
# A determinant is a pair of spin strings. Frozen orbitals are occupied in both
# strings, so their irreps cancel in the total and the block condition is
#
#     xor of active up irreps  ==  xor of active down irreps  xor  sym_key
#
# with sym_key = 0 the block containing Hartree-Fock. Strings are bucketed by
# their irrep xor and every allowed pair of buckets is enumerated, so the space
# is produced directly rather than by a breadth-first closure.
# ---------------------------------------------------------------------------
function build_full_space(env::QCEnv{W}, IRREPS::Vector{Int}; sym_key::Int = 0) where {W}
    n    = env.norb_tot
    frz  = env.norbfrzn
    nact = env.nup - frz
    @assert env.nup == env.ndn
    active = collect((frz + 1):n)

    # strings[key] = list of active-orbital tuples whose irrep xor is key
    strings = Dict{Int,Vector{Vector{Int}}}()
    for c in combinations(active, nact)
        key = 0
        for o in c
            key ⊻= IRREPS[o]
        end
        push!(get!(strings, key, Vector{Vector{Int}}()), copy(c))
    end

    nS = 0
    for (key, us) in strings
        ds = get(strings, key ⊻ sym_key, Vector{Vector{Int}}())
        nS += length(us) * length(ds)
    end
    say(@sprintf("strings per spin %d, determinants in block %d",
                 sum(length, values(strings)), nS))
    p = Progress("space", nS; every = 15.0)

    S = Vector{Det{W}}(undef, nS)
    idx = 0
    frozen_bits = zeros(UInt64, W)
    for i in 1:frz
        w, b = wordbit(i);      frozen_bits[w] |= bitmask(b)
        w, b = wordbit(n + i);  frozen_bits[w] |= bitmask(b)
    end

    for (key, us) in strings
        ds = get(strings, key ⊻ sym_key, Vector{Vector{Int}}())
        isempty(ds) && continue
        for u in us
            uw = copy(frozen_bits)
            for o in u
                w, b = wordbit(o); uw[w] |= bitmask(b)
            end
            for d in ds
                words = copy(uw)
                for o in d
                    w, b = wordbit(n + o); words[w] |= bitmask(b)
                end
                idx += 1
                S[idx] = Det{W}(Tuple(words))
            end
            bump!(p, length(ds); report = true)
        end
    end
    @assert idx == nS
    finish!(p)

    # Put Hartree-Fock first when it belongs to this block, so that it is easy
    # to use as an ARPACK starting vector.
    if sym_key == 0
        pos_hf = findfirst(==(env.HF), S)
        @assert pos_hf !== nothing "HF is not in the requested symmetry block"
        S[1], S[pos_hf] = S[pos_hf], S[1]
    end
    return S
end

# ---------------------------------------------------------------------------
# Connection enumeration.
#
# foreach_connection! walks every symmetry-allowed single and double excitation
# out of determinant `d` and calls emit(row, H_{row,col}) once per connection.
# The loops are the ones in add_offdiag_row_contrib! (restricted_matvec.jl)
# with the accumulation replaced by the callback, so the matrix elements and
# signs are the tested ones. Because the space is closed under exactly these
# excitations, every generated determinant must be found; a miss is a bug and
# is raised rather than skipped.
# ---------------------------------------------------------------------------
@inline function foreach_connection!(emit::F, gs::GreedyState{W}, env::QCEnv{W},
                                     d::Det{W}, IRREPS::Vector{Int},
                                     up_occ::Vector{Int}, dn_occ::Vector{Int}) where {F,W}
    n   = env.norb_tot
    frz = env.norbfrzn
    Su  = env.single_targets
    Dss = env.doub_same
    Dod = env.doub_diff

    occupied_orbitals!(up_occ, dn_occ, d, n, env.nup, env.ndn)

    @inline function lookup(d2)
        j = get(gs.det2idx, d2, 0)
        j == 0 && error("generated determinant outside the space: the space is not closed")
        return j
    end

    # singles, spin up
    @inbounds for t in 1:env.nup
        i = up_occ[t]; (i <= frz) && continue
        for a in Su[i]
            (a <= frz) && continue
            if !isocc_fast(d, a, env.om)
                d2  = flip2(d, i, a, env.om)
                sgn = single_excitation_sign(d, i, a)
                hij = single_excitation_value_from_occ(up_occ, dn_occ, i, a,
                                                       env.HCORE, env.eris1, n; sign = sgn)
                emit(lookup(d2), hij)
            end
        end
    end

    # singles, spin down
    @inbounds for t in 1:env.ndn
        i_so = dn_occ[t]; i = i_so - n; (i <= frz) && continue
        for a in Su[i]
            (a <= frz) && continue
            a_so = a + n
            if !isocc_fast(d, a_so, env.om)
                d2  = flip2(d, i_so, a_so, env.om)
                sgn = single_excitation_sign(d, i_so, a_so)
                hij = single_excitation_value_from_occ(up_occ, dn_occ, i_so, a_so,
                                                       env.HCORE, env.eris1, n; sign = sgn)
                emit(lookup(d2), hij)
            end
        end
    end

    # doubles, up-up
    @inbounds for p in 1:(env.nup - 1)
        i = up_occ[p]; (i <= frz) && continue
        iri = IRREPS[i]
        for q in (p + 1):env.nup
            j = up_occ[q]; (j <= frz) && continue
            for (a, b) in Dss[(iri ⊻ IRREPS[j]) + 1]
                (a <= frz || b <= frz) && continue
                if !isocc_fast(d, a, env.om) && !isocc_fast(d, b, env.om)
                    d2  = flip4(d, i, j, a, b, env.om)
                    sgn = double_excitation_sign_via_two_singles(d, i, j, a, b, env.om)
                    hij = double_excitation_value(i, j, a, b, env.eris1, n; sign = sgn)
                    emit(lookup(d2), hij)
                end
            end
        end
    end

    # doubles, down-down
    @inbounds for p in 1:(env.ndn - 1)
        i_so = dn_occ[p]; i = i_so - n; (i <= frz) && continue
        iri = IRREPS[i]
        for q in (p + 1):env.ndn
            j_so = dn_occ[q]; j = j_so - n; (j <= frz) && continue
            for (a, b) in Dss[(iri ⊻ IRREPS[j]) + 1]
                (a <= frz || b <= frz) && continue
                a_so = a + n; b_so = b + n
                if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                    d2  = flip4(d, i_so, j_so, a_so, b_so, env.om)
                    sgn = double_excitation_sign_via_two_singles(d, i_so, j_so, a_so, b_so, env.om)
                    hij = double_excitation_value(i_so, j_so, a_so, b_so, env.eris1, n; sign = sgn)
                    emit(lookup(d2), hij)
                end
            end
        end
    end

    # doubles, up-down
    @inbounds for p in 1:env.nup
        i = up_occ[p]; (i <= frz) && continue
        iri = IRREPS[i]
        for q in 1:env.ndn
            j_so = dn_occ[q]; j = j_so - n; (j <= frz) && continue
            for (a, b) in Dod[(iri ⊻ IRREPS[j]) + 1]
                (a <= frz || b <= frz) && continue
                a_so = a; b_so = b + n
                if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                    d2  = flip4(d, i, j_so, a_so, b_so, env.om)
                    sgn = double_excitation_sign_via_two_singles(d, i, j_so, a_so, b_so, env.om)
                    hij = double_excitation_value(i, j_so, a_so, b_so, env.eris1, n; sign = sgn)
                    emit(lookup(d2), hij)
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Two-pass CSC assembly of H.
#
# H is symmetric, so column j is exactly determinant j's generation and columns
# can be built independently by disjoint thread blocks. Pass 1 counts the
# entries per column, a prefix sum gives colptr, and pass 2 writes each column
# into its own slice. Nothing is materialized as a triplet list, so peak memory
# is the final matrix itself. The excitations out of a determinant produce
# pairwise distinct determinants, so no summing of duplicates is needed; this
# is checked in pass 2 by verifying that the sorted row indices of each column
# are strictly increasing.
# ---------------------------------------------------------------------------
function _chunks(nS::Int, nt::Int)
    base, rem = divrem(nS, nt)
    out = Vector{UnitRange{Int}}(undef, nt)
    s = 1
    for c in 1:nt
        len = base + (c <= rem ? 1 : 0)
        out[c] = s:(s + len - 1)
        s += len
    end
    return out
end

# Threaded replacement for build_hdiag!, with progress. The serial version is
# otherwise several silent minutes at this size, and the loop parallelizes with
# no interaction between iterations.
function build_hdiag_progress!(gs::GreedyState{W}, env::QCEnv{W}) where {W}
    nS = length(gs.S)
    resize!(gs.hdiag, nS)
    nt = max(1, Threads.nthreads())
    chunks = _chunks(nS, nt)
    p = Progress("hdiag", nS; every = 15.0)
    @sync for c in 1:nt
        Threads.@spawn begin
            up_occ = Vector{Int}(undef, env.nup)
            dn_occ = Vector{Int}(undef, env.ndn)
            k = 0
            @inbounds for i in chunks[c]
                gs.hdiag[i] = diagonal_value!(up_occ, dn_occ, gs.S[i],
                                              env.HCORE_DIAG, env.J, env.K,
                                              env.norb_tot, env.nup, env.ndn)
                k += 1
                if k == 65536
                    bump!(p, k; report = (c == 1))
                    k = 0
                end
            end
            bump!(p, k; report = (c == 1))
        end
    end
    finish!(p)
    return nothing
end

function build_full_H(gs::GreedyState{W}, env::QCEnv{W}, IRREPS::Vector{Int}) where {W}
    nS = length(gs.S)
    nt = max(1, Threads.nthreads())
    chunks = _chunks(nS, nt)

    colcount = Vector{Int}(undef, nS)

    pc = Progress("count", nS; every = 30.0)
    t1 = @elapsed @sync for c in 1:nt
        Threads.@spawn begin
            up_occ = Vector{Int}(undef, env.nup)
            dn_occ = Vector{Int}(undef, env.ndn)
            cnt = Ref(0)
            k = 0
            @inbounds for col in chunks[c]
                cnt[] = 0
                foreach_connection!((j, v) -> (cnt[] += 1; nothing),
                                    gs, env, gs.S[col], IRREPS, up_occ, dn_occ)
                colcount[col] = cnt[] + 1          # + diagonal
                k += 1
                if k == 8192
                    bump!(pc, k; report = (c == 1))
                    k = 0
                end
            end
            bump!(pc, k; report = (c == 1))
        end
    end
    finish!(pc)

    colptr = Vector{Int}(undef, nS + 1)
    colptr[1] = 1
    @inbounds for j in 1:nS
        colptr[j + 1] = colptr[j] + colcount[j]
    end
    nnz_total = colptr[nS + 1] - 1
    say(@sprintf("counting pass %.1fs, nnz = %d, %.1f per column, matrix will need %.1f GB",
                 t1, nnz_total, nnz_total / nS, 16 * nnz_total / 2^30))
    require_memory(16 * nnz_total / 2^30, "the sparse matrix")

    rowval = Vector{Int}(undef, nnz_total)
    nzval  = Vector{Float64}(undef, nnz_total)
    say(@sprintf("allocated, rss %.1f GB", rss_gb()))

    pf = Progress("fill", nS; every = 30.0)
    t2 = @elapsed @sync for c in 1:nt
        Threads.@spawn begin
            up_occ = Vector{Int}(undef, env.nup)
            dn_occ = Vector{Int}(undef, env.ndn)
            rows = Int[];  vals = Float64[];  perm = Int[]
            k = 0
            @inbounds for col in chunks[c]
                empty!(rows); empty!(vals)
                push!(rows, col); push!(vals, gs.hdiag[col])
                foreach_connection!((j, v) -> (push!(rows, j); push!(vals, v); nothing),
                                    gs, env, gs.S[col], IRREPS, up_occ, dn_occ)
                m = length(rows)
                @assert m == colcount[col] "pass 1 and pass 2 disagree at column $col"
                resize!(perm, m); sortperm!(perm, rows)
                pos = colptr[col]
                prev = 0
                for t in 1:m
                    r = rows[perm[t]]
                    @assert r > prev "duplicate row $r in column $col"
                    prev = r
                    rowval[pos] = r
                    nzval[pos]  = vals[perm[t]]
                    pos += 1
                end
                k += 1
                if k == 8192
                    bump!(pf, k; report = (c == 1))
                    k = 0
                end
            end
            bump!(pf, k; report = (c == 1))
        end
    end
    finish!(pf)

    return SparseMatrixCSC{Float64,Int}(nS, nS, colptr, rowval, nzval)
end

# ---------------------------------------------------------------------------
# Memory preflight.
#
# Sys.free_memory reports the host, which is wrong under a cgroup, so the
# available figure is taken as the smaller of MemAvailable and whatever the
# cgroup still allows. MemAvailable already excludes memory other users hold
# and counts reclaimable cache as available, which is the right notion here.
# ---------------------------------------------------------------------------
function cgroup_available_bytes()
    lim = typemax(Int)
    for (limf, usef) in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                          "/sys/fs/cgroup/memory/memory.usage_in_bytes"))
        if isfile(limf)
            s = strip(read(limf, String))
            s == "max" && continue
            v = tryparse(Int, s)
            v === nothing && continue
            v > 2^60 && continue                 # the "no limit" sentinel
            used = isfile(usef) ? something(tryparse(Int, strip(read(usef, String))), 0) : 0
            lim = min(lim, v - used)
        end
    end
    return lim
end

function available_gb()
    host = typemax(Int)
    if isfile("/proc/meminfo")
        for line in eachline("/proc/meminfo")
            if startswith(line, "MemAvailable:")
                host = parse(Int, split(line)[2]) * 1024
                break
            end
        end
    end
    host == typemax(Int) && (host = Int(Sys.free_memory()))
    return min(host, cgroup_available_bytes()) / 2^30
end

# Abort before allocating rather than after, since an out-of-memory kill in the
# middle of the fill pass costs the whole assembly.
function require_memory(need_gb::Real, what::AbstractString; margin::Real = 1.15)
    avail = available_gb()
    say(@sprintf("memory check for %s: need %.1f GB, %.1f GB available, %.1f GB already resident",
                 what, need_gb, avail, rss_gb()))
    if avail < margin * need_gb
        error(@sprintf("not enough memory for %s: %.1f GB available against %.1f GB needed",
                       what, avail, margin * need_gb))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Threaded symmetric matrix-vector product.
#
# mul! for a SparseMatrixCSC is serial, which leaves 63 of 64 cores idle
# against 91 GB of memory traffic. H is symmetric and its row indices are
# sorted, so its CSC arrays are simultaneously its CSR arrays and column j can
# be read as row j. The product then becomes an independent gather per row with
# no write conflicts, so it threads with no reduction and no locking.
#
# The arrays are shared with H rather than copied, so this costs nothing in
# memory. Rows are partitioned by nonzero count rather than by count of rows,
# and the partition follows the column blocks used during the fill pass, so on
# a NUMA node most of the nonzeros a thread reads are the ones it wrote.
# ---------------------------------------------------------------------------
struct ThreadedSym{Tv}
    n::Int
    colptr::Vector{Int}
    rowval::Vector{Int}
    nzval::Vector{Tv}
    ranges::Vector{UnitRange{Int}}
end

function ThreadedSym(H::SparseMatrixCSC{Tv,Int}) where {Tv}
    n  = size(H, 1)
    nt = max(1, Threads.nthreads())
    cp = H.colptr
    nnz_total = cp[n + 1] - 1
    ranges = UnitRange{Int}[]
    lo = 1
    for c in 1:nt
        target = cp[1] + div(nnz_total * c, nt)
        hi = c == nt ? n : max(lo, searchsortedfirst(cp, target) - 1)
        hi = min(hi, n)
        push!(ranges, lo:hi)
        lo = hi + 1
        lo > n && break
    end
    return ThreadedSym{Tv}(n, cp, H.rowval, H.nzval, ranges)
end

Base.size(A::ThreadedSym) = (A.n, A.n)
Base.size(A::ThreadedSym, i::Int) = A.n
Base.eltype(::ThreadedSym{Tv}) where {Tv} = Tv

# ARPACK passes views into its own workspace rather than Vectors, so these take
# AbstractVector. The five-argument form is what LinearAlgebra falls back to,
# so it is the one that has to exist; the three-argument form defers to it.
function LinearAlgebra.mul!(y::AbstractVector, A::ThreadedSym, x::AbstractVector,
                            α::Number, β::Number)
    length(x) == A.n && length(y) == A.n ||
        throw(DimensionMismatch("ThreadedSym applied to a vector of the wrong length"))
    a = convert(eltype(A), α)
    b = convert(eltype(A), β)
    @sync for r in A.ranges
        Threads.@spawn @inbounds for i in r
            s = zero(eltype(A))
            for p in A.colptr[i]:(A.colptr[i + 1] - 1)
                s += A.nzval[p] * x[A.rowval[p]]
            end
            y[i] = iszero(b) ? a * s : a * s + b * y[i]
        end
    end
    return y
end

LinearAlgebra.mul!(y::AbstractVector, A::ThreadedSym, x::AbstractVector) =
    mul!(y, A, x, true, false)

Base.:*(A::ThreadedSym, x::AbstractVector) = mul!(similar(x, eltype(A), A.n), A, x)

# ---------------------------------------------------------------------------
# Check the assembled matrix against the matrix-free product the repository
# already trusts. mul_A_threaded! applies -eps*H, and env.epsilon is 1.0 here,
# so the reference is -H*x.
# ---------------------------------------------------------------------------
function verify_against_matvec(H, gs::GreedyState{W}, env::QCEnv{W},
                               IRREPS::Vector{Int}; ntrials::Int = 3) where {W}
    nS = size(H, 1)
    bf = build_bloom(gs)
    scratch = make_matvec_scratch(env)
    y = Vector{Float64}(undef, nS)
    rng = MersenneTwister(0)
    for _ in 1:ntrials
        x = randn(rng, nS)
        mul_A_threaded!(y, x, gs, env, IRREPS, scratch, bf)
        err = norm(H * x + y) / norm(y)
        say(@sprintf("assembly check, relative error %.3e", err))
        @assert err < 1e-10
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
function main(; norb_tot = 23, nelec_tot = 10, norbfrzn = 1,
                datadir = joinpath(NEON, "input_data"),
                outfile = joinpath(NEON, "results", "full_fci_neon.json"),
                vecfile = joinpath(NEON, "results", "full_fci_neon_v1.bin"),
                heartbeat = joinpath(NEON, "results", "full_fci_neon.progress"),
                sym_key = 0, nev_low = 3, ncv = 150, tol_lo = 1e-9, tol_hi = 1e-6,
                verify = true)

    mkpath(dirname(heartbeat))
    HEARTBEAT[] = heartbeat
    say("full FCI Neon: norb=$norb_tot nelec=$nelec_tot nfrzn=$norbfrzn sym_key=$sym_key")
    say("data from $datadir")

    env, IRREPS = build_env_plain(norb_tot, nelec_tot, norbfrzn,
                                  joinpath(datadir, "eris.txt"),
                                  joinpath(datadir, "hcore.txt"),
                                  joinpath(datadir, "symm.txt"))

    t_space = @elapsed S = build_full_space(env, IRREPS; sym_key = sym_key)
    nS = length(S)
    say(@sprintf("space built in %s, n = %d, rss %.1f GB", fmt_time(t_space), nS, rss_gb()))

    t_map = @elapsed gs = make_gs_from_S(S)
    say(@sprintf("determinant index built in %s, rss %.1f GB", fmt_time(t_map), rss_gb()))

    build_hdiag_progress!(gs, env)

    H = build_full_H(gs, env, IRREPS)
    Hs = ThreadedSym(H)
    verify && verify_against_matvec(Hs, gs, env, IRREPS)

    # Start from Hartree-Fock, which is index 1 when sym_key == 0.
    v0 = zeros(nS); v0[1] = 1.0

    # lambda_1, lambda_2: algebraically smallest eigenvalues, the ground and
    # first excited states within this block. The operator is wrapped so that
    # every ARPACK product prints, since eigs itself is silent. ARPACK accepts
    # a Ritz pair when its residual falls below tol times the Ritz value, and
    # the Ritz values here are of order 1e2, so tol_lo = 1e-9 asks for a
    # residual near 1e-7.
    #
    # ncv is the Krylov basis size per restart. The relative separation at the
    # bottom of this spectrum is small, of order the gap over the spectral
    # range, so a short basis restarts often and wastes products rebuilding
    # information it just discarded. A basis of 150 costs 150 * n * 8 bytes,
    # about 8 GB, which is cheap next to the 91 GB of the matrix.
    Hlo, _ = counting_operator(Hs, "eigs SR")
    require_memory(2 * ncv * nS * 8 / 2^30, "the Krylov basis at ncv = $ncv")
    t_lo = @elapsed λlo, Vlo, nconv_lo, niter_lo, nmult_lo, _ =
        eigs(Hlo; nev = nev_low, ncv = ncv, which = :SR, v0 = v0,
             tol = tol_lo, maxiter = 3000)
    λlo = real.(λlo); Vlo = real.(Vlo)
    perm = sortperm(λlo); λlo = λlo[perm]; Vlo = Vlo[:, perm]
    say(@sprintf("low end done in %s, nconv %d, %d products, lambda = %s",
                 fmt_time(t_lo), nconv_lo, nmult_lo, string(λlo)))

    # lambda_n: algebraically largest eigenvalue of this block. Only its
    # magnitude matters downstream, so it is asked for far less precisely.
    Hhi, _ = counting_operator(Hs, "eigs LR")
    t_hi = @elapsed λhi, _, nconv_hi, _, nmult_hi, _ =
        eigs(Hhi; nev = 1, ncv = ncv, which = :LR, tol = tol_hi, maxiter = 3000)
    λhi = real(λhi[1])
    say(@sprintf("high end done in %s, nconv %d, %d products, lambda_n = %.10f",
                 fmt_time(t_hi), nconv_hi, nmult_hi, λhi))

    # Achieved residuals, so the accuracy is measured rather than assumed. For
    # a symmetric matrix the Ritz value error is bounded by r^2 over the gap
    # and the eigenvector angle by r over the gap, so r is the quantity that
    # decides whether the stored eigenvector is accurate enough to serve as a
    # reference in the overlap plots.
    for j in 1:size(Vlo, 2)
        r = norm(Hs * Vlo[:, j] - λlo[j] * Vlo[:, j])
        say(@sprintf("residual for lambda_%d = %.3e", j, r))
    end

    v1 = Vlo[:, 1]
    v1 ./= norm(v1)
    if v1[1] < 0
        v1 .= -v1                 # fix the sign so the HF coefficient is positive
    end

    say(@sprintf("lambda_1 = %.12f", λlo[1]))
    say(@sprintf("lambda_2 = %.12f", λlo[2]))
    say(@sprintf("lambda_n = %.12f", λhi))
    say(@sprintf("gap = %.12f, HF weight = %.6f", λlo[2] - λlo[1], v1[1]^2))

    mkpath(dirname(outfile))
    open(outfile, "w") do io
        JSON.print(io, Dict(
            "norb_tot" => norb_tot, "nelec_tot" => nelec_tot, "norbfrzn" => norbfrzn,
            "sym_key" => sym_key, "n" => nS, "nnz" => nnz(H),
            "lambda_low" => λlo, "lambda_n" => λhi,
            "tol_lo" => tol_lo, "tol_hi" => tol_hi,
            "residual_1" => norm(Hs * v1 - λlo[1] * v1),
            "ncv" => ncv,
            "hf_coefficient" => v1[1],
        ))
    end

    # Eigenvector and its determinants, as raw binaries: v1 as Float64, the
    # determinant words as UInt64 in the same order.
    open(vecfile, "w") do io
        write(io, v1)
    end
    open(replace(vecfile, ".bin" => "_dets.bin"), "w") do io
        for d in S, w in d.words
            write(io, w)
        end
    end
    say("wrote $outfile and $vecfile")

    return H, gs, λlo, λhi, v1
end

# ---------------------------------------------------------------------------
# Matrix-free alternative, for when the 100 GB of the assembled matrix is not
# available. The same space is built, but H is applied through the repository's
# own restricted matvec, regenerating every connection on each product. No
# assembly, no storage beyond the determinant list, at the cost of a much more
# expensive matrix-vector product.
#
#   env, IRREPS = build_env_plain(23, 10, 1, eri, hcore, symm)
#   S  = build_full_space(env, IRREPS)
#   gs = make_gs_from_S(S); build_hdiag!(gs, env)
#   bf = build_bloom(gs); scr = make_matvec_scratch(env)
#   nS = length(S)
#   negH = LinearMap{Float64}((y, x) -> mul_A_threaded!(y, x, gs, env, IRREPS, scr, bf),
#                             nS, nS; ismutating = true, issymmetric = true)
#   # negH applies -H, so its algebraically largest eigenvalues are -lambda_1,
#   # -lambda_2, and its smallest is -lambda_n.
#   λ, V = eigs(negH; nev = 4, which = :LR, tol = 1e-10)
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end