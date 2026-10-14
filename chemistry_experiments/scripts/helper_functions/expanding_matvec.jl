# ============================================================================
# expanding_matvec.jl: matrix-free y = A * x, without support restricted to S
#
# For each source determinant d\in S with coefficient x_d, we find all
# neighbors d' of d (single and double excitations), compute the matrix elements
# H_{d,d'} on the fly, and add the contribution to 
#                  y[d'] += (-ε·H_{d,d'}) · x_d
# and also add the diagonal contribution 
#                  y[d]  += (1 − ε·H_{dd}) · x_d
# into a dictionary y: Det{W} => Float64.
#
# Parallelization: 
# 
# Split the set of all possible target determinants into P shards via a hash
# function owner_of(d') which assigns each d' to a number 1,...,P via
#               owner_of(d') = hash(d') mod P + 1
# where hash takes the determinant's bitstring and produces a UInt hash value. 
# For each shard, we define an accumulator dictionary (RobinDict y_det->y_val) and a lock.
# The thread will fill the dictionary with dets and sum up the constributions to those dets. 
# The lock only lets one thread access the shard accumulator at a time.
#
# Each thread gets a set of source determinants S_i to process, and emits contributions
# of the form (d', A_{d,d'} * x_d, owner_of(d')) into a thread-local UpdateBuffer.
# When a buffer fills, it gets flushed with flush! to drain it into the shards. 
# flush! first sorts the buffered contributions by owner shard and then applies
# each shard's batch under a single lock.
#
# Once all sources are processed and buffers flushed, the P shards together hold the 
# full vector y = Ax. If you only need the top k entries of y, you can extract those 
# via topk_from_shards, which iterates through all P shards and maintains a bounded
# min-heap of the top k entries seen so far. 
# ============================================================================


# Set DEBUG_FLUSH = false to strip flush! integrity diagnostics
const DEBUG_FLUSH = true

using LinearAlgebra
using DataStructures
using DataStructures: RobinDict
using Base.Threads

# --------------------------------------------------------------------------------------------
# Threading helpers
#
# Generally I set P = 8 * Threads.nthreads()
# --------------------------------------------------------------------------------------------

@inline function owner_of(d::Det{W}, P::Int) where {W} # which shard owns this determinant?
    return Int32(rem(hash(d), UInt(P)) + 1)
end

mutable struct UpdateBuffer{W} # buffer of thread-local pending updates (d', Δ, owner) to be flushed into shards
    # pending updates (det, val, owner) for up to capacity entries
    owners::Vector{Int32}
    dets::Vector{Det{W}}
    vals::Vector{Float64}
    len::Int32 # number of pending entries currently in the buffer; must be <= capacity

    # Used during flush to track which owners are present
    seen_stamp::Vector{Int32}   # list of buffers, length = P; seen_stamp[o] == stamp if owner o is present in current buffer
    stamp::Int32 # number of flushes in the life of this buffer
    owners_list::Vector{Int32}  # list of unique owners in current buffer, length = P
    nowners::Int32 # number of unique owners in current buffer (so owners_list[1:nowners] are valid)

    # scratch space for flush sorting; length = capacity
    perm::Vector{Int32}         

    # stats
    npush::Int64 # total number of pushes into this buffer
    nflush::Int64 # total number of flushes performed on this buffer
end
function UpdateBuffer{W}(P::Integer; capacity::Integer=4096) where {W}
    P = Int(P)
    capacity = Int(capacity)

    capacity <= typemax(Int32) || throw(ArgumentError("capacity too large for Int32"))
    P        <= typemax(Int32) || throw(ArgumentError("P too large for Int32"))
    capacity > 0               || throw(ArgumentError("capacity must be positive"))
    P > 0                      || throw(ArgumentError("P must be positive"))

    owners      = Vector{Int32}(undef, capacity)
    dets        = Vector{Det{W}}(undef, capacity)
    vals        = Vector{Float64}(undef, capacity)
    seen_stamp  = zeros(Int32, P)
    owners_list = Vector{Int32}(undef, P)
    perm        = Vector{Int32}(undef, capacity)

    return UpdateBuffer{W}(owners, dets, vals, Int32(0),
                           seen_stamp, Int32(0), owners_list, Int32(0),
                           perm,
                           0, 0)
end

# add to the UpdateBuffer, flushing if needed; owner is the shard index (1-based)
@inline function buf_push!(buf::UpdateBuffer{W}, det::Det{W}, val::Float64, owner::Int32,
                           acc_shards::Vector{RobinDict{Det{W},Float64}}, locks::Vector{SpinLock}) where {W}
    # Ensure there is space for one more element
    if buf.len + 1 > length(buf.owners)
        flush!(buf, acc_shards, locks)   # must reset buf.len to 0
    end
    # After flush, we MUST have space
    @assert buf.len < length(buf.owners)

    buf.len += 1
    @inbounds buf.owners[buf.len] = owner
    @inbounds buf.dets[buf.len]   = det
    @inbounds buf.vals[buf.len]   = val
    return nothing
end
@inline function buf_push!(buf::UpdateBuffer{W},
                           d::Det{W},
                           Δ::Float64,
                           owner::Integer,
                           acc_shards::Vector{RobinDict{Det{W},Float64}},
                           locks::Vector{Threads.SpinLock}) where {W}
    return buf_push!(buf, d, Δ, Int32(owner), acc_shards, locks)
end

@inline function next_stamp!(buf) # add one to the stamp, and reset seen_stamp if we wrap around
    buf.stamp += 1
    if buf.stamp == typemax(Int32)
        fill!(buf.seen_stamp, 0)
        buf.stamp = 1
    end
    return buf.stamp
end

function flush!(buf::UpdateBuffer{W},
                acc_shards::Vector{RobinDict{Det{W},Float64}},
                locks::Vector{Threads.SpinLock}) where {W}

    buf.nflush += 1
    L = Int(buf.len)

    # DEFENSIVE CHECK
    capacity = length(buf.owners)
    if L > capacity
        error("CORRUPTION: buf.len=$L exceeds capacity=$capacity")
    end

    L == 0 && return nothing

    P = length(acc_shards)

    @boundscheck length(locks) == P || throw(DimensionMismatch("locks length != P"))
    @boundscheck length(buf.seen_stamp) == P || throw(DimensionMismatch("seen_stamp length != P"))

    # IMPORTANT:
    # owners_list stores the UNIQUE owners encountered, so it must be able to hold up to P.
    # If P can exceed buffer capacity, owners_list must be at least P (or you must cap P).
    @boundscheck length(buf.owners_list) >= P || throw(DimensionMismatch("owners_list too small: need >= P=$P, got $(length(buf.owners_list))"))

    # perm stores one entry per buffered update => must be >= L
    @boundscheck length(buf.perm) >= L || throw(DimensionMismatch("perm length < L: need >= L=$L, got $(length(buf.perm))"))

    # owners/dets/vals are capacity-sized; should be >= L
    @boundscheck length(buf.owners) >= L || throw(DimensionMismatch("owners length < L"))
    @boundscheck length(buf.dets)   >= L || throw(DimensionMismatch("dets length < L"))
    @boundscheck length(buf.vals)   >= L || throw(DimensionMismatch("vals length < L"))

    # ------------------------------------------------------------
    # Pass 1: collect unique owners present (owners_list) using stamp
    # ------------------------------------------------------------
    s = next_stamp!(buf)
    buf.nowners = 0
    
    fill!(buf.seen_stamp, 0)

    @inbounds for i in 1:L
        o32 = buf.owners[i]           # Int32 expected in 1..P
        o   = Int(o32)
        DEBUG_FLUSH && ((1 <= o <= P) || error("Bad owner o=$o (raw=$o32) at i=$i, L=$L, len=$(buf.len), P=$P"))

        if buf.seen_stamp[o] != s
            buf.seen_stamp[o] = s
            n = Int(buf.nowners) + 1
            if n > length(buf.owners_list)
                error("owners_list overflow: n=$n, capacity=$(length(buf.owners_list)), P=$P, L=$L")
            end
            buf.nowners = Int32(n)
            buf.owners_list[n] = o32
        end
    end

    nowners = Int(buf.nowners)

    # ------------------------------------------------------------
    # Pass 2: counts per owner (reuse seen_stamp for counts)
    # ------------------------------------------------------------
    @inbounds for ui in 1:nowners
        o = Int(buf.owners_list[ui])
        buf.seen_stamp[o] = 0
    end
    @inbounds for i in 1:L
        o = Int(buf.owners[i])
        if DEBUG_FLUSH && (o < 1 || o > P)
            error("INVALID OWNER: buf.owners[$i]=$o, valid range is [1,$P]")
        end
        buf.seen_stamp[o] += 1
    end

    if DEBUG_FLUSH
    # DIAGNOSTIC: verify counts sum to L
    total = 0
    @inbounds for ui in 1:nowners
        o = Int(buf.owners_list[ui])
        total += buf.seen_stamp[o]
    end
    if total != L
        # Find which owners are missing
        println("COUNTING BUG: counted=$total, L=$L, missing=$(L-total)")
        println("nowners=$(buf.nowners), P=$P")
        
        # Check: are there items with owners NOT in owners_list?
        owners_set = Set(Int(buf.owners_list[ui]) for ui in 1:nowners)
        bad_count = 0
        for i in 1:min(20, L)  # check first 20
            o = Int(buf.owners[i])
            if !(o in owners_set)
                bad_count += 1
                println("  buf.owners[$i]=$o NOT in owners_list")
            end
        end
        
        error("COUNTING BUG: counted total=$total items, but L=$L")
    end
    end

    # ------------------------------------------------------------
    # Pass 3: prefix sums -> write cursors (stored in seen_stamp)
    # ------------------------------------------------------------
    off = 1
    @inbounds for ui in 1:nowners
        o = Int(buf.owners_list[ui])
        c = Int(buf.seen_stamp[o])     # count
        buf.seen_stamp[o] = off        # start offset (1-based)
        off += c
    end
    # now seen_stamp[o] is write cursor
    # VALIDATE: prefix sum should total to L
    if DEBUG_FLUSH && (off != L + 1)
        error("PREFIX SUM BUG: off=$off, expected L+1=$(L+1). Counts don't match actual items!")
    end

    # ------------------------------------------------------------
    # Pass 4: fill perm (stable bucketing by owner)
    # ------------------------------------------------------------
    perm = buf.perm
    @inbounds for i in 1:L
        o = Int(buf.owners[i])
        pos = Int(buf.seen_stamp[o])   # current cursor in 1..L
        @boundscheck 1 <= pos <= L || error("perm write cursor out of range: pos=$pos, L=$L")
        perm[pos] = Int32(i)           # store source index i
        buf.seen_stamp[o] = pos + 1    # advance cursor
    end

    # ------------------------------------------------------------
    # Pass 5: restore starts for iteration (recompute starts)
    # ------------------------------------------------------------
    off = 1
    @inbounds for ui in 1:nowners
        o = Int(buf.owners_list[ui])
        # cursor ended at start+count, so count = cursor - off
        c = Int(buf.seen_stamp[o]) - off
        buf.seen_stamp[o] = off
        off += c
    end

    # ------------------------------------------------------------
    # Pass 6: apply updates shard-by-shard (one lock per owner)
    # ------------------------------------------------------------
    @inbounds for ui in 1:nowners
        o32 = buf.owners_list[ui]
        o   = Int(o32)

        start = Int(buf.seen_stamp[o])
        stop  = if ui == nowners
            L
        else
            Int(buf.seen_stamp[Int(buf.owners_list[ui + 1])]) - 1
        end

        l = locks[o]
        lock(l)
        shard = acc_shards[o]
        for t in start:stop
            i = Int(perm[t])
            d = buf.dets[i]
            Δ = buf.vals[i]
            shard[d] = get(shard, d, 0.0) + Δ
        end
        unlock(l)
    end

    buf.len = 0
    return nothing
end


# --------------------------------------------------------------------------------------------
# Matrix-free multiplication of the Hamiltonian with a vector, not restricted to a bag S 
# --------------------------------------------------------------------------------------------
# Off-diagonal row contribution:
# This enumerates all single+double neighbors of d, neighbor determinants d2
# For every neighbor d2, it adds the contribution to acc[d2] += -(epsilon) * H_{d, d2} * ci
# It does not add diagonal contributions.
function add_offdiag_emit_contrib!(
    buf::UpdateBuffer{W},
    d::Det{W},
    ci::Float64,
    env::QCEnv{W},
    IRREPS::Vector{Int},
    up_occ::Vector{Int},
    dn_occ::Vector{Int},
    P::Int,
    acc_shards::Vector{RobinDict{Det{W},Float64}},
    locks::Vector{Threads.SpinLock},
) where {W}

    n   = env.norb_tot
    frz = env.norbfrzn
    Su  = env.single_targets
    Dss = env.doub_same
    Dod = env.doub_diff
    epsilon = env.epsilon

    # Singles: up-spin
    @inbounds for t in 1:env.nup
        i = up_occ[t]
        (i <= frz) && continue
        i_so = i
        targets = Su[i]
        for u in 1:length(targets)
            a = targets[u]
            (a <= frz) && continue
            a_so = a
            if !isocc_fast(d, a_so, env.om)
                d2 = flip2(d, i_so, a_so, env.om)
                sgn = single_excitation_sign(d, i_so, a_so)
                hij = single_excitation_value_from_occ(
                    up_occ, dn_occ, i_so, a_so,
                    env.HCORE, env.eris1, n; sign=sgn
                )
                owner = owner_of(d2, P)
                buf_push!(buf, d2, (-epsilon)*hij*ci, owner, acc_shards, locks)
            end
        end
    end

    # Singles: down-spin
    @inbounds for t in 1:env.ndn
        i_so = dn_occ[t]
        i    = i_so - n
        (i <= frz) && continue
        targets = Su[i]
        for u in 1:length(targets)
            a = targets[u]
            (a <= frz) && continue
            a_so = a + n
            if !isocc_fast(d, a_so, env.om)
                d2 = flip2(d, i_so, a_so, env.om)
                sgn = single_excitation_sign(d, i_so, a_so)
                hij = single_excitation_value_from_occ(
                    up_occ, dn_occ, i_so, a_so,
                    env.HCORE, env.eris1, n; sign=sgn
                )
                owner = owner_of(d2, P)
                buf_push!(buf, d2, (-epsilon)*hij*ci, owner, acc_shards, locks)
            end
        end
    end

    # Doubles: up-up
    @inbounds for p in 1:(env.nup - 1)
        i = up_occ[p]
        (i <= frz) && continue
        i_so = i
        iri  = IRREPS[i]
        for q in (p + 1):env.nup
            j = up_occ[q]
            (j <= frz) && continue
            j_so = j

            key  = iri ⊻ IRREPS[j]
            cand = Dss[key + 1]
            for u in 1:length(cand)
                a, b = cand[u]
                (a <= frz || b <= frz) && continue
                a_so = a
                b_so = b
                if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                    d2 = flip4(d, i_so, j_so, a_so, b_so, env.om)
                    sgn = double_excitation_sign_via_two_singles(d, i_so, j_so, a_so, b_so, env.om)
                    hij = double_excitation_value(i_so, j_so, a_so, b_so, env.eris1, n; sign=sgn)
                    owner = owner_of(d2, P)
                    buf_push!(buf, d2, (-epsilon)*hij*ci, owner, acc_shards, locks)
                end
            end
        end
    end

    # Doubles: down-down
    @inbounds for p in 1:(env.ndn - 1)
        i_so = dn_occ[p]
        i    = i_so - n
        (i <= frz) && continue
        iri  = IRREPS[i]
        for q in (p + 1):env.ndn
            j_so = dn_occ[q]
            j    = j_so - n
            (j <= frz) && continue

            key  = iri ⊻ IRREPS[j]
            cand = Dss[key + 1]
            for u in 1:length(cand)
                a, b = cand[u]
                (a <= frz || b <= frz) && continue
                a_so = a + n
                b_so = b + n
                if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                    d2 = flip4(d, i_so, j_so, a_so, b_so, env.om)
                    sgn = double_excitation_sign_via_two_singles(d, i_so, j_so, a_so, b_so, env.om)
                    hij = double_excitation_value(i_so, j_so, a_so, b_so, env.eris1, n; sign=sgn)
                    owner = owner_of(d2, P)
                    buf_push!(buf, d2, (-epsilon)*hij*ci, owner, acc_shards, locks)
                end
            end
        end
    end

    # Doubles: up-down
    @inbounds for p in 1:env.nup
        i = up_occ[p]
        (i <= frz) && continue
        i_so = i
        iri  = IRREPS[i]
        for q in 1:env.ndn
            j_so = dn_occ[q]
            j    = j_so - n
            (j <= frz) && continue

            key  = iri ⊻ IRREPS[j]
            cand = Dod[key + 1]
            for u in 1:length(cand)
                a, b = cand[u]
                (a <= frz || b <= frz) && continue
                a_so = a
                b_so = b + n
                if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                    d2 = flip4(d, i_so, j_so, a_so, b_so, env.om)
                    sgn = double_excitation_sign_via_two_singles(d, i_so, j_so, a_so, b_so, env.om)
                    hij = double_excitation_value(i_so, j_so, a_so, b_so, env.eris1, n; sign=sgn)
                    owner = owner_of(d2, P)
                    buf_push!(buf, d2, (-epsilon)*hij*ci, owner, acc_shards, locks)
                end
            end
        end
    end

    return nothing
end


# Matrix-free multiplication of H with v over the full space of determinants S
# using threaded accumulation into P shards
# you can then extract the top k values and indices from the shareds using topk_from_shards
function mulA_expand!(
    acc_shards::Vector{RobinDict{Det{W},Float64}},  # length P
    locks::Vector{Threads.SpinLock},                # length P
    buf_tls::Vector{UpdateBuffer{W}},               # length = number of threads
    S::Vector{Det{W}},
    v::Vector{Float64},
    env::QCEnv{W},
    IRREPS::Vector{Int},
    scratch_tls::Vector{MatvecScratch},
) where {W}

    k  = length(S)
    @assert length(v) == k
    nt = Threads.maxthreadid()   
    @assert length(buf_tls) == nt
    @assert length(scratch_tls) == nt
    P  = length(acc_shards)
    @assert length(locks) == P

    # Clear shards
    @inbounds for o in 1:P
        empty!(acc_shards[o])
    end

    # Clear buffers
    @inbounds for tid in 1:nt
        b = buf_tls[tid]
        b.len = 0
        b.npush = 0
        b.nflush = 0
    end

    epsilon = env.epsilon
    n   = env.norb_tot

    Threads.@threads :static for i in 1:k
        tid = Threads.threadid()
        scr = scratch_tls[tid]
        buf = buf_tls[tid]

        ci = v[i]
        ci == 0.0 && continue
        d = S[i]

        # occupied lists
        occupied_orbitals!(scr.up_occ, scr.dn_occ, d, n, env.nup, env.ndn)

        # diagonal update: y(d) += (shift - eps*Hdd)*ci
        Hdd = diagonal_value_from_occ(scr.up_occ, scr.dn_occ,
                              env.HCORE_DIAG, env.J, env.K,
                              n, env.nup, env.ndn)
        owner = owner_of(d, P)
        buf_push!(buf, d, diag_A(env, Hdd)*ci, owner, acc_shards, locks)

        # off-diagonal updates
        add_offdiag_emit_contrib!(buf, d, ci, env, IRREPS,
                                  scr.up_occ, scr.dn_occ,
                                  P, acc_shards, locks)
    end

    # Final flush from all threads
    @inbounds for tid in 1:nt
        flush!(buf_tls[tid], acc_shards, locks)
    end

    return nothing
end

# --------------------------------------------------------------------------------------------
# Sparsification helpers
# --------------------------------------------------------------------------------------------
struct TopKItem{W}
    a::Float64        # abs(val)
    t::Int64          # tie-breaker
    d::Det{W}
    v::Float64
end

@inline Base.isless(x::TopKItem{W}, y::TopKItem{W}) where {W} =
    (x.a < y.a) || (x.a == y.a && x.t < y.t)

function topk_from_shards(acc_shards::Vector{RobinDict{Det{W},Float64}}, k::Int) where {W}
    k > 0 || throw(ArgumentError("k must be positive"))

    heap = BinaryMinHeap{TopKItem{W}}()
    tiebreak = Int64(0)

    for shard in acc_shards
        for (d, val) in shard
            a = abs(val)
            tiebreak += 1
            item = TopKItem{W}(a, tiebreak, d, val)

            if length(heap) < k
                push!(heap, item)
            else
                if a > first(heap).a
                    pop!(heap)
                    push!(heap, item)
                end

            end
        end
    end

    m = length(heap)
    m == 0 && return Det{W}[], Float64[]

    items = Vector{TopKItem{W}}(undef, m)
    @inbounds for i in 1:m
        items[i] = pop!(heap)   # ascending
    end
    reverse!(items)             # descending by abs

    dets = Vector{Det{W}}(undef, m)
    vals = Vector{Float64}(undef, m)
    @inbounds for i in 1:m
        dets[i] = items[i].d
        vals[i] = items[i].v
    end

    return dets, vals
end

function topk_from_shards_excluding(
    acc_shards::Vector{RobinDict{Det{W},Float64}}, 
    k::Int,
    exclude::Union{Set{Det{W}}, RobinDict{Det{W},<:Any}}
) where {W}
    k > 0 || throw(ArgumentError("k must be positive"))

    heap = BinaryMinHeap{TopKItem{W}}()
    tiebreak = Int64(0)

    for shard in acc_shards
        for (d, val) in shard
            haskey(exclude, d) && continue  # skip if already in S
            
            a = abs(val)
            tiebreak += 1
            item = TopKItem{W}(a, tiebreak, d, val)

            if length(heap) < k
                push!(heap, item)
            elseif a > first(heap).a
                pop!(heap)
                push!(heap, item)
            end
        end
    end

    m = length(heap)
    m == 0 && return Det{W}[], Float64[]

    items = Vector{TopKItem{W}}(undef, m)
    @inbounds for i in 1:m
        items[i] = pop!(heap)
    end
    reverse!(items)

    dets = Vector{Det{W}}(undef, m)
    vals = Vector{Float64}(undef, m)
    @inbounds for i in 1:m
        dets[i] = items[i].d
        vals[i] = items[i].v
    end

    return dets, vals
end

function normalize!(v::Vector{Float64})
    nrm2 = 0.0
    @inbounds @simd for i in eachindex(v)
        nrm2 += v[i]*v[i]
    end
    nrm = sqrt(nrm2)
    nrm == 0.0 && return 0.0
    @inbounds @simd for i in eachindex(v)
        v[i] /= nrm
    end
    return nrm
end

# Keep all of S (in order), then append any new dets from add that aren't already present.
function union_keep_order(S::Vector{Det{W}}, add::Vector{Det{W}}) where {W}
    seen = RobinDict{Det{W},Bool}()
    sizehint!(seen, length(S) + length(add))

    out = Det{W}[]
    sizehint!(out, length(S) + length(add))

    @inbounds for d in S
        if !haskey(seen, d)
            seen[d] = true
            push!(out, d)
        end
    end
    @inbounds for d in add
        if !haskey(seen, d)
            seen[d] = true
            push!(out, d)
        end
    end
    return out
end

# returns dets, vals including ALL entries from the shards, but 
# dets[1:k] are the top-k sroted by abs(val), and dets[k+1:end] are the rest in arbitrary order.
function sorted_from_shards(acc_shards::Vector{RobinDict{Det{W},Float64}}, k::Int) where {W}
    k > 0 || throw(ArgumentError("k must be positive"))

    ntot = 0
    for shard in acc_shards
        ntot += length(shard)
    end
    ntot == 0 && return Det{W}[], Float64[]

    kk   = min(k, ntot)
    dets = Vector{Det{W}}(undef, ntot)
    vals = Vector{Float64}(undef, ntot)

    heap = BinaryMinHeap{TopKItem{W}}()
    tail = ntot                      # tail write cursor, fills from the back
    tiebreak = Int64(0)

    for shard in acc_shards
        for (d, val) in shard
            tiebreak += 1
            item = TopKItem{W}(abs(val), tiebreak, d, val)
            if length(heap) < kk
                push!(heap, item)
            elseif item.a > first(heap).a       # bigger than current k-th largest
                evicted = pop!(heap)
                push!(heap, item)
                @inbounds dets[tail] = evicted.d
                @inbounds vals[tail] = evicted.v
                tail -= 1
            else                                # not in the top-k
                @inbounds dets[tail] = item.d
                @inbounds vals[tail] = item.v
                tail -= 1
            end
        end
    end
    @assert tail == kk                          # tail filled positions kk+1:ntot

    # Drain the heap: pops ascending, so write into kk:-1:1 to get descending.
    @inbounds for i in kk:-1:1
        it = pop!(heap)
        dets[i] = it.d
        vals[i] = it.v
    end

    return dets, vals
end
