# unsparsified_helpers.jl
# functions only used by the unsparsified version of my code (run_unsparsified.jl).


# `next_frontier` is supplied by the caller (the unsparsified driver owns the BFS
# frontier state); a newly-inserted determinant's index is appended to it.
@inline function insert_det!(gs::GreedyState{W}, d::Det{W}, next_frontier::Vector{Int})::Int where {W}
    idx = get(gs.det2idx, d, 0)
    if idx == 0
        idx = length(gs.S) + 1
        push!(gs.S, d)
        gs.det2idx[d] = idx
        push!(next_frontier, idx)
    end
    return idx
end

# Unsparsified bag expansion
# Naming convention: i,j are occupied spatial orbitals; a,b are unoccupied spatial orbitals; i_so, j_so, a_so, b_so are spin-orbital indices (1..2n)
function expand_phase1!(
    gs::GreedyState{W},
    env::QCEnv{W},
    IRREPS::Vector{Int},      # must match the irreps used to build doub_* buckets
    norbfrzn::Int,            # frozen spatial orbitals are 1:norbfrzn
    max_dets::Int,
    up_occ::Vector{Int},      # scratch (length nup), spatial indices 1..n
    dn_occ::Vector{Int},      # scratch (length ndn), spin-orb indices (n+1..2n)
    frontier::Vector{Int},        # indices to expand this round (owned by the caller)
    next_frontier::Vector{Int}    # filled with indices of dets newly added this round
) where {W}

    empty!(next_frontier)

    n   = env.norb_tot
    frz = norbfrzn

    Su  = env.single_targets
    Dss = env.doub_same
    Dod = env.doub_diff

    @inbounds for idx in frontier
        d = gs.S[idx]

        # fill occ lists once per parent
        occupied_orbitals!(up_occ, dn_occ, d, n, env.nup, env.ndn)

        # --------------------------
        # Singles: up-spin
        # --------------------------
        for t in 1:env.nup
            i = up_occ[t]            # spatial
            (i <= frz) && continue   # never remove from frozen
            i_so = i                 # up spin-orb

            targets = Su[i]
            for u in 1:length(targets)
                a = targets[u]       # spatial
                (a <= frz) && continue
                a_so = a             # up spin-orb

                if !isocc_fast(d, a_so, env.om)
                    d2 = flip2(d,i_so, a_so, env.om)
                    insert_det!(gs, d2, next_frontier)
                    (length(gs.S) >= max_dets) && return
                end
            end
        end

        # --------------------------
        # Singles: down-spin
        # --------------------------
        for t in 1:env.ndn
            i_so = dn_occ[t]         # down spin-orb (n+1..2n)
            i = i_so - n             # spatial
            (i <= frz) && continue
            targets = Su[i]

            for u in 1:length(targets)
                a = targets[u]       # spatial
                (a <= frz) && continue
                a_so = a + n         # down spin-orb

                if !isocc_fast(d, a_so, env.om)
                    d2 = flip2(d, i_so, a_so, env.om)
                    insert_det!(gs, d2, next_frontier)
                    (length(gs.S) >= max_dets) && return
                end
            end
        end

        # --------------------------
        # Doubles: up-up (same spin)
        # --------------------------
        for p in 1:(env.nup - 1)
            i = up_occ[p]            # spatial
            (i <= frz) && continue
            i_so = i
            iri = IRREPS[i]

            for q in (p + 1):env.nup
                j = up_occ[q]        # spatial
                (j <= frz) && continue
                j_so = j

                key  = iri ⊻ IRREPS[j]
                cand = Dss[key + 1]  # (a,b) spatial, a<b

                for u in 1:length(cand)
                    a, b = cand[u]
                    (a <= frz || b <= frz) && continue

                    a_so = a
                    b_so = b
                    if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                        d2 = flip4(d, i_so, j_so, a_so, b_so, env.om)
                        insert_det!(gs, d2, next_frontier)
                        (length(gs.S) >= max_dets) && return
                    end
                end
            end
        end

        # --------------------------
        # Doubles: down-down (same spin)
        # --------------------------
        for p in 1:(env.ndn - 1)
            i_so = dn_occ[p]
            i = i_so - n
            (i <= frz) && continue
            iri = IRREPS[i]

            for q in (p + 1):env.ndn
                j_so = dn_occ[q]
                j = j_so - n
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
                        insert_det!(gs, d2, next_frontier)
                        (length(gs.S) >= max_dets) && return
                    end
                end
            end
        end

        # --------------------------
        # Doubles: up-down (opposite spin)
        # interpret cand (a,b) as: a is up target, b is down target
        # --------------------------
        for p in 1:env.nup
            i = up_occ[p]            # spatial
            (i <= frz) && continue
            i_so = i
            iri = IRREPS[i]

            for q in 1:env.ndn
                j_so = dn_occ[q]
                j = j_so - n
                (j <= frz) && continue

                key  = iri ⊻ IRREPS[j]
                cand = Dod[key + 1]  # (a,b) spatial (a up, b down), includes a==b

                for u in 1:length(cand)
                    a, b = cand[u]
                    (a <= frz || b <= frz) && continue

                    a_so = a
                    b_so = b + n
                    if !isocc_fast(d, a_so, env.om) && !isocc_fast(d, b_so, env.om)
                        d2 = flip4(d, i_so, j_so, a_so, b_so, env.om)
                        insert_det!(gs, d2, next_frontier)
                        (length(gs.S) >= max_dets) && return
                    end
                end
            end
        end
    end

    return nothing
end