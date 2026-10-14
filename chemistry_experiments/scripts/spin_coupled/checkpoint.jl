#=------------------------------------------------------------------------------
checkpoint.jl

Save and load full resume state for greedy / TPM ozone runs.

A checkpoint contains everything needed to resume an iteration loop:
  - iteration number completed
  - support  S :: Vector{Det{W}}
  - coefficients  v :: Vector{Float64}
  - eigval_history :: Vector{Float64}
  - bagsize_history :: Vector{Int}
  - iter_times :: Vector{Float64}
------------------------------------------------------------------------------=#

const CHECKPOINT_MAGIC = UInt32(0xC4EC4495)

"""
    save_checkpoint(path, iteration, S, v, eigval_history, bagsize_history,
                    iter_times)

Atomically write a checkpoint to `path`. The write goes to `path * ".tmp"`
first, then renames into place — so a crash mid-write leaves the previous
checkpoint intact.
"""
function save_checkpoint(
    path::String,
    iteration::Int,
    S::Vector{Det{W}},
    v::Vector{Float64},
    eigval_history::Vector{Float64},
    bagsize_history::Vector{Int},
    iter_times::Vector{Float64},
) where {W}
    @assert length(v) == length(S) "save_checkpoint: |v| ($(length(v))) != |S| ($(length(S)))"

    # Make sure parent directory exists
    parent = dirname(path)
    isdir(parent) || mkpath(parent)

    nS         = length(S)
    nh         = length(eigval_history)
    nb         = length(bagsize_history)
    nt         = length(iter_times)
    det_bytes  = nS * W * 8
    val_bytes  = nS * 8
    hist_bytes = nh * 8
    bag_bytes  = nb * 8
    time_bytes = nt * 8
    header_bytes = 4 + 4 + 4 + 8 + 8 + 8 + 8     # 7 fields, see below

    total_bytes = header_bytes + det_bytes + val_bytes +
                  hist_bytes + bag_bytes + time_bytes
    buf = Vector{UInt8}(undef, total_bytes)

    GC.@preserve buf begin
        base = pointer(buf)

        unsafe_store!(Ptr{UInt32}(base + 0),  CHECKPOINT_MAGIC)
        unsafe_store!(Ptr{UInt32}(base + 4),  UInt32(W))
        unsafe_store!(Ptr{UInt32}(base + 8),  UInt32(iteration))
        unsafe_store!(Ptr{UInt64}(base + 12), UInt64(nS))
        unsafe_store!(Ptr{UInt64}(base + 20), UInt64(nh))
        unsafe_store!(Ptr{UInt64}(base + 28), UInt64(nb))
        unsafe_store!(Ptr{UInt64}(base + 36), UInt64(nt))

        off = header_bytes

        # Pack S as W*8 bytes per det (NTuple{W,UInt64} laid out flat)
        @inbounds for i in 1:nS
            d = S[i]
            for k in 1:W
                unsafe_store!(Ptr{UInt64}(base + off + (k - 1) * 8), d.words[k])
            end
            off += W * 8
        end

        # Pack v
        @inbounds for i in 1:nS
            unsafe_store!(Ptr{Float64}(base + off), v[i])
            off += 8
        end

        # Pack histories
        @inbounds for i in 1:nh
            unsafe_store!(Ptr{Float64}(base + off), eigval_history[i])
            off += 8
        end
        @inbounds for i in 1:nb
            unsafe_store!(Ptr{Int64}(base + off), Int64(bagsize_history[i]))
            off += 8
        end
        @inbounds for i in 1:nt
            unsafe_store!(Ptr{Float64}(base + off), iter_times[i])
            off += 8
        end

        @assert off == total_bytes "Checkpoint pack offset mismatch: $off vs $total_bytes"
    end

    tmp = path * ".tmp"
    open(tmp, "w") do io
        write(io, buf)
    end
    mv(tmp, path; force = true)

    return total_bytes
end

"""
    load_checkpoint(path, ::Type{Det{W}}) ->
        (iteration, S, v, eigval_history, bagsize_history, iter_times) | nothing

Load a checkpoint. Returns `nothing` if the file does not exist. Throws if the
file exists but the magic number or W do not match (schema mismatch).
"""
function load_checkpoint(path::String, ::Type{Det{W}}) where {W}
    isfile(path) || return nothing

    raw = read(path)
    @assert length(raw) >= 44 "Checkpoint file too small: $(length(raw)) bytes"

    local magic, w_in_file, iter, nS, nh, nb, nt
    GC.@preserve raw begin
        base = pointer(raw)
        magic     = unsafe_load(Ptr{UInt32}(base + 0))
        w_in_file = unsafe_load(Ptr{UInt32}(base + 4))
        iter      = Int(unsafe_load(Ptr{UInt32}(base + 8)))
        nS        = Int(unsafe_load(Ptr{UInt64}(base + 12)))
        nh        = Int(unsafe_load(Ptr{UInt64}(base + 20)))
        nb        = Int(unsafe_load(Ptr{UInt64}(base + 28)))
        nt        = Int(unsafe_load(Ptr{UInt64}(base + 36)))
    end

    magic == CHECKPOINT_MAGIC || error(
        "Checkpoint magic mismatch in $path  (got 0x$(string(magic, base=16)) " *
        "expected 0x$(string(CHECKPOINT_MAGIC, base=16))). " *
        "Likely a different schema; delete and restart fresh.")
    Int(w_in_file) == W || error(
        "Checkpoint W mismatch in $path  (file W=$w_in_file, code W=$W). " *
        "Different system geometry — delete checkpoint and restart fresh.")

    expected_total = 44 + nS * W * 8 + nS * 8 + nh * 8 + nb * 8 + nt * 8
    @assert length(raw) == expected_total "Checkpoint truncated: $(length(raw)) vs $expected_total"

    S = Vector{Det{W}}(undef, nS)
    v = Vector{Float64}(undef, nS)
    eigval_history  = Vector{Float64}(undef, nh)
    bagsize_history = Vector{Int}(undef, nb)
    iter_times      = Vector{Float64}(undef, nt)

    GC.@preserve raw begin
        base = pointer(raw)
        off = 44

        @inbounds for i in 1:nS
            words = ntuple(k -> unsafe_load(Ptr{UInt64}(base + off + (k - 1) * 8)), Val(W))
            S[i] = Det{W}(words)
            off += W * 8
        end
        @inbounds for i in 1:nS
            v[i] = unsafe_load(Ptr{Float64}(base + off))
            off += 8
        end
        @inbounds for i in 1:nh
            eigval_history[i] = unsafe_load(Ptr{Float64}(base + off))
            off += 8
        end
        @inbounds for i in 1:nb
            bagsize_history[i] = Int(unsafe_load(Ptr{Int64}(base + off)))
            off += 8
        end
        @inbounds for i in 1:nt
            iter_times[i] = unsafe_load(Ptr{Float64}(base + off))
            off += 8
        end
    end

    return (iter, S, v, eigval_history, bagsize_history, iter_times)
end

"""
    checkpoint_path_for(json_outfile)

Build the canonical checkpoint path from a JSON output path.
"""
function checkpoint_path_for(json_outfile::String, shift::Float64 = 0.0)
    json_dir = dirname(json_outfile)
    base = basename(json_outfile)
    name_root = endswith(base, ".json") ? base[1:end-5] : base

    shift == 0.0 || (name_root *= @sprintf("_s%g", shift))

    # results_dir is the parent of <method>/  (e.g. results/greedy/run.json → results)
    method_dir = json_dir
    results_dir = dirname(method_dir)
    method_name = basename(method_dir)

    ckpt_dir = joinpath(results_dir, "checkpoints")
    return joinpath(ckpt_dir, method_name * "__" * name_root * ".ckpt")
end