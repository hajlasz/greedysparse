#=------------------------------------------------------------------------------
batched_expand_and_select_ooc.jl

Out-of-core (OOC) implementation of one "expand + sparsify" step in the
spin-coupled basis.

ONE APPLICATION OF A, processed in source batches:

  1. EXPAND. expanding_matvec_coupled! accumulates each batch's contributions
     to Ax, including diagonal and allowed single/double-excitation terms,
     in sharded in-memory dictionaries.
  2. DRAIN, PASS 1. Map determinant keys to rep_det and partition entries by
     the representative's hash into thread-local buffers, without locks.
  3. DRAIN, PASS 2. For each output shard, concatenate its buffers, sort by
     representative, sum duplicate entries, and append the resulting sorted
     sequence to disk as one run.
  4. Clear the accumulator and reset the buffers for reuse in the next batch.

FINAL SELECT. A k-way merge sums coefficients for each representative across
all runs in each shard and retains at most m entries of largest magnitude.
------------------------------------------------------------------------------=#

using Printf
using DataStructures: BinaryMinHeap
using CodecZlib: GzipCompressorStream, GzipDecompressorStream

@inline rec_bytes(::Type{Det{W}}) where {W} = 8 * W + 8

# =============================================================================
# Thread-local buffer used in Pass 1 → Pass 2
# (moved verbatim from the in-RAM sharded drain when that file was retired;
#  see file header for lineage)
# =============================================================================

"""
    DrainBuffer{W}

A growable vector of (det, val) pairs. One DrainBuffer per (thread, output-shard)
pair. Buffers are cleared but not reallocated between batches — `len` is reset
to 0 and the backing storage is reused.

This is what lets Pass 1 run without any locks: thread t appends the pairs
destined for output shard j only to its own buffer buf_tls[t][j], and Pass 2
(where shard j has a single owning thread) is the only reader.
"""
mutable struct DrainBuffer{W}
    dets::Vector{Det{W}}
    vals::Vector{Float64}
    len::Int
end

function DrainBuffer{W}(capacity::Int = 4096) where {W}
    @assert capacity > 0
    return DrainBuffer{W}(
        Vector{Det{W}}(undef, capacity),
        Vector{Float64}(undef, capacity),
        0,
    )
end

@inline function push_buf!(b::DrainBuffer{W}, d::Det{W}, v::Float64) where {W}
    cap = length(b.dets)
    if b.len >= cap
        # Amortized doubling. Rare in practice because initial capacity covers
        # the typical per-batch load.
        new_cap = cap * 2
        resize!(b.dets, new_cap)
        resize!(b.vals, new_cap)
    end
    b.len += 1
    @inbounds b.dets[b.len] = d
    @inbounds b.vals[b.len] = v
    return nothing
end

@inline function reset_buf!(b::DrainBuffer{W}) where {W}
    b.len = 0
    return nothing
end

# =============================================================================
# Pass 1
# =============================================================================

function drain_pass1_ooc!(
    acc::ExpandingAccumulator{W},
    buf_tls::Vector{Vector{DrainBuffer{W}}},
    T_plan::SpinFlipPlan,
    P_out::Int,
) where {W}
    P_in = acc.P
    nt   = length(buf_tls)
    @assert nt >= 1

    Threads.@threads :static for chunk_id in 1:nt
        @inbounds my_bufs = buf_tls[chunk_id]
        for si in chunk_id:nt:P_in
            shard = acc.shards[si]
            for (d, val) in shard
                d_rep = rep_det(d, T_plan)
                h = det_hash64(d_rep)
                owner = Int(h % UInt64(P_out)) + 1
                push_buf!(my_bufs[owner], d_rep, val)
            end
            empty!(shard)
        end
    end
    return nothing
end

# =============================================================================
# Pass 2 — sort, dedup, spill
# =============================================================================

"""
    sort_dedup_pair!(dets, vals, n, perm_scratch, sorted_dets_scratch, sorted_vals_scratch)
    -> n_out

Sort first n entries of (dets, vals) by det.words and collapse adjacent
same-det entries by summing vals. Mutates inputs in place. Returns new
length n_out (≤ n).

Caller-provided scratch arrays must each have length ≥ n. They are mutated.
This avoids per-call allocation, which at ~23k calls per iteration was
the source of >100 GB of GC pressure in the earlier version.
"""
function sort_dedup_pair!(
    dets::Vector{Det{W}},
    vals::Vector{Float64},
    n::Int,
    perm_scratch::Vector{Int},
    sorted_dets_scratch::Vector{Det{W}},
    sorted_vals_scratch::Vector{Float64},
) where {W}
    n == 0 && return 0
    @assert length(perm_scratch) >= n
    @assert length(sorted_dets_scratch) >= n
    @assert length(sorted_vals_scratch) >= n

    # Build identity permutation in-place
    @inbounds for i in 1:n
        perm_scratch[i] = i
    end

    # In-place sort of perm by dets[perm[i]].words
    sort!(view(perm_scratch, 1:n), by = i -> dets[i].words)

    # Apply permutation into scratch
    @inbounds for i in 1:n
        p = perm_scratch[i]
        sorted_dets_scratch[i] = dets[p]
        sorted_vals_scratch[i] = vals[p]
    end

    # Adjacent-merge in-place into (dets, vals)
    n_out = 1
    @inbounds dets[1] = sorted_dets_scratch[1]
    @inbounds vals[1] = sorted_vals_scratch[1]
    @inbounds for i in 2:n
        if sorted_dets_scratch[i] == dets[n_out]
            vals[n_out] += sorted_vals_scratch[i]
        else
            n_out += 1
            dets[n_out] = sorted_dets_scratch[i]
            vals[n_out] = sorted_vals_scratch[i]
        end
    end

    return n_out
end

"""
    pack_records!(byte_buf, dets, vals, n) -> bytes_used

Pack records (no header) into byte_buf at offset 0. Returns bytes used.
The caller writes any header separately.
"""
function pack_records!(
    byte_buf::Vector{UInt8},
    dets::Vector{Det{W}}, vals::Vector{Float64}, n::Int,
) where {W}
    rb = rec_bytes(Det{W})
    needed = n * rb
    @assert length(byte_buf) >= needed "byte_buf too small: $(length(byte_buf)) < $needed"

    GC.@preserve byte_buf begin
        base = pointer(byte_buf)
        @inbounds for i in 1:n
            rec_off = (i - 1) * rb
            d = dets[i]
            for k in 1:W
                word_off = rec_off + (k - 1) * 8
                unsafe_store!(Ptr{UInt64}(base + word_off), d.words[k])
            end
            val_off = rec_off + 8 * W
            unsafe_store!(Ptr{Float64}(base + val_off), vals[i])
        end
    end
    return needed
end

"""
    write_run!(io, raw_bytes, raw_len, n_records; compressed::Bool) -> bytes_written_to_io

Write one run to `io`. Format:

    [UInt8  flag]              # 0 = raw, 1 = gzip-compressed
    [UInt32 n_records]
    [UInt32 size]              # size of body in bytes
    [size bytes of body]       # raw_bytes[1:raw_len] if flag=0,
                               # else gzip(raw_bytes[1:raw_len])

For empty runs (n_records == 0), writes [flag=0, n=0, size=0] = 9 bytes
regardless of `compressed` argument.

Per-batch drain writes use compressed=false: writing raw is far cheaper
than compressing, and per-batch runs are typically consumed by compaction
within ~200 batches, so storing them uncompressed costs little disk over
their short lifetime.

Compaction writes use compressed=true: the consolidated runs are kept for
the rest of the iteration (potentially hours of additional batches), so
compressing them gives ~3× shrink that dominates disk usage.

Errors if any write returns short (typically disk full).
"""
function write_run!(
    io::IO, raw_bytes::Vector{UInt8}, raw_len::Int, n_records::Int;
    compressed::Bool,
)
    # Empty run sentinel — no body
    if n_records == 0
        header = Vector{UInt8}(undef, 9)
        GC.@preserve header begin
            unsafe_store!(Ptr{UInt8}(pointer(header)),      UInt8(0))   # flag=0
            unsafe_store!(Ptr{UInt32}(pointer(header) + 1), UInt32(0))  # n=0
            unsafe_store!(Ptr{UInt32}(pointer(header) + 5), UInt32(0))  # size=0
            nw = unsafe_write(io, pointer(header), 9)
            nw == 9 || error("write_run!: short header write $nw/9 (disk full?)")
        end
        return 9
    end

    @assert raw_len > 0
    @assert length(raw_bytes) >= raw_len

    if compressed
        # Stream-compress raw_bytes[1:raw_len] into an IOBuffer, then write
        # that buffer's bytes to io.
        compressed_io = IOBuffer()
        gz = GzipCompressorStream(compressed_io; level=1)
        write(gz, view(raw_bytes, 1:raw_len))
        close(gz)   # MUST close to flush gzip trailer
        body = take!(compressed_io)
        flag = UInt8(1)
        bsize = length(body)
    else
        # No compression — body IS the raw bytes
        body = raw_bytes
        flag = UInt8(0)
        bsize = raw_len
    end

    # Write 9-byte header
    header = Vector{UInt8}(undef, 9)
    GC.@preserve header begin
        unsafe_store!(Ptr{UInt8}(pointer(header)),      flag)
        unsafe_store!(Ptr{UInt32}(pointer(header) + 1), UInt32(n_records))
        unsafe_store!(Ptr{UInt32}(pointer(header) + 5), UInt32(bsize))
        nw = unsafe_write(io, pointer(header), 9)
        nw == 9 || error("write_run!: short header write $nw/9 (disk full?)")
    end

    # Write body
    GC.@preserve body begin
        nw = unsafe_write(io, pointer(body), bsize)
        nw == bsize || error(
            "write_run!: short body write $nw/$bsize (disk full?)")
    end
    return 9 + bsize
end

"""
    drain_pass2_ooc!(buf_tls, file_handles, sort_scratch_dets, sort_scratch_vals,
                     write_byte_bufs, perm_scratch_per_shard,
                     sorted_dets_scratch_per_shard, sorted_vals_scratch_per_shard)

Each thread owns one output shard. Concatenates buf_tls[*][j], sorts+dedups
using caller-provided scratch (no per-call allocation), packs, writes.
"""
function drain_pass2_ooc!(
    buf_tls::Vector{Vector{DrainBuffer{W}}},
    file_handles::Vector{<:IO},
    sort_scratch_dets::Vector{Vector{Det{W}}},
    sort_scratch_vals::Vector{Vector{Float64}},
    write_byte_bufs::Vector{Vector{UInt8}},
    perm_scratch_per_shard::Vector{Vector{Int}},
    sorted_dets_scratch_per_shard::Vector{Vector{Det{W}}},
    sorted_vals_scratch_per_shard::Vector{Vector{Float64}},
) where {W}
    nt    = length(buf_tls)
    P_out = length(file_handles)

    Threads.@threads :static for j in 1:P_out
        ntot = 0
        for tid in 1:nt
            ntot += @inbounds buf_tls[tid][j].len
        end

        if ntot == 0
            # Empty run sentinel
            write_run!(file_handles[j], UInt8[], 0, 0; compressed=false)
            for tid in 1:nt
                reset_buf!(@inbounds buf_tls[tid][j])
            end
            continue
        end

        sd = sort_scratch_dets[j]
        sv = sort_scratch_vals[j]
        if length(sd) < ntot
            resize!(sd, ntot)
            resize!(sv, ntot)
        end

        idx = 0
        @inbounds for tid in 1:nt
            b = buf_tls[tid][j]
            L = b.len
            for i in 1:L
                idx += 1
                sd[idx] = b.dets[i]
                sv[idx] = b.vals[i]
            end
            reset_buf!(b)
        end

        # Resize per-shard scratch if needed
        ps = perm_scratch_per_shard[j]
        sds = sorted_dets_scratch_per_shard[j]
        svs = sorted_vals_scratch_per_shard[j]
        if length(ps) < ntot
            resize!(ps, ntot)
            resize!(sds, ntot)
            resize!(svs, ntot)
        end

        n_out = sort_dedup_pair!(sd, sv, ntot, ps, sds, svs)

        # Pack just the records (no header) into wb, then write raw (no compression
        # for per-batch runs — they're consumed by compaction within ~200 batches
        # so storing them uncompressed costs little disk).
        rb = rec_bytes(Det{W})
        needed = n_out * rb
        wb = write_byte_bufs[j]
        if length(wb) < needed
            resize!(wb, needed)
        end
        pack_records!(wb, sd, sv, n_out)

        write_run!(file_handles[j], wb, needed, n_out; compressed=false)
    end
    return nothing
end

# =============================================================================
# K-way merge (unchanged)
# =============================================================================

# =============================================================================
# K-way merge with per-run gzip decompression
#
# Each RunReader owns a private file handle pointing at its run's compressed
# body, wrapped in a GzipDecompressorStream. The decompressor reads bytes
# from the file lazily (via libdeflate inside CodecZlib) and produces
# decompressed bytes on demand.

# =============================================================================

mutable struct RunReader{W}
    body::IO              # Either raw_io (raw run) or GzipDecompressorStream (compressed run)
    raw_io::IOStream      # underlying file handle (kept so we can close it)
    n_total::Int
    n_consumed::Int
    buf_dets::Vector{Det{W}}
    buf_vals::Vector{Float64}
    buf_pos::Int
    buf_len::Int
    chunk_size::Int
end

"""
    open_run_reader(shard_path, body_offset, n_records, flag; chunk_size)

Open a single RunReader for a run starting at byte offset `body_offset` in
`shard_path`. `flag=0` means raw body, `flag=1` means gzip-compressed.
The file is opened privately and seeked to body_offset.

Caller must eventually close via close_run!(r).
"""
function open_run_reader(
    ::Type{Det{W}}, shard_path::String, body_offset::Int64, n_records::Int, flag::UInt8;
    chunk_size::Int = 65536,
) where {W}
    raw_io = open(shard_path, "r")
    seek(raw_io, body_offset)
    body = flag == UInt8(1) ? GzipDecompressorStream(raw_io) : raw_io
    return RunReader{W}(
        body, raw_io, n_records, 0,
        Vector{Det{W}}(undef, chunk_size),
        Vector{Float64}(undef, chunk_size),
        1, 0, chunk_size,
    )
end

@inline has_more(r::RunReader) = r.n_consumed < r.n_total

"""
    close_run!(r)

Close the body stream (which is either the file handle directly, or a
GzipDecompressorStream wrapping it) and the underlying file handle.
"""
function close_run!(r::RunReader)
    # If body is a wrapper stream (compressed), close it first to release
    # its internal state. If body IS raw_io (raw), this becomes the same
    # close as the second one — that's why both are wrapped in try blocks.
    try
        close(r.body)
    catch
    end
    try
        close(r.raw_io)
    catch
    end
    return nothing
end

function refill!(r::RunReader{W}) where {W}
    remaining = r.n_total - r.n_consumed
    n_read = min(remaining, r.chunk_size)
    rb = rec_bytes(Det{W})
    bytes_to_read = n_read * rb

    # Read decompressed-or-raw bytes from r.body into a fresh Vector{UInt8}.
    # If body is raw, read goes straight from the file. If body is gzip,
    # decompressor pulls from raw_io behind the scenes.
    raw = read(r.body, bytes_to_read)
    @assert length(raw) == bytes_to_read "RunReader: short read $(length(raw))/$bytes_to_read"

    GC.@preserve raw begin
        base = pointer(raw)
        @inbounds for i in 1:n_read
            rec_off = (i - 1) * rb
            words = ntuple(k -> unsafe_load(Ptr{UInt64}(base + rec_off + (k - 1) * 8)), Val(W))
            r.buf_dets[i] = Det{W}(words)
            r.buf_vals[i] = unsafe_load(Ptr{Float64}(base + rec_off + 8 * W))
        end
    end
    r.buf_pos = 1
    r.buf_len = n_read
    return nothing
end

@inline function peek_det(r::RunReader{W}) where {W}
    if r.buf_pos > r.buf_len
        refill!(r)
    end
    return @inbounds r.buf_dets[r.buf_pos]
end

@inline function peek_val(r::RunReader{W}) where {W}
    if r.buf_pos > r.buf_len
        refill!(r)
    end
    return @inbounds r.buf_vals[r.buf_pos]
end

@inline function advance!(r::RunReader)
    r.buf_pos += 1
    r.n_consumed += 1
    return nothing
end

"""
    open_runs_for_shard(path, ::Type{Det{W}}) -> Vector{RunReader{W}}

Scan the shard file's compressed-run headers, build a RunReader per
non-empty run. Each RunReader gets its own file handle (because each
needs a private gzip decompressor state).

File format per run:
    [UInt32 n_records]              4 bytes
    [UInt32 compressed_byte_count]  4 bytes
    [compressed_byte_count bytes]   compressed body (gzip stream)

Empty runs encoded as [UInt32 0][UInt32 0] (8 bytes, no body).

Returns just the readers — there is no shared IO handle anymore (each
reader owns its own fd). Caller closes via close_run!(r) after merge.

Number of fds used: ~one per non-empty run. With ozone iter 3 lazy
compaction at ~10 compacted runs per shard × 64 shards = ~640 fds.
Default Linux soft fd limit is 1024; we're under but the user may want
to bump via `ulimit -n 4096` for safety.
"""
function open_runs_for_shard(path::String, ::Type{Det{W}}) where {W}
    runs = RunReader{W}[]
    if !isfile(path)
        return runs
    end

    # Single scan handle to read 9-byte headers and skip past bodies.
    # Each non-empty run gets its OWN file handle for the decompressor.
    # Header layout:  [UInt8 flag][UInt32 n][UInt32 size]
    scan_io = open(path, "r")
    try
        while !eof(scan_io)
            flag = read(scan_io, UInt8)
            n_raw = read(scan_io, UInt32)
            size_raw = read(scan_io, UInt32)
            n = Int(n_raw)
            bsize = Int(size_raw)
            body_offset = position(scan_io)
            if n > 0
                push!(runs, open_run_reader(Det{W}, path, Int64(body_offset), n, flag))
            end
            # Skip past this run's body to the next header.
            seek(scan_io, body_offset + bsize)
        end
    catch e
        # Clean up any already-opened RunReaders so we don't leak fds
        for r in runs
            close_run!(r)
        end
        empty!(runs)
        rethrow(e)
    finally
        try
            close(scan_io)
        catch
        end
    end

    return runs
end

struct MergeHeapEntry{W}
    d::Det{W}
    run_idx::Int
end

@inline Base.isless(a::MergeHeapEntry{W}, b::MergeHeapEntry{W}) where {W} = a.d.words < b.d.words

function merge_shard_topk(
    runs::Vector{RunReader{W}}, k_local::Int,
) where {W}
    isempty(runs) && return Det{W}[], Float64[]

    topk = BinaryMinHeap{TopKItem{W}}()
    tiebreak = Int64(0)

    merge_heap = BinaryMinHeap{MergeHeapEntry{W}}()
    for (i, r) in enumerate(runs)
        if has_more(r)
            push!(merge_heap, MergeHeapEntry{W}(peek_det(r), i))
        end
    end

    while !isempty(merge_heap)
        top = first(merge_heap)
        cur_det = top.d
        sum_v   = 0.0

        while !isempty(merge_heap) && first(merge_heap).d == cur_det
            entry = pop!(merge_heap)
            r = runs[entry.run_idx]
            sum_v += peek_val(r)
            advance!(r)
            if has_more(r)
                push!(merge_heap, MergeHeapEntry{W}(peek_det(r), entry.run_idx))
            end
        end

        a = abs(sum_v)
        if sum_v != 0.0
            tiebreak += 1
            item = TopKItem{W}(a, tiebreak, cur_det, sum_v)
            if length(topk) < k_local
                push!(topk, item)
            elseif a > first(topk).a
                pop!(topk)
                push!(topk, item)
            end
        end
    end

    n = length(topk)
    items = Vector{TopKItem{W}}(undef, n)
    @inbounds for i in 1:n
        items[i] = pop!(topk)
    end
    reverse!(items)

    S_local = Vector{Det{W}}(undef, n)
    x_local = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        S_local[i] = items[i].d
        x_local[i] = items[i].v
    end
    return S_local, x_local
end

# =============================================================================
# Compaction — periodic merging of accumulated runs into a single sorted run.
#
# Without this, a shard's file accumulates one run per batch (2000 runs at
# iter 3 of m=1M ozone), with the same det stored once per batch it appeared
# in. After 100 batches at iter 3 we observed 234 GB total spill, projecting
# to ~5 TB by batch 2000 — bigger than disk allows.
#
# Compaction reads all current runs, K-way merges them (summing same-det
# values across runs), and rewrites the shard as one consolidated run. After
# compaction the shard file size is bounded by (unique dets in shard) × 32
# bytes, which for ozone iter 3 is a few GB instead of tens of GB.
#
# We compact every M batches. Choice of M trades off:
#  - larger M: fewer compactions, more disk used between them
#  - smaller M: more I/O overhead, less disk used
# M=200 (10 compactions per iter 3) gives ~75 GB disk peak, total I/O ~3×
# the no-compaction would be without OOM, finishes well within wall time.
# =============================================================================

"""
    compact_run_to_file(runs, output_path, ::Type{Det{W}})

K-way merge a list of RunReaders into a single sorted run on disk, summing
values for matching dets. Output format matches per-batch run writes:

    [UInt32 n_records uncompressed]
    [UInt32 compressed_byte_count]
    [compressed_byte_count bytes of gzip stream]

Streaming: records flow through a GzipCompressorStream as they're produced
by the merge, so we never hold the full uncompressed run in RAM. At end,
we close the gz stream, fseek back to the header, and patch in the real
record count and compressed size.

Memory: ~few MB of merge state + ~64 KB gzip internal buffer + ~2 MB
record-pack scratch.
"""
function compact_run_to_file(
    runs::Vector{RunReader{W}}, output_path::String,
) where {W}
    rb = rec_bytes(Det{W})
    PACK_CHUNK = 65536    # records per pack-and-flush — ~2 MB raw

    out_io = open(output_path, "w")
    try
        # Reserve 9 bytes for header (flag + n_records + size).
        # We'll seek back and fill them after we know the final values.
        write(out_io, UInt8(1))     # flag=1 (compressed)
        write(out_io, UInt32(0))    # n_records placeholder
        write(out_io, UInt32(0))    # compressed_size placeholder

        # The gzip stream wraps out_io directly — writes flow through gzip
        # into the file at its current position (right after the 9-byte header).
        # Use level=1 for fast compression (~150 MB/s vs default ~50 MB/s).
        gz = GzipCompressorStream(out_io; level=1)

        # Pre-allocate a small raw-record buffer; we pack PACK_CHUNK records
        # at a time, then write the raw bytes into gz which compresses on the fly.
        wb = Vector{UInt8}(undef, PACK_CHUNK * rb)

        merge_heap = BinaryMinHeap{MergeHeapEntry{W}}()
        for (i, r) in enumerate(runs)
            if has_more(r)
                push!(merge_heap, MergeHeapEntry{W}(peek_det(r), i))
            end
        end

        wb_count = 0
        n_total  = Int64(0)

        while !isempty(merge_heap)
            top = first(merge_heap)
            cur_det = top.d
            sum_v   = 0.0

            while !isempty(merge_heap) && first(merge_heap).d == cur_det
                entry = pop!(merge_heap)
                r = runs[entry.run_idx]
                sum_v += peek_val(r)
                advance!(r)
                if has_more(r)
                    push!(merge_heap, MergeHeapEntry{W}(peek_det(r), entry.run_idx))
                end
            end

            if sum_v != 0.0
                rec_off = wb_count * rb
                GC.@preserve wb begin
                    base = pointer(wb)
                    for kk in 1:W
                        unsafe_store!(Ptr{UInt64}(base + rec_off + (kk - 1) * 8),
                                      cur_det.words[kk])
                    end
                    unsafe_store!(Ptr{Float64}(base + rec_off + 8 * W), sum_v)
                end
                wb_count += 1
                n_total  += 1

                if wb_count == PACK_CHUNK
                    bytes_to_write = wb_count * rb
                    write(gz, view(wb, 1:bytes_to_write))
                    wb_count = 0
                end
            end
        end

        # Flush remaining records into gz
        if wb_count > 0
            bytes_to_write = wb_count * rb
            write(gz, view(wb, 1:bytes_to_write))
        end

        # Closing gz finalizes the gzip stream (writes trailer) and flushes
        # buffered compressed bytes into out_io.
        close(gz)
        try
            flush(out_io)
        catch
        end
        try
            close(out_io)
        catch
        end

        # Now compute the compressed size from on-disk size.
        total_size = filesize(output_path)
        compressed_size = total_size - 9
        @assert compressed_size >= 0

        # Reopen to patch the header.
        patch_io = open(output_path, "r+")
        try
            seek(patch_io, 0)
            write(patch_io, UInt8(1))            # flag=1
            write(patch_io, UInt32(n_total))     # n_records
            write(patch_io, UInt32(compressed_size))
            flush(patch_io)
        finally
            close(patch_io)
        end

        return n_total
    catch e
        try; close(out_io); catch; end
        rethrow(e)
    end
end

"""
    lazy_compact_shard!(shard_path, fh, last_compacted_offset, ::Type{Det{W}})
      -> (new_compacted_offset, n_records_in_new_compaction)

Lazy compaction: merge ONLY runs added since `last_compacted_offset` into a
single consolidated run, and append it to the file. Already-compacted runs
earlier in the file are left untouched.
"""
function lazy_compact_shard!(
    shard_path::String, fh::IOStream,
    last_compacted_offset::Int64, ::Type{Det{W}},
) where {W}
    # Always close fh on entry — caller will reopen for append after we
    # return. This keeps the file-handle contract simple regardless of
    # whether we actually do any compaction work.
    flush(fh)
    close(fh)

    cur_size = filesize(shard_path)
    if cur_size <= last_compacted_offset
        # No new data since last compaction — nothing to do.
        return last_compacted_offset, Int64(0)
    end

    # ---- Step A: scan only the new-runs region; build per-run readers ----
    runs = RunReader{W}[]
    scan_io = open(shard_path, "r")
    try
        try
            seek(scan_io, last_compacted_offset)
            while position(scan_io) < cur_size
                flag      = read(scan_io, UInt8)
                n_raw     = read(scan_io, UInt32)
                size_raw  = read(scan_io, UInt32)
                n         = Int(n_raw)
                bsize     = Int(size_raw)
                body_offset = position(scan_io)
                if n > 0
                    # Each reader gets its own private file handle
                    push!(runs, open_run_reader(Det{W}, shard_path, Int64(body_offset), n, flag))
                end
                seek(scan_io, body_offset + bsize)
            end
        catch e
            # Mid-scan error — close any RunReaders we've already opened
            # to avoid fd leak before re-raising.
            for r in runs
                close_run!(r)
            end
            empty!(runs)
            rethrow(e)
        end

        if isempty(runs)
            return last_compacted_offset, Int64(0)
        end

        # ---- Step B: K-way merge of new runs into temp file ----
        tmp_path = shard_path * ".compact.tmp"
        n_emitted = try
            compact_run_to_file(runs, tmp_path)
        finally
            for r in runs
                close_run!(r)
            end
        end

        # ---- Step C: truncate shard file, append temp ----
        # (writer was already closed at function entry)

        # Truncate to drop the raw runs we just consumed
        truncate_io = open(shard_path, "r+")
        truncate(truncate_io, last_compacted_offset)
        close(truncate_io)

        # Append temp file's contents to shard file via streaming copy
        if n_emitted > 0
            tmp_size = filesize(tmp_path)
            CHUNK = 8 * 1024 * 1024   # 8 MB streaming buffer
            tmp_io = open(tmp_path, "r")
            append_io = open(shard_path, "a")
            try
                bytes_remaining = tmp_size
                while bytes_remaining > 0
                    n_read = min(bytes_remaining, CHUNK)
                    raw = read(tmp_io, n_read)
                    @assert length(raw) == n_read
                    GC.@preserve raw begin
                        nw = unsafe_write(append_io, pointer(raw), n_read)
                        nw == n_read || error(
                            "lazy_compact_shard!: short write $nw/$n_read on $shard_path " *
                            "(disk full or filesystem error)")
                    end
                    bytes_remaining -= n_read
                end
            finally
                close(append_io)
                close(tmp_io)
            end
        end

        # Clean up temp file
        try
            isfile(tmp_path) && rm(tmp_path)
        catch
        end

        new_offset = filesize(shard_path)
        return new_offset, Int64(n_emitted)
    finally
        try
            close(scan_io)
        catch
        end
    end
end

# =============================================================================
# Main entry point
# =============================================================================

function batched_expand_merge_select_ooc(
    S::Vector{Det{W}},
    x::Vector{Float64},
    norm_fac::Vector{Float64},
    env::QCEnv{W},
    T_plan::SpinFlipPlan,
    C::CoupledConsts,
    IRREPS::Vector{Int},
    k::Int;
    batch_size::Int,
    spill_dir::String,
    P_out::Int = 0,
    verbose::Bool = false,
    gc_every::Int = 50,        # force GC every N batches to release dead bucket arrays
    compact_every::Int = 200,  # compact each shard's runs every N batches; bounds disk usage
    max_disk_gb::Float64 = 3000.0,  # abort if total spill exceeds this many GB
) where {W}

    nS = length(S)
    @assert length(x) == nS
    @assert length(norm_fac) == nS
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))

    nt    = Threads.nthreads()
    P_in  = max(64, 10 * nt)
    P_out = P_out > 0 ? P_out : max(64, 2 * nt)

    isdir(spill_dir) || mkpath(spill_dir)
    shard_paths = [joinpath(spill_dir, "shard_$(j).bin") for j in 1:P_out]
    for p in shard_paths
        isfile(p) && rm(p)
    end

    file_handles = IOStream[open(p, "w") for p in shard_paths]

    # Track which prefix of each shard file is "already compacted." The
    # lazy compaction step only re-merges runs added since this offset.
    # Initialized to 0 because we just opened fresh empty files.
    last_compacted_offset = zeros(Int64, P_out)

    expected_per_batch = 10 * batch_size
    acc = ExpandingAccumulator{W}(P_in, expected_per_batch)

    initial_buf_cap = 4096
    buf_tls = [[DrainBuffer{W}(initial_buf_cap) for _ in 1:P_out] for _ in 1:nt]

    # Pre-allocated per-shard scratch (no per-call allocations in sort_dedup)
    sort_scratch_dets             = [Vector{Det{W}}(undef, 0)   for _ in 1:P_out]
    sort_scratch_vals             = [Vector{Float64}(undef, 0)  for _ in 1:P_out]
    perm_scratch_per_shard        = [Vector{Int}(undef, 0)      for _ in 1:P_out]
    sorted_dets_scratch_per_shard = [Vector{Det{W}}(undef, 0)   for _ in 1:P_out]
    sorted_vals_scratch_per_shard = [Vector{Float64}(undef, 0)  for _ in 1:P_out]

    write_byte_bufs = [Vector{UInt8}(undef, 4 + 1024 * rec_bytes(Det{W}))
                       for _ in 1:P_out]

    n_batches = cld(nS, batch_size)
    t_total   = time()

    if verbose
        println("  batched_expand_merge_select_ooc:")
        @printf("    |S| = %d, k = %d, batch_size = %d, n_batches = %d\n",
                nS, k, batch_size, n_batches)
        @printf("    nthreads = %d, P_in = %d, P_out = %d\n", nt, P_in, P_out)
        @printf("    spill_dir = %s\n", spill_dir)
        flush(stdout)
    end

    try
        for b in 1:n_batches
            i_start = (b - 1) * batch_size + 1
            i_end   = min(b * batch_size, nS)

            S_batch  = collect(view(S,        i_start:i_end))
            x_batch  = collect(view(x,        i_start:i_end))
            nf_batch = collect(view(norm_fac, i_start:i_end))

            t_expand = @elapsed expanding_matvec_coupled!(
                acc, S_batch, x_batch, nf_batch, env, T_plan, C, IRREPS,
            )

            t_p1 = @elapsed drain_pass1_ooc!(acc, buf_tls, T_plan, P_out)

            t_p2 = @elapsed drain_pass2_ooc!(
                buf_tls, file_handles,
                sort_scratch_dets, sort_scratch_vals,
                write_byte_bufs,
                perm_scratch_per_shard,
                sorted_dets_scratch_per_shard,
                sorted_vals_scratch_per_shard,
            )

            # Periodic GC to release dead bucket arrays from acc.shards' empty!()
            # and any drained DrainBuffer pages.
            if b % gc_every == 0
                GC.gc()
            end

            # Periodic LAZY compaction: K-way merge each shard's runs added
            # since the last compaction into a single new compacted run,
            # appended to the existing file. Already-compacted runs from
            # earlier intervals are left untouched.
            #
            # Skip on the very last batch — the final merge_shard_topk pass
            # at the bottom of this function K-way merges all runs straight
            # into top-k, so compacting first would just be wasted I/O.
            if b % compact_every == 0 && b < n_batches
                t_compact = @elapsed begin
                    Threads.@threads :static for j in 1:P_out
                        new_off, _n_emitted = lazy_compact_shard!(
                            shard_paths[j], file_handles[j],
                            last_compacted_offset[j], Det{W},
                        )
                        # File was closed inside lazy_compact_shard!; reopen
                        # for append to continue receiving new runs.
                        file_handles[j] = open(shard_paths[j], "a")
                        last_compacted_offset[j] = new_off
                    end
                    GC.gc()
                end
                disk_after = sum(filesize(p) for p in shard_paths if isfile(p))
                if verbose
                    @printf("    [batch %d] lazy compaction: %.1fs, disk after = %.2f GB\n",
                            b, t_compact, disk_after / (1024.0^3))
                    flush(stdout)
                end

                # Disk-budget enforcement: abort cleanly if we're approaching
                # the configured limit. 
                if disk_after / (1024.0^3) > max_disk_gb
                    error("OOC drain disk usage $(round(disk_after / 1024.0^3, digits=1)) GB " *
                          "exceeds budget $(max_disk_gb) GB at batch $b/$n_batches. " *
                          "Aborting before disk fills. Either: (1) increase max_disk_gb if " *
                          "your /scratch quota allows, (2) decrease compact_every for more " *
                          "frequent compactions, (3) reduce m to limit total emissions.")
                end
            end

            if verbose && (b == 1 || b == n_batches || b % max(1, cld(n_batches, 20)) == 0)
                @printf("    batch %d/%d: expand %.2fs, p1 %.2fs, p2 %.2fs\n",
                        b, n_batches, t_expand, t_p1, t_p2)
                flush(stdout)
            end
        end

        for fh in file_handles
            try
                isopen(fh) && close(fh)
            catch
            end
        end

        # Free large in-memory state before the merge phase, which spawns its
        # own working set.
        empty!(buf_tls)
        empty!(sort_scratch_dets)
        empty!(sort_scratch_vals)
        empty!(perm_scratch_per_shard)
        empty!(sorted_dets_scratch_per_shard)
        empty!(sorted_vals_scratch_per_shard)
        empty!(write_byte_bufs)
        # acc held by closure; clear shards explicitly
        for s in acc.shards
            empty!(s)
        end
        GC.gc()

        if verbose
            disk_total = sum(filesize(p) for p in shard_paths if isfile(p))
            @printf("    drain done in %.1fs.  Total spill on disk: %.2f GB\n",
                    time() - t_total, disk_total / (1024.0^3))
            flush(stdout)
        end

        # Top-k phase
        t_merge = time()

        local_S = Vector{Vector{Det{W}}}(undef, P_out)
        local_x = Vector{Vector{Float64}}(undef, P_out)

        Threads.@threads :static for j in 1:P_out
            runs = open_runs_for_shard(shard_paths[j], Det{W})
            sj, xj = merge_shard_topk(runs, k)
            for r in runs
                close_run!(r)
            end
            local_S[j] = sj
            local_x[j] = xj
        end

        global_heap = BinaryMinHeap{TopKItem{W}}()
        tiebreak = Int64(0)
        for j in 1:P_out
            sj = local_S[j]
            xj = local_x[j]
            for i in 1:length(xj)
                a = abs(xj[i])
                tiebreak += 1
                item = TopKItem{W}(a, tiebreak, sj[i], xj[i])
                if length(global_heap) < k
                    push!(global_heap, item)
                elseif a > first(global_heap).a
                    pop!(global_heap)
                    push!(global_heap, item)
                end
            end
        end

        n_final = length(global_heap)
        items = Vector{TopKItem{W}}(undef, n_final)
        @inbounds for i in 1:n_final
            items[i] = pop!(global_heap)
        end
        reverse!(items)

        S_new = Vector{Det{W}}(undef, n_final)
        x_new = Vector{Float64}(undef, n_final)
        @inbounds for i in 1:n_final
            S_new[i] = items[i].d
            x_new[i] = items[i].v
        end

        if verbose
            @printf("    merge+topk: %.2fs  → returning %d entries (total %.2fs)\n",
                    time() - t_merge, n_final, time() - t_total)
            flush(stdout)
        end

        return S_new, x_new

    finally
        for fh in file_handles
            try
                isopen(fh) && close(fh)
            catch
            end
        end
        for p in shard_paths
            try
                isfile(p) && rm(p)
            catch
            end
        end
    end
end
