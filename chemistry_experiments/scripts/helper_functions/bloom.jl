# bloom.jl
# ----------------------------------------------------------------------------
# Bloom filter over Det{W}.
#
# Used by the restricted matvec to cheaply reject candidate neighbors that are
# not in the current bag S, before paying for a det2idx hash lookup. Membership
# queries have NO false negatives but may have false positives.
# 
# Bloom filter is definted by setting k iid hash functions h_i: U -> {0,...,m-1}
# Then hash each x in S to bitarray b_x with nnz positions set: h_1(x), h_2(x), ..., h_k(x).
# Set b to an m-length bitstring with nnz(b) = union of nnz(b_x) for x in S.
# To test if z is in S, check if all bits h_i(z) are equal to 1 in b.
# 
# Choosing m and k: For N items in S, the false positive rate is approximately (1 - exp(-k*N/m))^k.
# ----------------------------------------------------------------------------

# `bits`  : bitarray that will serve as the Bloom filter, stored as a vector of UInt64 words for efficient bit operations.
# `mask` : m - 1, where m (a power of two) is the number of bits in the filter.
#          A scrambled hash h is reduced to a bit position with `h & mask`,
#          which equals `h mod m` because m is a power of two (cheap, no division).
# `k`     : number of iid hash functions to use
struct BloomFilter
    bits::Vector{UInt64}
    mask::UInt64
    k::Int
end

# ----------------------------------------------------------------------------
# Hashing
#
# We approximate a family of k independent uniform hash functions by "double hashing"
#
# We define two hash functions h1 and h2 as follows:
# h1(d) = det_hash64(d) is a 64-bit hash of the determinant d, mixing all its W words together.
# h2(d) = _mix64(h1(d) ⊻ 0xD6E8FEB86659FD93) is a second hash derived from h1 by scrambling it with a different constant.
# Then we get k locations in an m-length bitarray via
# h_i(d) = (h1(d) + i * h2(d) mod 2^64) mod m, for i = 0, 1, ..., k-1
# 
# ----------------------------------------------------------------------------

# _mix64 is the scrambler: it stirs a 64-bit word so that changing even one input
# bit changes the output unpredictably. Returns a 64-bit hash of its input. 
@inline function _mix64(x::UInt64)
    x ⊻= x >>> 30
    x *= 0xbf58476d1ce4e5b9
    x ⊻= x >>> 27
    x *= 0x94d049bb133111eb
    x ⊻= x >>> 31
    return x
end

# det_hash64 turns a whole determinant. Returns a 64-bit hash of its input.
@inline function det_hash64(d::Det{W}) where {W}
    h = 0x9e3779b97f4a7c15
    @inbounds for w in 1:W
        h ⊻= d.words[w]
        h = _mix64(h)
    end
    return h
end

# Compute the (h1, h2) pair used to generate the k probe positions
@inline function _double_hash(d::Det{W}) where {W}
    h1 = det_hash64(d)
    h2 = _mix64(h1 ⊻ 0xD6E8FEB86659FD93)
    return h1, h2
end

# _probe converts one 64-bit hash h into the concrete (word, single-bit) it points
# at inside the packed bit array
@inline function _probe(bf::BloomFilter, h::UInt64)
    idx = h & bf.mask              # 0 .. m-1
    w   = (idx >>> 6) + 1          # 1-based word
    bit = UInt64(1) << (idx & 0x3f)
    return w, bit
end

# ----------------------------------------------------------------------------
# Construction and membership
# ----------------------------------------------------------------------------

# Build an empty filter sized for `nitems` elements:
#   m      = number of bits in the filter, rounded up to a power of two. We
#            want m >= nitems * bits_per_item
#   nwords = ⌈m/64⌉ words needed to store m bits  ((m + 63) >>> 6 = ⌈m/64⌉).
function BloomFilter(nitems::Int; bits_per_item::Int = 12, k::Int = 3)
    m      = nextpow(2, max(1, nitems * bits_per_item)) 
    nwords = (m + 63) >>> 6
    return BloomFilter(zeros(UInt64, nwords), UInt64(m - 1), k)
end

# Insert d by setting its k bits to 1 in the filter.
@inline function bloom_add!(bf::BloomFilter, d::Det{W}) where {W}
    h1, h2 = _double_hash(d)
    @inbounds for t in 0:(bf.k - 1)
        w, bit = _probe(bf, h1 + UInt64(t) * h2)
        bf.bits[w] |= bit
    end
    return nothing
end

# Membership test: look at the same k bits that inserting d would have set.
#   - If ALL k are 1, d MIGHT be present (or those bits were set by other
#     elements, in which case this is a harmless false positive).
#   - If ANY is 0, d was never inserted, so it is DEFINITELY absent.
@inline function bloom_maybe_contains(bf::BloomFilter, d::Det{W}) where {W}
    h1, h2 = _double_hash(d)
    @inbounds for t in 0:(bf.k - 1)
        w, bit = _probe(bf, h1 + UInt64(t) * h2)
        (bf.bits[w] & bit) != 0 || return false
    end
    return true
end

# Build and populate a filter sized for the determinants currently in the bag S.
# `gs` is left untyped so this file need not depend on bag.jl's GreedyState; it
# only requires that `gs.S` is an iterable of Det{W}.
function build_bloom(gs)::BloomFilter
    bf = BloomFilter(length(gs.S); bits_per_item = 12, k = 3)
    @inbounds for d in gs.S
        bloom_add!(bf, d)
    end
    return bf
end