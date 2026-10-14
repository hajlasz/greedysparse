# bitops.jl
# Bit-packed determinants Det{W}, orbital masks, occupation/flip/sign-support
# helpers, and HF determinant construction.

# ------------------------------------------------------------------------------
# Bitstring Helper Functions
# 
# Each determinant d is a bunch of 64-bit "words" (UInt64)s. Each word stores 
# occupations for 64 spin-orbitas (bits). 
#
# 1...n are spin-up orbitals, n+1...2n are spin-down orbitals.
# ------------------------------------------------------------------------------
struct Det{W}
    words::NTuple{W,UInt64}   # occupancy bits for 2n spin-orbitals
end
struct OrbMasks 
    # Precomputed lookup table: which word and bitmask to use for occupancy checks and flips
    # orbital index i (1-based) has word index word[i] and bitmask mask[i]
    word::Vector{Int}
    mask::Vector{UInt64}
end
function build_orbmasks(nso::Int)
    word = Vector{Int}(undef, nso)
    mask = Vector{UInt64}(undef, nso)

    @inbounds for i in 1:nso
        w = ((i - 1) >>> 6) + 1
        b = (i - 1) & 0x3f
        word[i] = w
        mask[i] = UInt64(1) << b
    end

    return OrbMasks(word, mask)
end

# translate between spatial and spin-orbital indices
@inline up(i::Int, n::Int) = i          # 1..n
@inline dn(i::Int, n::Int) = i + n      # 1..n  -> n+1..2n
@inline is_up_so(p_so::Int, n::Int) = (p_so <= n)
@inline spatial(p_so::Int, n::Int)  = (p_so <= n) ? p_so : (p_so - n)

# check whether two determinants are equal
import Base: ==, isequal, hash
@inline ==(a::Det{W}, b::Det{W}) where {W} = a.words == b.words
@inline isequal(a::Det{W}, b::Det{W}) where {W} = isequal(a.words, b.words)
@inline hash(a::Det{W}, h::UInt) where {W} = hash(a.words, h)

# returns w = 1-based index of word that holds orbital i
# returns b = 0-based bit position inside that word (0 to 63)
@inline wordbit(i::Int) = ((i-1) >>> 6) + 1, (i-1) & 0x3f

@inline bitmask(b::Int) = (UInt64(1) << b)

# is orbital i occupied in determinant d?
@inline function isocc(d::Det{W}, i::Int) where {W}
    w, b = wordbit(i)
    return ((d.words[w] >>> b) & 0x01) == 0x01
end
@inline function isocc_fast(d::Det{W}, i::Int, om::OrbMasks) where {W}
    @inbounds return (d.words[om.word[i]] & om.mask[i]) != 0
end

# Return a new determinant where orbital i is flipped
@inline function xorocc(d::Det{W}, i::Int) where {W}
    w, b = wordbit(i)
    mask = UInt64(1) << b
    newwords = ntuple(k -> k == w ? (d.words[k] ⊻ mask) : d.words[k], W)
    return Det{W}(newwords)
end
# Return a new determinant where i and a are flipped.
@inline function flip2(d::Det{W}, i::Int, a::Int, om::OrbMasks) where {W}
    @inbounds begin
        wi = om.word[i]; mi = om.mask[i]
        wa = om.word[a]; ma = om.mask[a]
        if wi == wa
            x = mi ⊻ ma
            return Det{W}(ntuple(k -> k == wi ? (d.words[k] ⊻ x) : d.words[k], W))
        else
            return Det{W}(ntuple(k -> k == wi ? (d.words[k] ⊻ mi) :
                                     k == wa ? (d.words[k] ⊻ ma) : d.words[k], W))
        end
    end
end
# Return a new determinant where i,j,a,b are flipped.
@inline function flip4(d::Det{W}, i::Int, j::Int, a::Int, b::Int, om::OrbMasks) where {W}
    @inbounds begin
        wi = om.word[i]; mi = om.mask[i]
        wj = om.word[j]; mj = om.mask[j]
        wa = om.word[a]; ma = om.mask[a]
        wb = om.word[b]; mb = om.mask[b]

        newwords = ntuple(k -> begin
            x = d.words[k]
            if k == wi; x ⊻= mi; end
            if k == wj; x ⊻= mj; end
            if k == wa; x ⊻= ma; end
            if k == wb; x ⊻= mb; end
            x
        end, W)

        return Det{W}(newwords)
    end
end
@inline function flip4(d::Det{1}, i::Int, j::Int, a::Int, b::Int, om::OrbMasks)
    @inbounds begin
        x = d.words[1]
        x ⊻= om.mask[i]
        x ⊻= om.mask[j]
        x ⊻= om.mask[a]
        x ⊻= om.mask[b]
        return Det{1}((x,))
    end
end

# Count occupied bits strictly between orbitals i and a (i.e. not including the endpoints).
@inline function _count_occ_between(d::Det{W}, i::Int, a::Int) where {W}
    i > a && ((i, a) = (a, i))
    lo = i + 1
    hi = a - 1
    lo > hi && return 0

    lo0 = lo - 1
    hi0 = hi - 1

    wlo = (lo0 >>> 6) + 1
    whi = (hi0 >>> 6) + 1
    blo = lo0 & 0x3f
    bhi = hi0 & 0x3f

    cnt = 0
    @inbounds if wlo == whi
        # mask bits blo..bhi in one word
        mask = ((UInt64(1) << (bhi - blo + 1)) - 1) << blo
        cnt += count_ones(d.words[wlo] & mask)
    else
        # first partial word
        cnt += count_ones(d.words[wlo] & (~UInt64(0) << blo))
        # full middle words
        for w in (wlo + 1):(whi - 1)
            cnt += count_ones(d.words[w])
        end
        # last partial word
        cnt += count_ones(d.words[whi] & ((UInt64(1) << (bhi + 1)) - 1))
    end

    return cnt
end

# Fill `out` with the occupied indices in the inclusive bit-range [lo, hi]
# Returns number filled (should equal `ne`).
@inline function _fill_occ_range!(out::Vector{Int}, d::Det{W},
                                  lo::Int, hi::Int, ne::Int) where {W}
    @inbounds begin
        # convert to 0-based bit positions
        lo0 = lo - 1
        hi0 = hi - 1

        wlo = (lo0 >>> 6) + 1
        whi = (hi0 >>> 6) + 1
        blo = lo0 & 0x3f
        bhi = hi0 & 0x3f

        j = 0

        for w in wlo:whi
            x = d.words[w]

            # mask off bits below lo (first word)
            if w == wlo && blo != 0
                x &= ~((UInt64(1) << blo) - 1)
            end
            # mask off bits above hi (last word)
            if w == whi && bhi != 63
                x &= (UInt64(1) << (bhi + 1)) - 1
            end

            base = (w - 1) << 6  # 0-based index of bit 0 in this word

            while x != 0
                t = trailing_zeros(x)          # position of lowest set bit (0..63)
                j += 1
                out[j] = base + t + 1          # back to 1-based orbital index
                j == ne && return j            # early exit (you know ne)
                x &= x - 1                     # clear lowest set bit
            end
        end

        return j
    end
end

# Fill `up_occ` (length nup) with occupied spin-up orbitals in 1:n,
# and `dn_occ` (length ndn) with occupied spin-down orbitals in n+1:2n.
@inline function occupied_orbitals!(up_occ::Vector{Int}, dn_occ::Vector{Int},
                                    d::Det{W}, n::Int, nup::Int, ndn::Int) where {W}
    @boundscheck begin
        length(up_occ) == nup || throw(ArgumentError("up_occ length must be $nup"))
        length(dn_occ) == ndn || throw(ArgumentError("dn_occ length must be $ndn"))
    end

    _fill_occ_range!(up_occ, d, 1, n, nup)
    _fill_occ_range!(dn_occ, d, n+1, 2n, ndn)

    return up_occ, dn_occ
end

function bitstring_print(d::Det{W}, n::Int) where {W}
    io = IOBuffer()
    for i in 2n:-1:1
        w = (i-1) >>> 6 + 1
        b = (i-1) & 0x3f
        bit = (d.words[w] >>> b) & 0x01
        print(io, bit)
    end
    return String(take!(io))
end

# Generate the Hartree-Fock determinant for n spatial orbitals, nup spin-up electrons,
# and ndn spin-down electrons.
function get_ground_state(n::Int, nup::Int, ndn::Int)
    W = cld(2n, 64)
    tmp = fill(UInt64(0), W) 

    @inline function setbit!(tmp::Vector{UInt64}, i::Int)
        w, b = wordbit(i)
        tmp[w] |= bitmask(b)
        return nothing
    end

    for i in 1:nup
        setbit!(tmp, i)
    end
    for i in 1:ndn
        setbit!(tmp, n + i)
    end

    return Det{W}(Tuple(tmp))
end
