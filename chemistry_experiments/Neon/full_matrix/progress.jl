#=============================================================================
progress.jl

Progress reporting for the full FCI build
=============================================================================#

using Printf
using Base.Threads

const HEARTBEAT = Ref{String}("")     # set to a path to also write progress there

function say(msg::AbstractString)
    println(msg)
    flush(stdout)
    if !isempty(HEARTBEAT[])
        open(HEARTBEAT[], "a") do io
            println(io, msg)
        end
    end
    return nothing
end

rss_gb() = Sys.maxrss() / 2^30

function fmt_time(s::Real)
    s = round(Int, s)
    h, r = divrem(s, 3600)
    m, sec = divrem(r, 60)
    return h > 0 ? @sprintf("%dh%02dm", h, m) : @sprintf("%dm%02ds", m, sec)
end

mutable struct Progress
    label::String
    total::Int
    done::Threads.Atomic{Int}
    t0::Float64
    tlast::Threads.Atomic{Float64}
    every::Float64          # seconds between printed lines
end

function Progress(label::AbstractString, total::Integer; every::Real = 30.0)
    t = time()
    say(@sprintf("[%s] starting, %d items, %d threads", label, total, Threads.nthreads()))
    return Progress(label, Int(total), Threads.Atomic{Int}(0), t,
                    Threads.Atomic{Float64}(t), Float64(every))
end

# Record `k` completed items. Only the worker passing `report = true` prints,
# and only once per `every` seconds, so the atomics stay cheap.
function bump!(p::Progress, k::Int; report::Bool = false)
    n = Threads.atomic_add!(p.done, k) + k
    if report
        t = time()
        if t - p.tlast[] >= p.every
            p.tlast[] = t
            el   = t - p.t0
            rate = n / max(el, 1e-9)
            eta  = rate > 0 ? (p.total - n) / rate : Inf
            say(@sprintf("[%s] %6.2f%%  %d/%d  elapsed %s  eta %s  %.3g items/s  rss %.1f GB",
                         p.label, 100n / p.total, n, p.total, fmt_time(el),
                         fmt_time(eta), rate, rss_gb()))
        end
    end
    return nothing
end

function finish!(p::Progress)
    el = time() - p.t0
    say(@sprintf("[%s] done in %s, rss %.1f GB", p.label, fmt_time(el), rss_gb()))
    return el
end

# ---------------------------------------------------------------------------
# Matrix-vector counter for the eigensolves.
# ---------------------------------------------------------------------------
mutable struct MatvecCounter
    label::String
    n::Int
    t0::Float64
end

function counting_operator(H, label::AbstractString)
    ctr = MatvecCounter(label, 0, time())
    nS = size(H, 1)
    f = function (y, x)
        t = @elapsed mul!(y, H, x)
        ctr.n += 1
        say(@sprintf("[%s] matvec %4d  %.1fs  elapsed %s  rss %.1f GB",
                     ctr.label, ctr.n, t, fmt_time(time() - ctr.t0), rss_gb()))
        return y
    end
    return LinearMap{Float64}(f, nS, nS; ismutating = true, issymmetric = true), ctr
end
