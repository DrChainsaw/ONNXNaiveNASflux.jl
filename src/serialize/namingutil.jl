
default_namestrat(f) = name_runningnr()
default_namestrat(g::CompGraph) = default_namestrat(vertices(g))
function default_namestrat(vs::AbstractVector{<:AbstractVertex})
    # Even if all vertices have unique names, we can't be certain that no vertex produces more than one node
    # Therefore, we must take a pass through name_runningnr for each op even after we have mapped the vertex to a nextname. This is the reason for the v -> f -> namegen(v, "") wierdness

    all(isnamed, vs) && length(unique(name.(vs))) == length(name.(vs)) && return NamedNodeContext("", name_runningnr(;addtofirst=false))
    namegen = name_runningnr()
    ng(v::AbstractVertex) = namegen
    ng(f) = namegen(f)
    return ng
end

function default_namestrat(c::Chain)
    !(eltype(keys(c)) <: NameType) && return name_runningnr()
    NamedNodeContext("", name_runningnr(;addtofirst=false))
end

struct NameRunningNr{F}
    addtofirst::Bool
    init::Int
    runningnrs::Dict{String, Int}
    namefun::F
end

Base.Broadcast.broadcastable(n::NameRunningNr) = Ref(n)

function (n::NameRunningNr)(f)
    bname = n.namefun(f)
    nextnr = get(n.runningnrs, bname, n.init)
    n.runningnrs[bname] = nextnr + 1
    return if nextnr != n.init || n.addtofirst
        string(bname, "_", nextnr)
    else
        bname
    end
end

name_runningnr(namefun = genname; addtofirst=true, init=0) = NameRunningNr(addtofirst, init, Dict{String, Int}(), namefun)

const NameType = Union{Symbol, AbstractString}

struct NamedNodeContext{F}
    prefix::String
    namegen::F
end

Base.Broadcast.broadcastable(n::NamedNodeContext) = Ref(n)

(ctx::NamedNodeContext)(args...) = isempty(ctx.prefix) ? ctx.namegen(args...) : ctx.namegen(ctx.prefix)

function (ctx::NamedNodeContext)(v::AbstractVertex) 
    @set ctx.prefix = string(ctx.prefix, isempty(ctx.prefix) ? "" : ".", name(v))
end

chainlayername(f, ::Any, ::Any) = f
function chainlayername(ctx::NamedNodeContext, name::NameType, layer) 
    @set ctx.prefix = string(ctx.prefix, isempty(ctx.prefix) ? "" : ".", name)
end
function chainlayername(ctx::NamedNodeContext{<:NameRunningNr}, nr::Integer, layer) 
    !isempty(ctx.prefix) && return @set ctx.prefix = string(ctx.prefix, '[', nr, ']')
    return name_runningnr() 
end

isnamed(v::AbstractVertex) = isnamed(base(v))
isnamed(v::CompVertex) = false
isnamed(v::InputVertex) = true
isnamed(v::SourceVertex) = true
isnamed(v::MutationVertex) = isnamed(trait(v))
isnamed(t::DecoratingTrait) = isnamed(base(t))
isnamed(t) = false
isnamed(::NamedTrait) = true

genname(v::AbstractVertex) = name(v)
genname(::F) where F = lowercase(string(nameof(F)))
genname(s::AbstractString) = s
genname(f::Function) = lowercase(string(f))

recursename(f, namestrat) = recursename(f, namestrat(f))
recursename(f, fname::String) = fname
function recursename(f, ctx::NamedNodeContext) 
    res = ctx(f)
    res isa NamedNodeContext && return recursename(f, res.prefix)
    res
end

