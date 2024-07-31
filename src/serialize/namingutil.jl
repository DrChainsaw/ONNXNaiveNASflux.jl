# The naming strategies not well organized and should probably be rewritten. Until that happens, 
# here is a little synopsis of whats going on here:

# There are two main ways to create names:
# 1) By just using the function name (or lowercase name of the struct, e.g dense for Flux.Dense) 
# and add a running number after to make names unique (e.g. dense_1)

# 2) By intercepting names (e.g. the named tuple keys of Flux.Chain or the names of NaiveNASlibs
# vertices) and make sure all ops "below" in the call stack (e.g. the actual layers) use those names

# Both strategies will be attached to the input ProtoProbe as the (badly named) field nextname 
# when we start traversing the call stack.
# The strategy might be changed based on what we encounter. One example of when this happens
# is for the activation functions of the Flux layers which need to be separate nodes in ONNX
# so we make them inherit the name of the layer node (e.g. dense, conv) and suffix them with
# the name of the function.

# For strategy 1) we just use NameRunningNr and thats the end of the story, no fuss!

# For strategy 2) however, there are quite a few moving pieces, three to be exact. 
# The whole thing is probably overengineered and could/should be done in a simpler way.

# The first entity is the strategy itself: NamedNodeContext which signals that we are in the 
# business of trying to find named things in the call stack. It also remembers what the last 
# name we saw was (and the hierarchy of names in case of nesting, e.g. a Chain inside a Parallel
# inside a Chain). Note that for the sake of safety NamedNodeContext wraps a NameRunningNr to
# ensure that we don't end up with duplicate names.

# In addition, whenever we create an input ProtoProbe with the nextname strategy being a 
# NamedNodeContext we wrap the ProtoProbe in a NameInterceptProbe which is the second entity in this 
# little scheme. This allows us to dispatch on high level (i.e non-primitive) function calls such as
#  Chain, Parallel, SkipConnection and MutationVertex to just catch the names we are after as they 
# are not visible on the primitive (e.g. Flux.Dense) level.

# We could also have tried to dispatch on something like 
# (c::Chain)(pp::ProtoProbe{<:Any, <:NamedNodeContext}), but this has the following problem: What do
# we do then after we have caught the name? We then want Chain to just do its thing on pp, but if 
# we call c(pp) we just end up with infinite recursion. 

# We also don't want to replace the naming strategy at this point since we want to handle nested 
# name holders (i.e the Chain inside a Parallel inside a Chain, or a Chain inside a MutationVertex
# if you'd like).

# Instead we just unwrap the NameInterceptProbe whenever we hit a name holder before we forward the
# call.

# The annoying thing here is that when we hold an object like a Flux.Parallel in our hand we know 
# its structure and can see how it names things (e.g. through Functors.fmap_with_path), but we 
# don't know for sure in what order the stuff inside it will be called when we call it as a function 
# (ok, in practice we know since it has a doc-string contract on what it does, but we also don't 
# want to reimplement it and all other things we want to catch names from).

# To prevent that we need to encode in this library what each possible high level name holder (i.e 
# Chain, Parallel, etc.) does when called, we use a third entity, the NamedFunction, which just 
# wraps anything callable (well, it does not care if it is callable or not, but obviously things 
# will not work if it is not) and the name it has. We then use a shallow Functors.fmap_with_path 
# to wrap all named things (with methods) inside the name holder in a NamedFunction. 

# Yup, you read that right, for example, if Chain(layer1=l1, Layer2=l2) is called with a 
# NameInterceptProbe as input it will first be fmap_with_path:ed into 
# Chain(layer1=NamedFunction(l1, "layer1"), layer2=NamedFunction(l2, "layer2")), then the new chain
# will be called with the ProtoProbe wrapped inside the NameInterceptProbe.

# When a NamedFunction is called with an AbstractProbe as input, it does two things:
# 1) Wrap the AbstractProbe in a NameInterceptProbe so we can intercept nested name holders (e.g. a
#    Chain inside a Parallel inside a Chain).
# 2) Create a new naming strategy for all ONNX ops encountered when calling the function wrapped
#    in the NamedFunction which is to use the name of the NamedFunction.

# Note that at step 2) we append the name to any previously encountered names so that nested name holders 
# get the correct path. In other words, the new naming strategy is a new NamedNodeContext where we have 
# appended the name of the named node to the prefix.

# One option to simplify could have been to just fmap_with_path the entire model before calling it.
# This might turn out to be simpler. The reasons I did not go for it was:
# 1) There would not be any way to name things inside things that are not fmap:able
# 2) It is still not easy to know which things are named (although we could just always use 
#    fieldnames)
# 3) Less control over what we actually want to wrap inside a NamedFunction (maybe we could use
#    dispatch for this when walking though)
# 4) The functor structure for CompGraphs is pretty horrible (but we anyways don't use the 
#    fmap_with_path method for it since the vertices are the universal name holders for it)
# None of the above seem like showstoppers, the current way was just the path of least resistance.



default_namestrat(f) = name_runningnr()
default_namestrat(g::CompGraph) = default_namestrat(vertices(g))
function default_namestrat(vs::AbstractVector{<:AbstractVertex})
    all(isnamed, vs) && length(unique(name.(vs))) == length(name.(vs)) && return NamedNodeContext("", name_runningnr(;addtofirst=false))
    # TODO: Maybe we should use NamedNodeContext here as well and just add runningnumbers to duplicated names?
    name_runningnr()
end

const NameType = Union{Symbol, AbstractString}

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

struct NamedFunction{F}
    f::F
    name::String
end
NaiveNASlib.base(n::NamedFunction) = n.f
NaiveNASlib.name(n::NamedFunction) = n.name
(n::NamedFunction)(args...; kwargs...) = n.f(args...; kwargs...)

const NamedNode = Union{NamedFunction, MutationVertex}

struct NamedNodeContext{F}
    prefix::String
    namegen::F
end

Base.Broadcast.broadcastable(n::NamedNodeContext) = Ref(n)

(ctx::NamedNodeContext)(args...) = isempty(ctx.prefix) ? ctx.namegen(args...) : ctx.namegen(ctx.prefix)
function (ctx::NamedNodeContext)(v::Union{NamedNode, AbstractVertex}) 
    nodename = name(v)
    # Maybe add some field in NamedFunction to indicate wether it is an array element or a field instead of 
    # startswith here
    sep = isempty(ctx.prefix) || startswith(nodename, '[') ? "" : "."
    @set ctx.prefix = string(ctx.prefix, sep, name(v))
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

