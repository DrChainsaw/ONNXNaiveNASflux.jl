"""
    AbstractProbe

Abstract base class for probes used for serialization.

Idea is that probe will "record" all seen operations based on how methods for those operations on the probe is defined.
"""
abstract type AbstractProbe end

nextshape(p::AbstractProbe, f::Function) = f(shape(p))
Base.ndims(p::AbstractProbe) = length(shape(p))

"""
    WrappedProbe

Abstract class which promises that it wraps another `AbstractProbe` to which it delegates most of its tasks. 
"""
abstract type WrappedProbe <: AbstractProbe end

unwrap(p::WrappedProbe) = p.wrapped
unwrap(p::WrappedProbe, dt::Type{<:AbstractProbe}) = p isa dt ? unwrap(p) : rewrap(p, unwrap(unwrap(p), dt))
unwrap(::AbstractProbe, dt::Type{<:AbstractProbe}) = throw(ArgumentError("Failed to unwrap probe of type $dt"))

Base.Broadcast.broadcastable(p::WrappedProbe) = Ref(p)

NaiveNASlib.name(p::WrappedProbe) = NaiveNASlib.name(unwrap(p))

nextname(p::WrappedProbe) = nextname(unwrap(p))
add!(p::WrappedProbe, args...) = add!(unwrap(p), args...)
shape(p::WrappedProbe) = shape(unwrap(p))
add_output!(p::WrappedProbe) = add_output!(unwrap(p))
newnamestrat(p::WrappedProbe, args...) = rewrap(p, newnamestrat(unwrap(p), args...))
newfrom(p::WrappedProbe, args...) = rewrap(p, newfrom(unwrap(p), args...))
nodeprotos(p::WrappedProbe) = nodeprotos(unwrap(p))
graphproto(p::WrappedProbe) = graphproto(unwrap(p))

# Called by several activation functions
Base.oftype(::AbstractProbe, x) = x

"""
    ProtoProbe <: AbstractProbe
    ProtoProbe(name, shape, nextname, graph)

Probe which builds an [`ONNX.GraphProto`](@ref) from seen operations.
"""
struct ProtoProbe{N,F,P,S} <: AbstractProbe
    name::N
    shape::S
    nextname::F
    graph::P
end
Base.Broadcast.broadcastable(p::ProtoProbe) = Ref(p)
NaiveNASlib.name(p::ProtoProbe) = p.name
nextname(p::ProtoProbe) = p.nextname
add!(p::ProtoProbe, n) = add!(p.graph, n)
shape(p::ProtoProbe) = p.shape
add_output!(p::ProtoProbe) = push!(p.graph.output, ONNX.ValueInfoProto(name(p), shape(p)))

inputprotoprobe!(args...) = _inputprotoprobe!(args...)
function _inputprotoprobe!(gp, name, shape, namestrat)
    push!(gp.input, ONNX.ValueInfoProto(name, shape))
    ProtoProbe(name, shape, namestrat, gp)
end

"""
    newnamestrat(p::ProtoProbe, f, pname=p.name)

Return a new `ProtoProbe` from `p` with `nextname = f` and name `pname`.
"""
newnamestrat(p::ProtoProbe, f, pname=p.name) = ProtoProbe(pname, p.shape, f, p.graph)

"""
    newfrom(p::ProtoProbe, outname::AbstractString, fshape=identity)

Return a new `ProtoProbe` with name `outname`. Argument `fshape` is used to determine a new shape (typically a function).
"""
newfrom(p::ProtoProbe, outname::AbstractString, fshape) = ProtoProbe(outname, nextshape(p, fshape), p.nextname, p.graph)

add!(gp::ONNX.GraphProto, np::ONNX.NodeProto) = push!(gp.node, np)

add!(gp::ONNX.GraphProto, tp::ONNX.TensorProto) = push!(gp.initializer, tp)

nodeprotos(p::ProtoProbe) = p.graph.node
graphproto(p::ProtoProbe) = p.graph

## Don't forget to check if new methods need to be added for any WrappedProbe implementations if you add something here!

# Special purpose probes. Stuff below here might be tailored to a single use case. Reuse with care!

"""
    ActivationAttributeProbe

Used to get activation functions as [`ONNX.AttributeProto`](@ref)s.
"""
struct ActivationAttributeProbe end

"""
    IncompatibleProbe

Mark an output as incompatible with ONNX. Any attempt to make use of the output will result in an error.

Use case is for functions which have multiple outputs where only some subset is compatible with ONNX. One
such example is `Flux.LSTM` which outputs a two-element tuple where the first element is consistent with the first
ONNX output while the second element is not.
    
Note that subsequent operations may reformat the output and by that make it compatible with ONNX. Such operations
can use `unwrap(p, IncompatibleProbe)` to remove the `IncompatibleProbe` from the hierarchy.
"""
struct IncompatibleProbe{P} <: WrappedProbe
    wrapped::P
    msg::String
end
# Feel free to remove this exception by putting a function like this somewhere after op which generated this:
# rm_incompat(x) = x
# rm_incompat(p::ONNXNaiveNASflux.AbstractProbe) = ONNXNaiveNASflux.unwrap(p, ONNXNaiveNASflux.IncompatibleProbe) 
IncompatibleProbe(p::AbstractProbe) = IncompatibleProbe(p, string("Output ", name(p), " does not comply with ONNX and can't be exported (as the resulting model would be invalid)."))

add!(p::IncompatibleProbe, args...) = throw(ArgumentError(p.msg))
add_output!(p::IncompatibleProbe) = throw(ArgumentError(p.msg))
rewrap(ip::IncompatibleProbe, p) = IncompatibleProbe(p, ip.msg)

"""
    OutputSelectProbe

Stores data about a past ops so that subsequent ops can be used to select outputs.

Currently completely tailored towards the recurrent layers where Flux generally outputs
all timesteps only while ONNX versions have additional optional outputs with only the 
last timestep.

Probably very difficult to use in for other purposes. Consider making a new probe type instead.
"""
struct OutputSelectProbe{P, T} <: WrappedProbe
    wrapped::P
    originname::String
    outputused::T
    lifeleft::Int
end

function newfrom(p::OutputSelectProbe, args...) 
    newp= newfrom(unwrap(p), args...)
    p.lifeleft < 1 && return newp # Remove if noone wants to select outputs
    setproperties(p, wrapped=newp, lifeleft=p.lifeleft-1) 
end
rewrap(p::OutputSelectProbe, pwrapped) = @set p.wrapped = pwrapped

function select_output!(p::OutputSelectProbe, nr, suffix, newshape=identity)
    newname = string(p.originname, suffix)
    allnodes = nodeprotos(p) 
    node_i = findfirst(==(p.originname) ∘ name, allnodes)
    p.outputused[nr] = true

    @assert !isnothing(node_i) "Could not find node with name $(p.originname)!"

    node_to_add_output = allnodes[node_i]
    while length(node_to_add_output.output) < nr-1
        push!(node_to_add_output.output, "")
    end

    if length(node_to_add_output.output) < nr
        push!(node_to_add_output.output, newname)
    else
        # Note: Untested as of now since there is op where this can happen. 
        # I think this is the right thing to do though...
        node_to_add_output.output[nr] = newname
    end
    
    if nr > 1 && !p.outputused[nr - 1]
        # Previous output was not used. Since we assumed (?) that the next node
        # will use output 1 we need to change this
        for subsequent_node in @view allnodes[node_i+1:end]
            for i in eachindex(subsequent_node.input)
                if subsequent_node.input[i] == p.originname
                    subsequent_node.input[i] = newname
                end
            end
        end
    end
    
    # The first output is assumed to have the correct probe, but subsequent ops are not
    # Maybe someday in the future there will be some OP for which the Flux/Julia output
    # maps to some output other than 1 and then we might need to generalize this (e.g store 
    # the ONNX output nr in the probe).
    if nr > 1 
        @set p.wrapped=newfrom(p.wrapped, newname, newshape)
    else 
        p
    end
end
select_output!(p::WrappedProbe, args...) = rewrap(p, select_output!(unwrap(p), args...))
select_output!(p::AbstractProbe, args...) = throw(ArgumentError("Failed to set output for $(name(p))! Forgot to wrap it in a SelectOutputProbe?"))

struct IgnoreDropDimsProbe{P} <: WrappedProbe
    # TODO: Add dim number to ignore?
    wrapped::P
end
rewrap(::IgnoreDropDimsProbe, p) = IgnoreDropDimsProbe(p)
Base.dropdims(p::IgnoreDropDimsProbe; kwargs...) = p
function add!(p::IgnoreDropDimsProbe, n::ONNX.NodeProto) 
    if optype(n) == "Squeeze"
        throw(ArgumentError("Tried to add Squeeze to IgnoreDropDims. It might have been wrapped in something else..."))
    end
    add!(unwrap(p), n)
end
function (l::AddSingletonDim)(p::AbstractProbe)
    out = l.wrapped(IgnoreDropDimsProbe(p))
    _apply_unwrap(out, IgnoreDropDimsProbe)
end

_apply_unwrap(p::AbstractProbe, dt) = unwrap(p, dt)
_apply_unwrap(t::Tuple, dt) = map(p -> _apply_unwrap(p, dt), t)