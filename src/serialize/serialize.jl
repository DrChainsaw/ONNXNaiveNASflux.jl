
"""
    AbstractProbe

Abstract base class for probes used for serialization.

Idea is that probe will "record" all seen operations based on how methods for those operations on the probe is defined.
"""
abstract type AbstractProbe end

# Called by several activation functions
Base.oftype(p::AbstractProbe, x) = x

"""
    ProtoProbe <: AbstractProbe
    ProtoProbe(name, shape, nextname, graph)

Probe which builds an [`ONNX.Proto.GraphProto`](@ref) from seen operations.
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
add_output!(p::ProtoProbe) = push!(p.graph.output, ONNX.Proto.ValueInfoProto(name(p), shape(p)))
function inputprotoprobe!(gp, name, shape, namestrat)
    push!(gp.input, ONNX.Proto.ValueInfoProto(name, shape))
    pp = ProtoProbe(name, shape, namestrat, gp)
end

"""
    newnamestrat(p::ProtoProbe, f, pname=p.name)

Return a new `ProtoProbe` from `p` with `nextname = f` and name `pname`.
"""
newnamestrat(p::ProtoProbe, f, pname=p.name) = ProtoProbe(pname, p.shape, f, p.graph)

"""
    newfrom(p::ProtoProbe, outname::AbstractString, fshape=identity)

Return a new `ProtoProbe` with name `outname`. Argument `fshape` can be used to determine a new shape.
"""
newfrom(p::ProtoProbe, outname::AbstractString, fshape=identity) = ProtoProbe(outname, nextshape(p, fshape), p.nextname, p.graph)

nextshape(p::AbstractProbe, f::Function) = f(shape(p))


add!(gp::ONNX.Proto.GraphProto, np::ONNX.Proto.NodeProto) = push!(gp.node, np)

function add!(gp::ONNX.Proto.GraphProto, tp::ONNX.Proto.TensorProto)
    push!(gp.initializer, tp)
    push!(gp.input, ONNX.Proto.ValueInfoProto(tp.name, tp.dims))
end

"""
    graphproto()

Return an [`ONNX.Proto.GraphProto`](@ref) with all fields initialized to empty arrays.
"""
graphproto() = ONNX.Proto.GraphProto(
node = ONNX.Proto.NodeProto[],
initializer =  ONNX.Proto.TensorProto[],
input =  ONNX.Proto.ValueInfoProto[],
output =  ONNX.Proto.ValueInfoProto[],
value_info =  ONNX.Proto.ValueInfoProto[]
)

"""
    graphproto(g::CompGraph, outshape = shape, namestrat=default_namestrat(g))

Return an [`ONNX.Proto.GraphProto`](@ref) from `g`.

Argument `outshape` is a function which returns the shape of an `AbstractVertex`.

Argument `namestrat` determines how nodes shall be named.
"""
function graphproto(g::CompGraph, outshape = shape, namestrat=default_namestrat(g))
    gp = graphproto()
    pps = map(g.inputs) do v
        inputprotoprobe!(gp, recursename(v, namestrat), outshape(v), namestrat)
    end

    outpps = g(pps...)

    add_output!.(outpps)

    return gp
end

function (v::NaiveNASlib.MutationVertex)(pps::AbstractProbe...)
    ppsname = map(pps) do pp
        newnamestrat(pp, nextname(pp)(v))
    end
    ppout = base(v)(ppsname...)
    return newnamestrat(ppout, nextname(pps[1]))
end

function weightlayer(lt::FluxParLayer, l, pp, optype;attributes = ONNX.Proto.AttributeProto[])
    lname = recursename(l, nextname(pp))
    wname, bname = lname .* ("_weight", "_bias")

    add!(pp, ONNX.Proto.NodeProto(
        input=[name(pp), wname, bname],
        output=[lname],
        name=lname,
        attribute = attributes,
        op_type=optype))
    add!(pp, ONNX.Proto.TensorProto(weights(l), wname))
    add!(pp, ONNX.Proto.TensorProto(bias(l), bname))

    ppout = actfun(lt, l)(newnamestrat(pp, f -> join([lname, genname(f)], "_"), lname))
    return newnamestrat(ppout, nextname(pp))
end

(l::Flux.Dense)(pp::AbstractProbe) = weightlayer(layertype(l), l, pp, "Gemm")
actfun(::FluxDense, l) = l.σ

(l::Flux.Conv)(pp::AbstractProbe) = weightlayer(layertype(l), l, pp, "Conv"; attributes= ONNX.Proto.AttributeProto.([ "pads", "strides", "dilations"], [l.pad, l.stride, l.dilation]))
actfun(::FluxConv, l) = l.σ

function(l::Flux.BatchNorm)(pp::AbstractProbe)
    lname = recursename(l, nextname(pp))
    γname, βname, μname, σ²name = lname .* ("_scale", "_bias", "_mean", "_var")

    add!(pp, ONNX.Proto.NodeProto(
        input=[name(pp), γname, βname, μname, σ²name],
        output=[lname],
        name=lname,
        attribute = ONNX.Proto.AttributeProto.(["epsilon", "momentum"], [l.ϵ, l.momentum]),
        op_type="BatchNormalization"))
    add!(pp, ONNX.Proto.TensorProto(l.γ, γname))
    add!(pp, ONNX.Proto.TensorProto(l.β, βname))
    add!(pp, ONNX.Proto.TensorProto(l.μ, μname))
    add!(pp, ONNX.Proto.TensorProto(l.σ², σ²name))

    ppout = actfun(layertype(l), l)(newnamestrat(pp, f -> join([lname, genname(f)], "_"), lname))
    return newnamestrat(ppout, nextname(pp))
end
actfun(::FluxBatchNorm, l) = l.λ

function attribfun(optype, pps::AbstractProbe...; attributes = ONNX.Proto.AttributeProto[])
    lname = recursename(lowercase(optype), nextname(pps[1]))
    add!(pps[1], ONNX.Proto.NodeProto(
    input = collect(name.(pps)),
    output = [lname],
    name=lname,
    attribute = attributes,
    op_type= optype))
    return newfrom(pps[1], lname)
end

Flux.relu(pp::AbstractProbe) = attribfun("Relu", pp)
Flux.elu(pp::AbstractProbe, α=1) = attribfun("Elu", pp; attributes = [ONNX.Proto.AttributeProto("alpha", α)])
Flux.selu(pp::AbstractProbe) = attribfun("Selu", pp)
Flux.selu(pp::AbstractProbe, γ, α) = attribfun("Selu", pp; attributes = ONNX.Proto.AttributeProto.(["gamma", "alpha"], [γ, α]))
(l::Flux.MaxPool)(pp::AbstractProbe) = attribfun("MaxPool", pp; attributes = ONNX.Proto.AttributeProto.(["kernel_shape", "pads", "strides"], [l.k, l.pad, l.stride]))


function globalmeanpool(pp::AbstractProbe, wrap)
     gpp = attribfun("GlobalAveragePool", pp)
     ppnext = newnamestrat(gpp, f -> join([gpp.name, genname(f)], "_"), gpp.name)
     wpp = wrap(ppnext)
     return newnamestrat(wpp, nextname(gpp))
end

Base.:+(pps::AbstractProbe...) = attribfun("Add", pps...)


function axisfun(optype, pps::AbstractProbe...; dims, axname="axes", fshape=identity)
    fname = recursename(lowercase(optype), nextname(pps[1]))

    np_axis = flux2numpydim.(dims, length(shape(pps[1])))

    add!(pps[1], ONNX.Proto.NodeProto(
        input = collect(name.(pps)),
        output = [fname],
        name = fname,
        attribute = [ONNX.Proto.AttributeProto(axname, np_axis)],
        op_type = optype
    ))
    return newfrom(pps[1], fname, fshape)
end

scal2tup(x) = (x,)
scal2tup(x::Tuple) = x

Base.cat(pps::AbstractProbe...; dims) = axisfun("Concat", pps...; dims=dims, axname="axis")
Statistics.mean(pp::AbstractProbe; dims) = axisfun("ReduceMean", pp; dims=scal2tup(dims))
Base.dropdims(pp::AbstractProbe; dims) = axisfun("Squeeze", pp; dims=scal2tup(dims), fshape = s -> rmdims(s, dims))
