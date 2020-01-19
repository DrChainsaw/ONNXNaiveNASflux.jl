
function default_namestrat(vs::AbstractVector{<:AbstractVertex})
    all(isnamed, vs) && return v -> name(v)
    namegen = name_runningnr()
    ng(v::AbstractVertex) = namegen
    ng(f) = namegen(f)
    return ng
end

isnamed(v::AbstractVertex) = isnamed(base(v))
isnamed(v::CompVertex) = false
isnamed(v::InputVertex) = true
isnamed(v::MutationVertex) = isnamed(trait(v))
isnamed(t::DecoratingTrait) = isnamed(base(t))
isnamed(t) = false
isnamed(::NamedTrait) = true

function name_runningnr(namefun = genname)
    exists = Set{String}()

    return function(f)
        bname = genname(f)
        next = 0
        candname = bname * "_" * string(next)
        while candname in exists
            next += 1
            candname = bname * "_" * string(next)
        end
        push!(exists, candname)
        return candname
    end
end

genname(v::AbstractVertex) = name(v)
genname(f::F) where F = lowercase(string(F.name))
genname(s::AbstractString) = s
genname(f::Function) = lowercase(string(f))

recursename(f, namestrat) = recursename(f, namestrat(f))
recursename(f, fname::String) = fname

abstract type AbstractProbe end

Base.oftype(p::AbstractProbe, x) = x

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

newnamestrat(p::ProtoProbe, f, pname=p.name) = ProtoProbe(pname, p.shape, f, p.graph)
newfrom(p::ProtoProbe, outname::String, f=nothing) = ProtoProbe(outname, nextshape(p, f), p.nextname, p.graph)
nextshape(p, f) = p.shape

add!(gp::ONNX.Proto.GraphProto, np::ONNX.Proto.NodeProto) = push!(gp.node, np)

function add!(gp::ONNX.Proto.GraphProto, tp::ONNX.Proto.TensorProto)
    push!(gp.initializer, tp)
    push!(gp.input, ONNX.Proto.ValueInfoProto(tp.name, tp.dims))
end

graphproto() = ONNX.Proto.GraphProto(
node = ONNX.Proto.NodeProto[],
initializer =  ONNX.Proto.TensorProto[],
input =  ONNX.Proto.ValueInfoProto[],
output =  ONNX.Proto.ValueInfoProto[],
value_info =  ONNX.Proto.ValueInfoProto[]
)

function graphproto(g::CompGraph, outshapes = v -> (nout(v), layertype(v)), namestrat=default_namestrat(vertices(g)))
    gp = graphproto()
    pps = map(g.inputs) do v
        inputprotoprobe!(gp, recursename(v, namestrat), outshapes(v), namestrat)
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

function protoprobe(lt::FluxParLayer, l, pp, optype)
    lname = recursename(l, nextname(pp))
    wname, bname = lname .* ("_weight", "_bias")

    add!(pp, ONNX.Proto.NodeProto(
        input=[name(pp), wname, bname],
        output=[lname],
        name=lname,
        op_type=optype))
    add!(pp, ONNX.Proto.TensorProto(weights(l), wname))
    add!(pp, ONNX.Proto.TensorProto(bias(l), bname))

    ppout = actfun(lt, l)(newnamestrat(pp, f -> join([lname, genname(f)], "_"), lname))
    return newnamestrat(ppout, nextname(pp))
end

(l::Flux.Dense)(pp::AbstractProbe) = protoprobe(layertype(l), l, pp, "Gemm")
actfun(::FluxDense, l) = l.σ

(l::Flux.Conv)(pp::AbstractProbe) = protoprobe(layertype(l), l, pp, "Conv")
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


function axisfun(optype, pps::AbstractProbe...; dims, axname="axes")
    fname = recursename(lowercase(optype), nextname(pps[1]))

    np_axis = flux2numpydim.(dims, ndims_shape(shape(pps[1])))

    add!(pps[1], ONNX.Proto.NodeProto(
        input = collect(name.(pps)),
        output = [fname],
        name = fname,
        attribute = [ONNX.Proto.AttributeProto(axname, np_axis)],
        op_type = optype
    ))
    return newfrom(pps[1], fname)
end

scal2tup(x) = (x,)
scal2tup(x::Tuple) = x

Base.cat(pps::AbstractProbe...; dims) = axisfun("Concat", pps...; dims=dims, axname="axis")
Statistics.mean(pp::AbstractProbe; dims) = axisfun("ReduceMean", pp; dims=scal2tup(dims))
Base.dropdims(pp::AbstractProbe; dims) = axisfun("Squeeze", pp; dims=scal2tup(dims))
