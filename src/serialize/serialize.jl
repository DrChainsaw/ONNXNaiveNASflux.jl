
"""
    onnx(filename::AbstractString, f, args...; kwargs...)
    onnx(io::IO, f, args...; kwargs...)

Serialize the result of `modelproto(f, args...; kwargs...)` to a file with path `filename` or to `io`.

See [`modelproto`](@ref) for description of arguments.
"""
onnx(filename::AbstractString, f, args...; kwargs...) = onnx(filename, modelproto(f, args...; kwargs...))
onnx(io::IO, f, args...; kwargs...) = onnx(io, modelproto(f, args...; kwargs...))

"""
    onnx(filename::AbstractString, mp::ONNX.Proto.ModelProto)
    onnx(io::IO, mp::ONNX.Proto.ModelProto)

Serialize the given [`ONNX.Proto.ModelProto`](@ref) to a file with path `filename` or to `io`.
"""
onnx(filename::AbstractString, mp::ONNX.Proto.ModelProto) = open(io -> onnx(io, mp), filename, "w")
onnx(io::IO, mp::ONNX.Proto.ModelProto) = ONNX.writeproto(io, mp)


"""
    modelproto(f; namestrat=name_runningnr(), posthook=validate, kwargs...)
    modelproto(f, inshapes::Tuple...; namestrat = name_runningnr(), posthook=validate, kwargs...)
    modelproto(f, indata::Pair{String, <:Any}...; namestrat = name_runningnr(), posthook=validate, kwargs...)

Return an [`ONNX.Proto.ModelProto`](@ref) from `f`.

Argument `inshapes` are size tuples representing the shape of each input. An attempt to infer sizes will be made if not  provided.
Argument `indata` are pairs mapping names to size tuples. Names will be created automatically if not provided.

Argument `namestrat` determines how nodes in the graph shall be named. Other keyword arguments are passed to the `ModelProto`.

Argument `posthook` will be called with the created `ONNX.Proto.ModelProto` as argument before returning it.

Other keyword arguments will be passed to `ONNX.Proto.ModelProto`.
"""
modelproto(f; kwargs...) = modelproto(f, infer_inshapes(f)...; kwargs...)
modelproto(f, inshapes::Union{Tuple, Missing}...;kwargs...) = modelproto(f, ("data_" .* string.(0:length(inshapes)-1) .=> inshapes)...; kwargs...)
function modelproto(f, indata::Pair{String, <:Any}...; namestrat = name_runningnr(), posthook=validate, kwargs...)
    mp = modelproto(;kwargs...)
    mp.graph = graphproto(f, indata...;namestrat=namestrat)
    posthook(mp)
    return mp
end

"""
    modelproto(g::CompGraph; outshape = shape, namestrat=default_namestrat(g); , posthook=validate, kwargs...)

Return an [`ONNX.Proto.ModelProto`](@ref) from `g`.

Argument `outshape` is a function which returns a size tuple representing the shape of the output of a given `AbstractVertex`.

Argument `namestrat` determines how nodes in the graph shall be named. Other keyword arguments are passed to the `ModelProto`.

Argument `posthook` will be called with the created `ONNX.Proto.ModelProto` as argument before returning it.

Other keyword arguments will be passed to `ONNX.Proto.ModelProto`.
"""
function modelproto(g::CompGraph; outshape = shape, namestrat=default_namestrat(g), posthook=validate, kwargs...)
    mp = modelproto(;kwargs...)
    mp.graph = graphproto(g, outshape, namestrat)
    posthook(mp)
    return mp
end

function infer_inshapes(f)
    ml = methods(f);
    for m in ml.ms
        m.sig isa DataType && return Tuple(infer_shape.(m.sig.types[2:end]))
    end
    return ntuple(i -> missing, ml.mt.max_args)
end
infer_shape(::Type{<:Any}) = missing
infer_shape(::Type{<:AbstractArray{T,N}}) where {T,N} = ntuple(i -> missing, N)

modelproto(;kwargs...) = ONNX.Proto.ModelProto(;
    ir_version=6,
    opset_import=[ONNX.Proto.OperatorSetIdProto(version=12)],
    producer_name="ONNXmutable.jl",
    producer_version=string(Pkg.Types.Context().env.project.version), # TODO: Ugh....
    kwargs...)




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
Base.ndims(p::AbstractProbe) = length(shape(p))
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
    Used to get activation functions as [`ONNX.Proto.AttributeProto`](@ref)s.
"""
struct ActivationAttributeProbe end

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
graphproto(g::CompGraph, outshape = shape, namestrat=default_namestrat(g)) = graphproto(g, (recursename.(g.inputs, namestrat) .=> shape.(g.inputs))...;namestrat=namestrat)

"""
    graphproto(f, indata::Pair{String, <:Any}...; namestrat = name_runningnr())

Return an [`ONNX.Proto.GraphProto`](@ref) from `g`.

Argument indata are name => shape pairs for the input data.

Argument `namestrat` determines how nodes shall be named.
"""
function graphproto(f, indata::Pair{String, <:Any}...; namestrat = name_runningnr())
    gp = graphproto()
    pps = map(indata) do (name, shape)
        inputprotoprobe!(gp, name, shape, namestrat)
    end

    outpps = f(pps...)

    add_output!.(outpps)

    return gp
end

# Only purpose is to snag the name in case this is the naming strategy
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


(m::Flux.Recur)(pps::AbstractProbe...) = m.cell(m.state, pps...)

(l::Flux.RNNCell)(h, pp::AbstractProbe) = recurrent_node(l, pp, "RNN")
(l::Flux.LSTMCell)(h, pp::AbstractProbe) = recurrent_node(l, pp, "LSTM")

function recurrent_node(l, pp, optype)
    lname = recursename(l, nextname(pp))
    wname, rname, bname = lname .* ("_W", "_R", "_B")

    hsattrib = ONNX.Proto.AttributeProto("hidden_size", size(l.Wh, 1))

    add!(pp, ONNX.Proto.NodeProto(
        input=[name(pp), wname, rname, bname],
        output=[lname],
        name=lname,
        attribute = push!(activation_attrib(l), hsattrib),
        op_type=optype))
    # Flux weights are of shape [hidden_size, input_size]
    # ONNX wants them on the form [num_directions, hidden_size, input_size] (where num_directions is 2 for bidirectional else 1)
    # To spice things up a bit, all julia arrays are saved in reverse order, i.e we need to create a TensorProto from an array with the arrangement [input_size, hidden_size, num_directions].
    # First transpose the weights into [input_size, hidden_size], then reshape by adding 1 extra dimension
    Wi = permutedims(l.Wi)
    add!(pp, ONNX.Proto.TensorProto(reshape(Wi, size(Wi)...,1), wname))
    Wh = permutedims(l.Wh)
    add!(pp, ONNX.Proto.TensorProto(reshape(Wh, size(Wh)..., 1), rname))
    # ONNX has a separate bias for the recurrent part and wants the concatenation of input and recurrent biases.
    # We'll just hard code it to zeros. Doesn't matter which part is which as they are just added together in the ONNX expression for RNNs.
    b = reshape(l.b, :, 1)
    add!(pp, ONNX.Proto.TensorProto(vcat(b, zeros(eltype(b), size(b))), bname))

    return newfrom(pp, lname)
end

activation_attrib(l) = l.σ(ActivationAttributeProbe())
activation_attrib(l::Flux.LSTMCell) = ONNX.Proto.AttributeProto[] #Only default values supported by Flux

Base.tanh(::ActivationAttributeProbe) = [ONNX.Proto.AttributeProto("activations", "Tanh")]
Flux.elu(::ActivationAttributeProbe, α=1) = ONNX.Proto.AttributeProto.(["activations", "activation_alpha"], ["Elu", α])

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
(l::Flux.MeanPool)(pp::AbstractProbe) = attribfun("AveragePool", pp; attributes = ONNX.Proto.AttributeProto.(["kernel_shape", "pads", "strides"], [l.k, l.pad, l.stride]))
(l::Flux.Dropout)(pp::AbstractProbe) = attribfun("Dropout", pp; attributes = [ONNX.Proto.AttributeProto("ratio", l.p)])


function globalmeanpool(pp::AbstractProbe, wrap)
     gpp = attribfun("GlobalAveragePool", pp)
     ppnext = newnamestrat(gpp, f -> join([gpp.name, genname(f)], "_"), gpp.name)
     wpp = wrap(ppnext)
     return newnamestrat(wpp, nextname(gpp))
end

Base.:+(pps::AbstractProbe...) = attribfun("Add", pps...)


function axisfun(optype, pps::AbstractProbe...; dims, axname="axes", fshape=identity)
    fname = recursename(lowercase(optype), nextname(pps[1]))

    attrib = if isempty(dims)
        ONNX.Proto.AttributeProto[]
    else
        np_axis = flux2numpydim.(dims, ndims(pps[1]))
        [ONNX.Proto.AttributeProto(axname, np_axis)]
    end

    add!(pps[1], ONNX.Proto.NodeProto(
        input = collect(name.(pps)),
        output = [fname],
        name = fname,
        attribute = attrib,
        op_type = optype
    ))
    return newfrom(pps[1], fname, fshape)
end

scal2tup(x) = (x,)
scal2tup(x::Tuple) = x

Base.cat(pps::AbstractProbe...; dims) = axisfun("Concat", pps...; dims=dims, axname="axis")
Statistics.mean(pp::AbstractProbe; dims=()) = axisfun("ReduceMean", pp; dims=scal2tup(dims))
Base.dropdims(pp::AbstractProbe; dims) = axisfun("Squeeze", pp; dims=scal2tup(dims), fshape = s -> rmdims(s, dims))

reshape_keepshape(pp::AbstractProbe, shape) = reshape(pp, shape)
Base.reshape(pp::AbstractProbe, shape...) = reshape(pp, shape)
function Base.reshape(pp::AbstractProbe, shape::Tuple)
    fname = recursename("Reshape", nextname(pp))
    sname = fname .* "_shape"
    fluxshape = collect(map(s -> s == Colon() ? -1 : s, shape))

    add!(pp, ONNX.Proto.NodeProto(
        input=[name(pp), sname],
        output=[fname],
        name=fname,
        op_type="Reshape"))
    add!(pp, ONNX.Proto.TensorProto(reverse(fluxshape), sname))

    fshape = function(s)
        return map(enumerate(fluxshape)) do (ind, new)
            new == -1 && return missing # CBA to figure out how to do this...
            new == 0 && return s[ind]
            return new
        end
    end

    return newfrom(pp, fname, fshape)
end
expanddims(p::AbstractProbe, x, dims) = p
