
"""
    save(filename::AbstractString, f, args...; kwargs...)
    save(io::IO, f, args...; kwargs...)

Serialize the result of `modelproto(f, args...; kwargs...)` to a file with path `filename` or to `io`.

See [`modelproto`](@ref) for description of arguments.
"""
save(filename::AbstractString, f, args...; modelname=filename, kwargs...) = save(filename, modelproto(f, args...; modelname=modelname, kwargs...))
save(io::IO, f, args...; kwargs...) = save(io, modelproto(f, args...; kwargs...))

"""
    save(filename::AbstractString, mp::ONNX.ModelProto)
    save(io::IO, mp::ONNX.ModelProto)

Serialize the given [`ONNX.ModelProto`](@ref) to a file with path `filename` or to `io`.
"""
save(filename::AbstractString, mp::ONNX.ModelProto) = open(io -> save(io, mp), filename, "w")
save(io::IO, mp::ONNX.ModelProto) = ONNX.writeproto(io, mp)


"""
    modelproto(f; namestrat=name_runningnr(), posthook=validate, kwargs...)
    modelproto(f, inshapes::Tuple...; namestrat = name_runningnr(), posthook=validate, kwargs...)
    modelproto(f, indata::Pair{String, <:Any}...; modelname="model", namestrat = name_runningnr(), posthook=validate, kwargs...)

Return an [`ONNX.ModelProto`](@ref) from `f`.

Argument `inshapes` are size tuples representing the shape of each input. An attempt to infer sizes will be made if not provided.
Argument `indata` are pairs mapping names to size tuples. Names will be created automatically if not provided.

Argument `modelname` is a string which will be used as the name of the model. Must be non-empty to be valid ONNX.

Argument `namestrat` determines how nodes in the graph shall be named. Other keyword arguments are passed to the `ModelProto`.

Argument `posthook` will be called with the created `ONNX.ModelProto` as argument before returning it.

Other keyword arguments will be passed to `ONNX.ModelProto`.
"""
modelproto(f; kwargs...) = modelproto(f, infer_inshapes(f)...; kwargs...)
modelproto(f, inshapes::Union{Tuple, Missing}...;kwargs...) = modelproto(f, ("data_" .* string.(0:length(inshapes)-1) .=> inshapes)...; kwargs...)
function modelproto(f, indata::Pair{String, <:Any}...; modelname="model", namestrat = default_namestrat(f), posthook=validate, kwargs...)
    mp = modelproto(;
    graph = graphproto(f, indata...; namestrat=namestrat, name=modelname),
    kwargs...)
    posthook(mp)
    return mp
end

"""
    modelproto(g::CompGraph; outshape = shape, namestrat=default_namestrat(g); , posthook=validate, kwargs...)

Return an [`ONNX.ModelProto`](@ref) from `g`.

Argument `outshape` is a function which returns a size tuple representing the shape of the output of a given `AbstractVertex`.

Argument `namestrat` determines how nodes in the graph shall be named. Other keyword arguments are passed to the `ModelProto`.

Argument `posthook` will be called with the created `ONNX.ModelProto` as argument before returning it.

Other keyword arguments will be passed to `ONNX.ModelProto`.
"""
function modelproto(g::CompGraph; modelname="model", outshape = shape, namestrat=default_namestrat(g), posthook=validate, kwargs...)
    mp = modelproto(;
    graph = graphproto(g; outshape, namestrat, name=modelname),
    kwargs...)
    posthook(mp)
    return mp
end

function infer_inshapes(c::Chain)
    sh = infer_inshapes(first(c))
    sh isa FluxLayer && length(c) > 1 && return infer_inshapes(Chain(Base.tail(c.layers)...))
    return sh
end
infer_inshapes(sc::SkipConnection) = infer_inshapes(sc.layers)
infer_inshapes(l) = infer_inshapes(layertype(l), l)
infer_inshapes(lt::FluxTransparentLayer, ::Any) = lt 
infer_inshapes(lt::FluxParLayer, l) = tuple(shape(lt, nin(l)...))

function infer_inshapes(::Any, f)
    ml = methods(f);
    for m in ml.ms
        m.sig isa DataType && return Tuple(infer_shape.(m.sig.types[2:end]))
    end
    return ntuple(i -> missing, ml.mt.max_args)
end
infer_shape(::Type{<:Any}) = missing
infer_shape(::Type{<:AbstractArray{T,N}}) where {T,N} = ntuple(i -> missing, N)

modelproto(;kwargs...) = ONNX.ModelProto(;
    ir_version=6,
    opset_import=[ONNX.OperatorSetIdProto(version=11)],
    producer_name="ONNXNaiveNASflux.jl",
    producer_version=string(Pkg.Types.Context().env.project.version), # TODO: Ugh....
    kwargs...)

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

Base.Broadcast.broadcastable(p::WrappedProbe) = Ref(p)

NaiveNASlib.name(p::WrappedProbe) = NaiveNASlib.name(unwrap(p))

nextname(p::WrappedProbe) = nextname(unwrap(p))
add!(p::WrappedProbe, args...) = add!(unwrap(p), args...)
shape(p::WrappedProbe) = shape(unwrap(p))
add_output!(p::WrappedProbe) = add_output!(unwrap(p))
newnamestrat(p::WrappedProbe, args...) = rewrap(p, newnamestrat(unwrap(p), args...))
newfrom(p::WrappedProbe, args...) = rewrap(p, newfrom(unwrap(p), args...))

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

## Don't forget to check if new methods need to be added for any WrappedProbe implementations if you add something here!

# Stuff whose only purpose is to override the name in case this is the naming strategy
# See namingutil for a little story about the design since it is a bit messy :(

"""
    NameInterceptProbe <: WrappedProbe

An AbstractProbe which is only used to intercept methods call above the primitive level to catch names (e.g. a `Chain` with a `NamedTuple` as layers).
"""
struct NameInterceptProbe{P<:AbstractProbe} <: WrappedProbe
    wrapped::P
end
rewrap(::NameInterceptProbe, p) = NameInterceptProbe(p)

inputprotoprobe!(gp, name, shape, namestrat::NamedNodeContext) = NameInterceptProbe(_inputprotoprobe!(gp, name, shape, namestrat))

# This little dance is just to avoid ambiguities of (n::NamedNode)(pps...) since NamedNode is abstract
# TODO: Generate with macro so we can add types to the union without worry? 
(v::MutationVertex)(pps::AbstractProbe...) = _apply_probe_call(v, pps...)
(n::NamedFunction)(pps::AbstractProbe...) = _apply_probe_call(n, pps...)

_apply_probe_call(n::NamedNode, pps::AbstractProbe...) = __apply_probe_call(n, nextname(first(pps)), pps...)

# We are not in the business of catching names here, so just forward the call
__apply_probe_call(n::NamedNode, ::Any, pps::AbstractProbe...) = base(n)(pps...)
# We just want to rewrap probes in NameInterceptProbes for the sole reason that we might encounter other 
# objects that we want to intercept names from (e.g. a Chain inside a Parallel)
__apply_probe_call(n::NamedNode, ::NamedNodeContext, pps::AbstractProbe...) = _apply_probe_call(n, map(NameInterceptProbe, pps)...)

function _apply_probe_call(n::NamedNode, pps::NameInterceptProbe...)
    ppsname = map(pps) do pp
        newnamestrat(pp, nextname(pp)(n))
    end
    ppout = base(n)(ppsname...)
    return newnamestrat(ppout, nextname(pps[1]))
end

# Here we wrap all callable children in a NamedFunction so that we can access the name when the child is called
(c::Chain)(pp::NameInterceptProbe) = _instrument_named_functor(c; addbasename=false)(unwrap(pp))
(p::Parallel)(pp::NameInterceptProbe) = _instrument_named_functor(p; addbasename=false)(unwrap(pp))
(p::SkipConnection)(pp::NameInterceptProbe) = _instrument_named_functor(p; addbasename=false)(unwrap(pp))

# The exclude is to ensure that we only see the fields of w, not the fields of its children
_instrument_named_functor(w; addbasename=true) = Functors.fmap_with_path(w; exclude = (k,c) -> c != w) do keypath, child
    _instrument_named_child(child, string(only(keypath)); addbasename)
end

_instrument_named_child(child, name; kwargs...) = !isempty(methods(child)) ? NamedFunction(child, name) : child
_instrument_named_child(child::Tuple, name; addbasename) = ntuple(length(child)) do i
    _instrument_named_child(child[i], string(addbasename ? name : "", '[', i, ']'))
end
_instrument_named_child(child::AbstractArray, name; addbasename) = map(enumerate(child)) do (i, elem)
    _instrument_named_child(elem, string(addbasename ? name : "", '[', i, ']'))
end
_instrument_named_child(child::NamedTuple{K}, name; addbasename) where K = ntuple(length(child)) do i
    _instrument_named_child(child[i], string(addbasename ? string(name, '.') : "", K[i]))
end |> NamedTuple{K}

# End of stuff whose only purpose is to override the name in case this is the naming strategy 


"""
    Used to get activation functions as [`ONNX.AttributeProto`](@ref)s.
"""
struct ActivationAttributeProbe end

"""
    graphproto(;kwargs...)

Return an [`ONNX.GraphProto`](@ref) with all fields initialized to empty arrays.

`kwargs` are just passed on to [`ONNX.GraphProto`](@ref), potentially overwriting the empty arrays.
"""
_graphproto(;kwargs...) = ONNX.GraphProto(;
node = ONNX.NodeProto[],
initializer = ONNX.TensorProto[],
input = ONNX.ValueInfoProto[],
output = ONNX.ValueInfoProto[],
value_info = ONNX.ValueInfoProto[],
kwargs...
)

"""
    graphproto(g::CompGraph; outshapefun = shape, namestrat=default_namestrat(g); kwargs...)

Return an [`ONNX.GraphProto`](@ref) from `g`.

Argument `outshape` is a function which returns the shape of an `AbstractVertex`.

Argument `namestrat` determines how nodes shall be named.

All other keyword arguments are passed on to `ONNX.GraphProto`.
"""
graphproto(g::CompGraph; outshape = shape, namestrat=default_namestrat(g), kwargs...) = _graphproto(g, (recursename.(inputs(g), namestrat) .=> outshape.(inputs(g)))...;namestrat=namestrat, kwargs...)

"""
    graphproto(f, indata::Pair{String, <:Any}...; namestrat = name_runningnr(), kwargs...)

Return an [`ONNX.GraphProto`](@ref) from `g`.

Argument indata are name => shape pairs for the input data.

Argument `namestrat` determines how nodes shall be named.

All other keyword arguments are passed on to `ONNX.GraphProto`.
"""
graphproto(args...; kwargs...) = _graphproto(args...; kwargs...)

function _graphproto(f, indata::Pair{String, <:Any}...; namestrat = name_runningnr(), kwargs...)
    gp = _graphproto(;kwargs...)
    pps = map(indata) do (name, shape)
        inputprotoprobe!(gp, name, shape, namestrat)
    end

    outpps = f(pps...)

    add_outputs!(gp, namestrat, outpps)

    return gp
end
add_outputs!(gp, ns, x) = add_outputs!(gp, ns, (x,))
add_outputs!(gp, ns, pps::NTuple{N, AbstractProbe}) where N  = add_output!.(pps)
function add_outputs!(gp, namestrat, pps::Tuple)
    # At least one of the outputs was not an AbstractProbe
    # This is probably because one of them is a constant
    # If there is at least one AbstractProbe we here assume that one contains the GraphProto for the non-constant ops
    anyprobeind = findfirst(x -> isa(x, AbstractProbe), pps)
    tempprobe = isnothing(anyprobeind) ? ProtoProbe("template", tuple(), namestrat, gp) : pps[anyprobeind]

    output_pps = constant.(pps, tempprobe, namestrat)
    add_output!.(output_pps)
end


actfun(::FluxLayer, l) = l.σ
function weightlayer(lt::FluxParLayer, l, pp, optype;attributes = ONNX.AttributeProto[])
    lname = recursename(l, nextname(pp))
    wname, bname = lname .* ("_weight", "_bias")

    add!(pp, ONNX.TensorProto(flipweights(lt, weights(l)), wname))
    inputnames = addbias!(lt, pp, bias(l), bname, [name(pp), wname])

    add!(pp, ONNX.NodeProto(
        input=inputnames,
        output=[lname],
        name=lname,
        attribute = attributes,
        op_type=optype))

    ppl = newfrom(pp, lname, s -> outshape(l, s))
    ppout = actfun(lt, l)(newnamestrat(ppl, f -> join([lname, genname(f)], "_"), lname))
    return newnamestrat(ppout, nextname(ppl))
end

function addbias!(lt, pp, b, name, inputnames)
    add!(pp, ONNX.TensorProto(b, name))
    return vcat(inputnames, name)
end
function addbias!(lt, pp, bparam::Number, name, inputnames) 
    @assert bparam == false "ONNX model with bias term $bparam not supported!"
    return inputnames
end

function(l::Flux.Dense)(pp::AbstractProbe)
    ppl = pp
    if !ismissing(shape(pp)) && ndims(pp) == 3
        # Special case: Recurrent -> Dense. This is nothing special in flux as the dense layers automatically broadcast
        # through all dimensions except the first.
        # For it to be valid ONNX however we need to add a reshape so that time dimension becomes batch dimension
        outsize = shape(pp)[1]
        lname = recursename(l, nextname(pp))
        ppn = newnamestrat(pp, s -> lname * genname(s))
        ppn = reshape(ppn, nin(l)[], :)
        ppl = newnamestrat(ppn, s -> lname)
    end
    ppout = weightlayer(layertype(l), l, ppl, "Gemm")
    return newnamestrat(ppout, nextname(ppl))
end

(l::Flux.Conv)(pp::AbstractProbe) = weightlayer(layertype(l), l, pp, "Conv"; attributes = attribs(l))
(l::Flux.ConvTranspose)(pp::AbstractProbe) = weightlayer(layertype(l), l, pp, "ConvTranspose"; attributes = attribs(l))

attribs(l) = attribs(layertype(l), l)
attribs(::FluxConvolutional{N}, l) where N = ONNX.AttributeProto.([ "pads", "strides", "dilations"], [padexpand(Val(N), l.pad), reverse(l.stride), reverse(l.dilation)])
attribs(l::Union{MaxPool{N}, MeanPool{N}}) where N = ONNX.AttributeProto.(["kernel_shape", "pads", "strides"],  [reverse(l.k), padexpand(Val(N), l.pad), reverse(l.stride)])

# Interleave padding! (1,2) => [2,1,2,1], (1,1,2,2,3,3) => (3,2,1,3,2,1)
padexpand(::Val{N}, x::NTuple{N}) where N =  repeat(reverse(collect(x)), 2)
padexpand(::Val{N}, x::NTuple{M}) where {N,M} = vcat(collect(x[end-1:-2:1]), collect(x[end:-2:2]))

function(l::Flux.BatchNorm)(pp::AbstractProbe)
    lname = recursename(l, nextname(pp))
    γname, βname, μname, σ²name = lname .* ("_scale", "_bias", "_mean", "_var")

    add!(pp, ONNX.NodeProto(
        input=[name(pp), γname, βname, μname, σ²name],
        output=[lname],
        name=lname,
        attribute = ONNX.AttributeProto.(["epsilon", "momentum"], [l.ϵ, l.momentum]),
        op_type="BatchNormalization"))
    add!(pp, ONNX.TensorProto(l.γ, γname))
    add!(pp, ONNX.TensorProto(l.β, βname)) # Bias not optional for batchnorm
    add!(pp, ONNX.TensorProto(l.μ, μname))
    add!(pp, ONNX.TensorProto(l.σ², σ²name))

    ppout = actfun(layertype(l), l)(newnamestrat(pp, f -> join([lname, genname(f)], "_"), lname))
    return newnamestrat(ppout, nextname(pp))
end
actfun(::FluxBatchNorm, l) = l.λ

function(l::Flux.InstanceNorm)(pp::AbstractProbe)
    @assert l.affine == true "ONNX InstanceNormalization does not support affine=false"
    @assert l.track_stats == false "ONNX InstanceNormalization does not support track_stats=true"
    lname = recursename(l, nextname(pp))
    γname, βname = lname .* ("_scale", "_bias")

    add!(pp, ONNX.NodeProto(
        input=[name(pp), γname, βname],
        output=[lname],
        name=lname,
        attribute = ONNX.AttributeProto.(["epsilon"], [l.ϵ]),
        op_type="InstanceNormalization"))

    add!(pp, ONNX.TensorProto(l.γ, γname))
    add!(pp, ONNX.TensorProto(l.β, βname))


    ppout = actfun(layertype(l), l)(newnamestrat(pp, f -> join([lname, genname(f)], "_"), lname))
    return newnamestrat(ppout, nextname(pp))
end
actfun(::FluxInstanceNorm, l) = l.λ


# Dropdims because ONNX expects recurrent layers to output tensors of shape [seq_length, num_directions, batch_size, hidden_size] where num_directions is 2 in case of bidirectional and 1 otherwise
# Flux.Recur is not bidirectional so we'll just assume the user wants to also drop num_directions so that recurrent layers can be stacked without hassle.
# Override Flux.Recur with some other method to circumvent this behaviour if not wanted
(m::Flux.RNN)(pps::AbstractProbe...) = dropdims(m.cell(pps...); dims=3)
(m::Flux.LSTM)(pps::AbstractProbe...) = dropdims.(m.cell(pps...); dims=3)

(l::Flux.RNNCell)(pp::AbstractProbe) = recurrent_node(l, pp, "RNN")
(l::Flux.LSTMCell)(pp::AbstractProbe) = recurrent_node(l, pp, "LSTM")

function recurrent_node(l, pp, optype)
    lname = recursename(l, nextname(pp))
    wname, rname, bname = lname .* ("_W", "_R", "_B")

    hsize = size(l.Wh, 2)
    hsattrib = ONNX.AttributeProto("hidden_size", hsize)

    inputnames = [name(pp), wname, rname]

    # Flux weights are of shape [hidden_size, input_size]
    # ONNX wants them on the form [num_directions, hidden_size, input_size] (where num_directions is 2 for bidirectional else 1)
    # To spice things up a bit, all julia arrays are saved in reverse order, i.e we need to create a TensorProto from an array with the arrangement [input_size, hidden_size, num_directions].
    # First transpose the weights into [input_size, hidden_size], then reshape by adding 1 extra dimension
    Wi = permutedims(flipweights(layertype(l), l.Wi, hsize))
    add!(pp, ONNX.TensorProto(reshape(Wi, size(Wi)...,1), wname))
    Wh = permutedims(flipweights(layertype(l), l.Wh, hsize))
    add!(pp, ONNX.TensorProto(reshape(Wh, size(Wh)..., 1), rname))

    if !isa(bias(l), Number)
        # ONNX has a separate bias for the recurrent part and wants the concatenation of input and recurrent biases.
        # We'll just hard code it to zeros. Doesn't matter which part is which as they are just added together in the ONNX expression for RNNs.
        b = flipweights(layertype(l), reshape(bias(l), :, 1), hsize)
        add!(pp, ONNX.TensorProto(vcat(b, zeros(eltype(b), size(b))), bname))
        push!(inputnames, bname)
    end

    add!(pp, ONNX.NodeProto(
        input=inputnames,
        output=[lname],
        name=lname,
        attribute = push!(activation_attrib(l), hsattrib),
        op_type=optype))
    # ONNX recurrent layers have multiple outputs, but Flux is generally only compatible
    # with the first output (the hidden state for all timesteps).
    # Even though Flux.LSTM also outputs the cell state, it does so for all timesteps
    # while ONNX only does so for the last timestep.
    # I guess that in theory we could support models which immediately remove all but
    # the last timestamp from the cell state, but it seems like it would be hell to
    # resolve this when importing so I'm not gonna bother right now.
    return newfrom(pp, lname, s -> outshape(l, s))
end

activation_attrib(l) = l.σ(ActivationAttributeProbe())
activation_attrib(::Flux.LSTMCell) = ONNX.AttributeProto[] #Only default values supported by Flux

Base.tanh(::ActivationAttributeProbe) = rnnactattribs("Tanh")
Flux.elu(::ActivationAttributeProbe, α=1f0) = rnnactattribs("Elu", α)

rnnactattribs(op::AbstractString, α=0f0, β=0f0) = rnnactattribs([op], [α], [β])
rnnactattribs(ops::AbstractVector, αs, βs) = ONNX.AttributeProto.(["activations", "activation_alpha", "activation_beta"], [ops, αs, βs])

function attribfun(fhshape, optype, pps::AbstractProbe...; attributes = ONNX.AttributeProto[], lname = recursename(lowercase(optype), nextname(pps[1])))
    add!(pps[1], ONNX.NodeProto(
    input = collect(name.(pps)),
    output = [lname],
    name=lname,
    attribute = attributes,
    op_type= optype))
    return newfrom(pps[1], lname, fhshape)
end

Flux.relu(pp::AbstractProbe) = attribfun(identity, "Relu", pp)
Flux.leakyrelu(pp::AbstractProbe, α=0.01f0) = attribfun(identity, "LeakyRelu", pp; attributes = [ONNX.AttributeProto("alpha", α)])
Flux.elu(pp::AbstractProbe, α=1f0) = attribfun(identity, "Elu", pp; attributes = [ONNX.AttributeProto("alpha", α)])
Flux.selu(pp::AbstractProbe) = attribfun(identity, "Selu", pp)
Flux.selu(pp::AbstractProbe, γ, α) = attribfun(identity, "Selu", pp; attributes = ONNX.AttributeProto.(["gamma", "alpha"], [γ, α]))
Flux.σ(pp::AbstractProbe) = attribfun(identity, "Sigmoid", pp)
Flux.sigmoid_fast(pp::AbstractProbe) = attribfun(identity, "Sigmoid", pp)   # Flux-specific construct

Base.tanh(pp::AbstractProbe) = attribfun(identity, "Tanh", pp) 
Flux.softmax(pp::AbstractProbe; dims=1) =  onnxsoftmax(pp, np_axis = flux2numpydim(dims[end], ndims(pp)))
onnxsoftmax(pp::AbstractProbe; np_axis=1) =  attribfun(identity, "Softmax", pp; attributes=[ONNX.AttributeProto("axis", np_axis)])

(l::Flux.MaxPool)(pp::AbstractProbe) = attribfun(s -> outshape(l, s), "MaxPool", pp; attributes = attribs(l))
(l::Flux.MeanPool)(pp::AbstractProbe) = attribfun(s -> outshape(l, s), "AveragePool", pp; attributes = attribs(l))
(l::Flux.Dropout)(pp::AbstractProbe) = attribfun(identity, "Dropout", pp; attributes = [ONNX.AttributeProto("ratio", l.p)])

(l::Flux.GlobalMaxPool)(pp::AbstractProbe) = globalmaxpool(pp, identity)
(l::Flux.GlobalMeanPool)(pp::AbstractProbe) = globalmeanpool(pp, identity)

globalmeanpool(pp::AbstractProbe, wrap) = globalpool(pp, wrap, "GlobalAveragePool")
globalmaxpool(pp::AbstractProbe, wrap) = globalpool(pp, wrap, "GlobalMaxPool")

function globalpool(pp::AbstractProbe, wrap, type)
     gpp = attribfun(s -> ismissing(s) ? s : (1, 1, s[3:end]...), type, pp)
     ppnext = newnamestrat(gpp, f -> join([name(gpp), genname(f)], "_"), name(gpp))
     wpp = wrap(ppnext)
     return newnamestrat(wpp, nextname(gpp))
end

# Generate explicit combinations as I couldn't figure out how to avoid type piracy with varargs: https://discourse.julialang.org/t/extend-a-varargs-function-for-mixed-types/38233
function generate_elemwise(fm::Module, f, optype, argperms, m=@__MODULE__)
    for argtypes in argperms
        args = ntuple(i -> Symbol(:x, i), length(argtypes))
        sig = map(zip(args, argtypes)) do (a, at)
            isnothing(at) && return a
            :($a::$at)
        end

        @eval m $fm.$f($(sig...)) = elemwisefun($optype, $(args...))
    end
end

"""
    override_broadcast(f::F, argperms) where F

Prevent broadcasting of `f` when invoked with any combination of argument types in argperms.

Needed because broadcasting happens inside several ONNX operations.

For example, `[1,2,3] .+ 4` shall translate to `Add([1,2,3], 4)`, not as `Add(1, 4)`, `Add(2, 4)` and `Add(3, 4)`. One way to accomplish this is to override broadcasting when an `AbstractProbe` is one of the inputs.
"""
function override_broadcast(f::F, argperms, m=@__MODULE__) where F
     for argtypes in argperms

        argnames = ntuple(i -> Symbol(:x, i), length(argtypes))
        sig = map(zip(argnames, argtypes)) do (a, at)
            isnothing(at) && return a
            :($a::$at)
        end

        @eval m begin
            Base.Broadcast.broadcasted(f::$F, $(sig...)) = f($(argnames...))      
        end
    end
end

dummyfun(x,y) = "dummy $x"

argpermutations(n, args...) = Iterators.product(ntuple(_ -> args, n)...)
argpermswith(t, n::Integer, args...) = (a for a in argpermutations(n, t, args...) if t in a)

function gen_broadcastable_elemwise(f, optype, n=2)
    fs = Symbol(f)
    fm = which(ONNXNaiveNASflux, fs)
    generate_elemwise(fm, fs, optype, argpermswith(AbstractProbe, n, nothing))
    override_broadcast(f, argpermswith(AbstractProbe, n, AbstractArray))
end

gen_broadcastable_elemwise(+, "Add")
gen_broadcastable_elemwise(*, "Mul")
gen_broadcastable_elemwise(/, "Div")

function elemwisefun(optype, args...)
    # This mess is only to make sure we first draw the name of the op so that any constants base their name of it
    anyprobe = args[findfirst(x -> isa(x, AbstractProbe), args)]
    oname = recursename(lowercase(optype), nextname(anyprobe))
    nf = name_runningnr()
    refprobe = newnamestrat(anyprobe, f -> join([oname, nf(f)], "_"))
    return attribfun(identity, optype, constant.(args, refprobe, nextname(anyprobe))...; lname=oname)
end

constant(x::AbstractProbe, ::AbstractProbe, ns) = x
function constant(x, pp::AbstractProbe, ns)
    cname = recursename("constant", nextname(pp))
    add!(pp, ONNX.NodeProto(
    input = [],
    output = [cname],
    name=cname,
    attribute = ONNX.AttributeProto.(["value"], [ONNX.TensorProto(x, cname * "_value")]),
    op_type= "Constant"))
    ppo = newfrom(pp, cname, identity)
    return newnamestrat(ppo, ns)
end


function axisfun(fshape, optype, pps::AbstractProbe...; dims, axname="axes")
    attrib = if isempty(dims)
        ONNX.AttributeProto[]
    else
        pok = filter(p -> !ismissing(shape(p)), pps)
        @assert !isempty(pok) "Must have at least one shape to determine axis!"
        np_axis = flux2numpydim.(dims, ndims(pok[1]))
        [ONNX.AttributeProto(axname, np_axis)]
    end
    axisfun(fshape, optype, attrib, pps...)
end

function axisfun(fshape, optype, attrib::AbstractArray{<:ONNX.AttributeProto}, pps::AbstractProbe...)   
    fname = recursename(lowercase(optype), nextname(pps[1]))

    add!(pps[1], ONNX.NodeProto(
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

Base.cat(pps::AbstractProbe...; dims) = axisfun("Concat", pps...; dims=dims, axname="axis") do s
    sumshape = aggshape.(+, vcat(shape.(pps)...)...)
    return ntuple(i -> i in dims ? sumshape[i] : s[i], length(s))
end
Statistics.mean(pp::AbstractProbe; dims=()) = axisfun("ReduceMean", pp; dims=scal2tup(dims)) do s
    return ntuple(i -> i in dims ? 1 : s[i], length(s))
end
Base.dropdims(pp::AbstractProbe; dims) = axisfun(s -> rmdims(s, dims), "Squeeze", pp; dims=scal2tup(dims))

reshape_keepshape(pp::AbstractProbe, shape) = reshape(pp, shape)
Base.reshape(pp::AbstractProbe, shape...) = reshape(pp, shape)
function Base.reshape(pp::AbstractProbe, shape::Tuple)
    fname = recursename("Reshape", nextname(pp))
    sname = fname .* "_shape"
    fluxshape = collect(Int, map(s -> s == Colon() ? -1 : s, shape))

    add!(pp, ONNX.NodeProto(
        input=[name(pp), sname],
        output=[fname],
        name=fname,
        op_type="Reshape"))
    add!(pp, ONNX.TensorProto(reverse(fluxshape), sname))

    fshape = function(s)
        return map(enumerate(fluxshape)) do (ind, new)
            new == -1 && return missing # CBA to figure out how to do this...
            new == 0 && return s[ind]
            return new
        end |> Tuple
    end

    return newfrom(pp, fname, fshape)
end
expanddims(p::AbstractProbe, x, dims) = p

Flux.flatten(pp::AbstractProbe) = flatten(pp, ndims(pp)-1)

function flatten(pp::AbstractProbe, dim)
    fname = recursename("Flatten", nextname(pp))

    add!(pp, ONNX.NodeProto(
        input=[name(pp)],
        output=[fname],
        name=fname,
        attribute = [ONNX.AttributeProto("axis", -dim)],
        op_type="Flatten"))

    fshape = function (s)
        dim == 0 && return (aggshape(*, s), 1)
        absdim = dim < 0 ? length(s) + dim : dim
        return (aggshape(*, s[1:absdim]...), aggshape(*, s[absdim+1:end]))
    end
    return newfrom(pp, fname, fshape)
end

Flux.unsqueeze(pp::AbstractProbe; dims) = axisfun(s -> insdims(s, dims), "Unsqueeze", pp; dims=scal2tup(dims))
unsqueeze_onnx(pp::AbstractProbe, npa::NumPyAxes) = axisfun(s -> insdims(s, npa), "Unsqueeze", [ONNX.AttributeProto("axes", npa.axes)], pp)

