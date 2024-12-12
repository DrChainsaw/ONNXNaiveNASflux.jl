
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
    set_unused_outputs_to_empty!(gp)

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

function set_unused_outputs_to_empty!(gp::ONNX.GraphProto)
    all_used_inputs = mapreduce(n -> n.input, vcat, gp.node; init=name.(gp.output)) |> Set

    for node in gp.node
        if length(node.output) > 1
            # Special case for nodes with more than one output as Flux layers always output everything
            # and "not used" just means that no other op used it as input 
            for (i, outname) in zip(eachindex(node.output), node.output)
                if outname ∉ all_used_inputs
                    # Empty names to signal positional outputs that are not used 
                    #(e.g. generate output nr 2 but not nr 1)  
                    node.output[i] = ""
                end
            end
        end
    end
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


(m::Flux.RNN)(pps::AbstractProbe...) = m.cell(pps...)
(m::Flux.LSTM)(pps::AbstractProbe...) = m.cell(pps...)

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
        # This happens to work since all currently supported Flux recurrent layers
        # output (close to) the same thing as the first output in their ONNX counterpart
        # We might push more outputs to this if we encounter some sort of "select and
        # reformat the data into this other output" operation down the road. 
        output=[lname],
        name=lname,
        attribute = push!(activation_attrib(l), hsattrib),
        op_type=optype))

    # Big sigh here:
    # ONNX recurrent layers have different number of outputs compared to Flux (e.g. 3 outputs for LSTM vs 2 in Flux)
    # At this point we need to return the exact number of outputs as Flux outputs or else the function will probably fail
    # (e.g the next operation expect a tuple of exactly 2 elements from the LSTM).

    # Therefore, at the time of writing, the fact that we return multiple outputs here is a false friend as it has
    # really nothing to do with being compliant with the ONNX spec. It is strictly just to ensure that the function
    # we are tracing through is still valid.

    # As a matter of fact, only the first output from Flux happens to be somewhat close to what ONNX defines
    # (more on that below).

    # For LSTM, the second output shall be the hidden state (same as first output) for only the last time step
    # while Flux outputs the cell state for all time steps there while the third output in ONNX is the cell
    # state but only for the last time step. 
    
    # The saga continues:
    # To prevent that this causes confusion when tryng to import the model in some other framework, we 
    # wrap the second output of Flux LSTM in a toxic AbstractProbe which will throw an exception stating
    # that this output is not ONNX compatible if anything touches it.  

    # But wait! There is more!, 
    # ONNX states that recurrent layers have 4D output while Flux has 3D output (the extra dimension being)
    # the two directions if bidirectional.
    # If we are just saving a native Flux model we handle this by adding a Squeeze op directly after this
    # layer which will remove the extra dimension (which is a singleton since Flux does not do bidirectional).

    # Now for the grand finale:
    # If we are saving a model which was imported from ONNX using ONNXNaiveNASflux there will be a future
    # op which changes the Flux output to ONNX output, both w.r.t the extra dimension for the directions
    # (we don't support bidirectional so it is always just a matter of adding a singleton dimension),
    # adding one extra output (i.e making the LSTM 2 element tuple into 3 element tuple) and shaving
    # off the extra time steps. 

    # Since we can't tell from there whether this will happen we need to prepare a hypothetical output
    # which can then be selected from the coming OP. This hypothetical output will clean itself up
    # if it does not encounter the Flux->ONNX specific OPs of this package. Note that it will have to
    # propagate through the Squeeze OP and then maybe go back and remove the Squeze as well as 
    # remove the toxic AbstractProbe
    return _returnvalue(l, pp, lname)
end

_outputnames(l, basename) = _outputnames(layertype(l), l, basename)
_outputnames(::Any, l, basename) = [basename]
_outputnames(::FluxLstmCell, l, basename) = [basename, string(basename, "_cell")]

_returnvalue(l, pp, lname) = _returnvalue(layertype(l), l, pp, lname)
# Dropdims because ONNX expects recurrent layers to output tensors of shape [seq_length, num_directions, batch_size, hidden_size] where num_directions is 2 in case of bidirectional and 1 otherwise
# Flux recurrent layers are not bidirectional so we'll just assume the user wants to also drop num_directions so that recurrent layers can be stacked without hassle.
function _returnvalue(::FluxRecurrentCell, l, pp, lname) 
    pnew = dropdims(newfrom(pp, lname, s -> outshape(l, s)); dims=3)
    OutputSelectProbe(pnew, lname, falses(2), 0)
end
# TODO Warning!! Flux does not comply to ONNX for LSTM. The first output is the same, but the second output in 
# ONNX is the last time step of the hidden and the third output in ONNX is the last time step of the cell state
function _returnvalue(::FluxLstmCell, l, pp, lname) 
     out1 = dropdims(newfrom(pp, lname, s -> outshape(l, s)); dims=3)
     out2 = IncompatibleProbe(newfrom(pp, lname, s -> outshape(l, s)))
     outputused = falses(3)
     (OutputSelectProbe(out1, lname, outputused, 0),
      OutputSelectProbe(out2, lname, outputused, 0))
end

# We are either trying to save a model which was imported, or someone has just used this to create an ONNX
# compatible model. We need to fake another output so that this becomes the second output of the RNN
# Note that this pretty much assumes full on that this is the next op after the RNN
_onnx_rnn_output1(p::AbstractProbe) = select_output!(p, 1, "")
function _onnx_rnn_output2(p::AbstractProbe) 
    pnew = select_output!(p, 2, "_hidden", s -> s[1:3])
    ndims(p) == 4 ? pnew : dropdims(pnew; dims=3)
end


_onnx_lstm_output1((h, c)::NTuple{2, AbstractProbe}) = select_output!(h, 1, "")
function _onnx_lstm_output2((h, c)::NTuple{2, AbstractProbe}) 
    psel = select_output!(h, 2, "_hidden", s -> s[1:3])
    ndims(h) == 4 ? psel : dropdims(psel; dims=3)
end
function _onnx_lstm_output3((h, c)::NTuple{2, AbstractProbe}) 
     psel = unwrap(select_output!(c, 3, "_cell", s -> s[1:3]), IncompatibleProbe)
     ndims(h) == 4 ? psel : dropdims(psel; dims=3)
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

