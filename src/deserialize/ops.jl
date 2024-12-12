const sources = Dict{Symbol, Any}()
const actfuns = Dict{Symbol, Any}()
const rnnactfuns = Dict{Symbol, Any}() # Recurrent layers have activation functions as attributes and use different parameter names compared to their respective operations.
const actlayers = Dict{Symbol, Any}()
const fluxlayers = Dict{Symbol, Any}()
const fluxrecurrentlayers = Dict{Symbol, Any}()
const invariantops = Dict{Symbol, Any}()
const pseudotransparentops = Dict{Symbol, Any}()
const verts = Dict{Symbol, Any}()
const fluxlayertypes = Dict{Symbol, Any}()

layerfuns = Dict{Symbol, Any}()

# Rundown of the basic idea here:

# Aspect 1
# ONNX does not have activation functions as an attribute to its layers but rather represents them as a separate node
# This would indeed be workable, but...
# 1. It is a bit annoying that model -> serialize -> deserialize does not result in the exact same thing
# 2. If one wants to use the mutation functionality of NaiveNASflux it might not be desirable to have activation
#    functions as separate vertices in the graph as this invites for things like inserting something else between
#    the layer and its activation function.

# To be able to have activation functions back inside their layers when deserializing, whenever an op which is a key
# in actlayers is encountered there is a "lookahead" to see if the op of the next node is in actfuns. If it is, the
# two ops will be merged into one vertex containing the layer and its activation function.
# A very similar thing is done for global pooling operations followed by squeeze or reshape.

# Aspect 2
# The vertices of NaiveNASflux require a few inputs when creating them. One in particular is knowledge of the size
# trait which is obviously not possible to obtain from the ONNX data. In order to spare users from having to supply
#  this extra input with each operation there is one dict per "general type".

# As NaiveNASflux already has the knowledge what is needed for all layers in Flux, they have their own dict
#  (fluxlayers) which just outsources the vertex creation to NaiveNASflux. Note that all actlayers are inserted
# in this dict.

# Functions which always produce the same number of outputs as inputs and are not defined in Flux, e.g.
#  GlobalAveragePool end up in invariantops.

# Functions which have dedicated vertex construction methods, such as Concat and Add end up in verts.


"""
    OutputSelection(selection, wrapped)

Selects outputs from `wrapped` using `selection`.

Typically used when `wrapped` outputs a `Tuple` from which other nodes in the computation graph
only wants a subset.

Can also be used to transform Flux output to ONNX output. One example is recurrent layers where
Flux outputs all time steps of the hidden state while some ONNX outputs are only the last step.

Note that the more useful and generic InputSelection (which would allow a node to pick a subset)
of some other nodes output as its input is not yet implemented. OutputSelection only works when
1) all nodes which take input from `wrapped` want the exact same outputs and 2) on output nodes
(which is the reason why I bothered to implement it to begin with).
"""
struct OutputSelection{FS, L} <: NaiveNASflux.AbstractMutableComp
    selection::FS
    wrapped::L
end
NaiveNASflux.wrapped(o::OutputSelection) = o.wrapped
(o::OutputSelection)(x...) = _apply_selection(o.selection, wrapped(o)(x...))

_apply_selection(fs::Tuple, x) = map(f -> f(x), fs)
_apply_selection(f, x) = f(x)

# Use for Recurrent layers since ONNX specifies on extra dimension for the number of directions
# which Flux does not have
struct AddSingletonDim{L} <: NaiveNASflux.AbstractMutableComp
    dim::Int
    wrapped::L
end
NaiveNASflux.wrapped(a::AddSingletonDim) = a.wrapped
function (a::AddSingletonDim)(x) 
    y = wrapped(a)(x)
    _apply_add_singleton_dim(y, a.dim) 
end

_apply_add_singleton_dim(x, dim) = reshape(x, size(x)[1:dim-1]..., 1, size(x)[dim:end]...)
_apply_add_singleton_dim(xt::Tuple, dim) = map(x -> _apply_add_singleton_dim(x, dim), xt)

struct OpNotSupportedError <: Exception
    msg::String
end
OpNotSupportedError(op_type::Symbol) = OpNotSupportedError(string("Operation type ", op_type, " not supported!"))
Base.showerror(io::IO, e::OpNotSupportedError) = print(io, "OpNotSupportedError: ", e.msg)

sources[:Constant] = function(params) 
    params = if ACTIVE_OUTPUTS_ATTRIBUTE_KEY in keys(params)
        delete!(copy(params), ACTIVE_OUTPUTS_ATTRIBUTE_KEY)
    end
    constant(Val.(keys(params))..., values(params)...)
end
constant(::Val{:value}, val::ONNX.TensorProto) = val |> array
constant(::Val{:value}, val) = val

actfuns[:Relu] = params -> Flux.relu
actfuns[:Sigmoid] = params -> Flux.σ

actfuns[:LeakyRelu] = function(params)
    α = get(params, :alpha, 0.01f0)
    return x -> Flux.leakyrelu(x, oftype(x, α))
end
rnnactfuns[:LeakyRelu] = (ind, params) -> actfuns[:LeakyRelu](Dict(:alpha => get(params, :activation_alpha, ntuple(i -> 0.01f0, ind))[ind]))

actfuns[:Elu] = function(params)
    α = get(params, :alpha, 1)
    return x -> Flux.elu(x, oftype(x, α))
end
rnnactfuns[:Elu] = (ind, params) -> actfuns[:Elu](Dict(:alpha => get(params, :activation_alpha, ntuple(i -> 1, ind))[ind]))

actfuns[:Selu] = function(params)
    haskey(params, :alpha) || haskey(params, :gamma) && return Flux.selu
    γ = get(params, :gamma, Float32(1.05070102214813232421875))
    α = get(params, :alpha, Float32(1.67326319217681884765625))
    return x -> selu(x, oftype(x, γ), oftype(x, α))
end
Flux.selu(x, γ, α) = γ * ifelse(x > 0, x/1, α * (exp(x) - 1))

actfuns[:Tanh] = params -> tanh
rnnactfuns[:Tanh] = (ind, params) -> tanh


mrev(x) = x
mrev(x::AbstractVector) = reverse(x)
prev(x) = x
prev(x::AbstractVector) = reshape(permutedims(reverse(reshape(x, length(x) ÷ 2,:);dims=1)),:)


# mrev = maybe reverse. prev = rearrange padding, e.g. (1,2,1,2) => (2,2,1,1) or (1,2,3,1,2,3) => (3,3,2,2,1,1)
_akpsd(params) = get(params, :activation, identity), mrev(get(params, :kernel_shape, 1)), prev(get(params, :pads, 0)), mrev(get(params, :strides, 1)), mrev(get(params, :dilations, 1))
akpsd(params) = a2t.(_akpsd(params))
a2t(x) = x
a2t(a::AbstractArray) = Tuple(a)

actlayers[:Conv] = function(params, weight::AbstractArray{T, N}, bias=false) where {T, N}
    a,_,p,s,d = akpsd(params)
    @assert get(params, :group, 1) == 1 "Group size not supported!" # TODO
    return Conv(flipweights(FluxConv{N-2}(), weight), bias, a, pad=p, stride=s, dilation=d)
end
fluxlayertypes[:Conv] = (weight, bias=nothing) -> FluxConv{length(size(weight))-2}()

actlayers[:ConvTranspose] = function(params, weight::AbstractArray{T, N}, bias=false) where {T, N}
    a,_,p,s,d = akpsd(params)

    @assert get(params, :group, 1) == 1 "Group size not supported!" # TODO
    @assert !haskey(params, :output_shape) "ConvTranspose: output_shape not supported"
    @assert !haskey(params, :output_padding) "ConvTranspose: output_padding not supported"

    return ConvTranspose(flipweights(FluxConvTranspose{N-2}(), weight), bias, a, pad=p, stride=s, dilation=d)
end
fluxlayertypes[:ConvTranspose] = (weight, bias=nothing) -> FluxConvTranspose{length(size(weight))-2}()

biasarray(b::Bool, esize, β) = b
biasarray(b::AbstractArray, esize, β) = length(b) === 1 ? repeat(β .* vec(b), esize) : β .* reshape(b, :)
biasarray(b::Number, esize, β) = repeat([β * b], esize)

actlayers[:Gemm] = function(params, weight::AbstractArray{T, N}, bias=false) where {T,N}
    act = get(params, :activation, identity)
    wt = Bool(get(params, :transB, 0)) ? permutedims : identity
    α = get(params, :alpha, 1)
    β = get(params, :beta, 1)

    weight = α .* wt(weight)
    bias = biasarray(bias, size(weight, 1), β)

    return Dense(weight, bias, act)
end
fluxlayertypes[:Gemm] = (pars...) -> FluxDense()

actlayers[:BatchNormalization] = function(params, γ, β, μ, σ²)
    λ = get(params, :activation, identity)
    ϵ = get(params, :epsilon, 1f-5)
    momentum = get(params, :momentum, 9f-1)

    return BatchNorm(λ, β, γ, μ, σ², ϵ, momentum, true, true, nothing, length(γ))
end
fluxlayertypes[:BatchNormalization] = (pars...) -> FluxBatchNorm()


default_Wb_Rb(Wh_WBh) = fill!(similar(Wh_WBh, (size(Wh_WBh, 2) * 2, size(Wh_WBh, 3))), 0)
default_init_h(Wb_Rb, sc) = fill!(similar(Wb_Rb, (size(Wb_Rb,1) ÷ sc, size(Wb_Rb,2))), 0)
# TODO when https://github.com/FluxML/Flux.jl/issues/1279 is resolved default_init_h(Wh_WBh, sc) = fill!(similar(Wh_WBh, (size(Wh_WBh, 2) ÷ sc, size(Wh_WBh, 3))), 0)

actlayers[:InstanceNormalization] = function(params, γ, β)
    λ = get(params, :activation, identity)
    ϵ = get(params, :epsilon, 1f-5)

    # ONNX InstanceNormalization does not support tracking μ and σ²
    momentum = NaN32
    μ = zeros(length(γ))
    σ² = ones(length(γ))

    return InstanceNorm(λ, β, γ, μ, σ², ϵ, momentum, true, false, nothing, length(γ))
end
fluxlayertypes[:InstanceNormalization] = (pars...) -> FluxInstanceNorm()

const SQUEEZED_RECURRENT_KEY = :ONNXNaiveNASflux_SQUEEZED_RECURRENT_KEY

fluxrecurrentlayers[:RNN] = function(params, Wi_WBi, Wh_WBh, Wb_Rb=default_Wb_Rb(Wh_WBh), seqlen=[], h3d = nothing)
    @assert size(Wi_WBi, 3) == 1 "Num directions must be 1! Bidirectional (num directions = 2) not supported!" # TODO: Add...
    if !isnothing(h3d)
        # We could probably create some wrapper struct for this if anyone ever needs it...
        @warn "Got initial hidden state for RNN. This can't be stored in Flux > 0.15 and will be ignored."
    end
    Wi,Wh,b = recurrent_arrays(FluxRnnCell(), Wi_WBi, Wh_WBh, Wb_Rb)
    act = rnnactfuns[Symbol(get(params, :activations, ["Tanh"])[])](1, params)
    cell = Flux.RNNCell(act, Wi, Wh, b)
    return Flux.RNN(cell)
end
fluxlayertypes[:RNN] = (pars...) -> FluxRnn()

_onnx_rnn_output1(h) = h
# Select last timestep
_onnx_rnn_output2(h::AbstractArray) = selectdim(h, 2, lastindex(h, 2))

_rnn_output_selection(i) = i === 1 ? _onnx_rnn_output1 :
                           i === 2 ? _onnx_rnn_output2 :
                           throw(ArgumentError("Unsupported RNN output: $i"))

layerfuns[:RNN] = function(params, args...)
    active_outputs = params[ACTIVE_OUTPUTS_ATTRIBUTE_KEY]
    selection = if length(active_outputs) == 1
        _rnn_output_selection(only(active_outputs))
    else
        ntuple(i -> _rnn_output_selection(active_outputs[i]), length(active_outputs))
    end
    paddims = haskey(params, SQUEEZED_RECURRENT_KEY) ? identity : l -> AddSingletonDim(3, l)
    layer -> paddims(OutputSelection(selection, layer))
end


fluxrecurrentlayers[:LSTM] = function(params, Wi_WBi, Wh_WBh, Wb_Rb=default_Wb_Rb(Wh_WBh), seqlen=[1], h3d = nothing, c3d = nothing, peep=nothing)
    @assert size(Wi_WBi, 3) == 1 "Num directions must be 1! Bidirectional (num directions = 2) not supported!" # TODO: Add...
    @assert isnothing(peep) "Peepholes not supported!" # Or?
    if !isnothing(h3d)
        # We could probably create some wrapper struct for this if anyone ever needs it...
        @warn "Got initial hidden state for LSTM. This can't be stored in Flux > 0.15 and will be ignored."
    end
    
    if !isnothing(c3d)
        # We could probably create some wrapper struct for this if anyone ever needs it...
        @warn "Got initial cell state for LSTM. This can't be stored in Flux > 0.15 and will be ignored."
    end

    Wi,Wh,b = recurrent_arrays(FluxLstmCell(), Wi_WBi, Wh_WBh, Wb_Rb)
    # Flux only supports default activation functions
    # We can only check that given values doesn't deviate
    supported = [:Sigmoid, :Tanh, :Tanh]
    acts = get(params, :activations, supported)
    @assert all(zip(supported, acts)) do (e,a)
        e == a
    end "Got unsupported activation function: $acts"

    # Should not be a problem when/if Flux adds this back as an optional output
    @assert 3 ∉ params[ACTIVE_OUTPUTS_ATTRIBUTE_KEY] "LSTM output 3 (the cell state) not implemnented!" 

    cell = Flux.LSTMCell(Wi, Wh, b)
    return Flux.LSTM(cell)
end
fluxlayertypes[:LSTM] = (pars...) -> FluxLstm()

_onnx_lstm_output1(h::AbstractArray) = h
_onnx_lstm_output2(h::AbstractArray) = selectdim(h, 2, lastindex(h, 2))
_onnx_lstm_output3(::AbstractArray) = throw(ArgumentError("LSTM output nr 3 (cell state) requires Flux.LSTM to output state. Please check you layer configuration!")) 

_onnx_lstm_output1((h, c)::NTuple{2, AbstractArray}) = h
_onnx_lstm_output2((h, c)::NTuple{2, AbstractArray}) = selectdim(h, 2, lastindex(h, 2))
_onnx_lstm_output3((h, c)::NTuple{2, AbstractArray}) = selectdim(c, 2, lastindex(c, 2))

_lstm_output_selection(i) = i === 1 ? _onnx_lstm_output1 :
                            i === 2 ? _onnx_lstm_output2 :
                            i === 3 ? _onnx_lstm_output3 :
                            throw(ArgumentError("Unsupported LSTM output: $i"))

layerfuns[:LSTM] = function(params, args...)
    active_outputs = params[ACTIVE_OUTPUTS_ATTRIBUTE_KEY]
    selection = if length(active_outputs) == 1
        # Can we be sure receiver does not want a single-element tuple here? No we can't :( :( :(
        _lstm_output_selection(only(active_outputs))
    else
        ntuple(i -> _lstm_output_selection(active_outputs[i]), length(active_outputs))
    end
    paddims = haskey(params, SQUEEZED_RECURRENT_KEY) ? identity : l -> AddSingletonDim(3, l)
    layer -> paddims(OutputSelection(selection, layer))
end


function recurrent_arrays(lt, Wi_WBi, Wh_WBh, Wb_Rb, h3ds...)
    # ONNX weights are on the form [num_directions, hidden_size, input_size] (where num_directions is 2 for bidirectional else 1)
    # Flux weights are of shape [hidden_size, input_size]
    # To spice things up a bit, all julia arrays are loaded in reverse order, i.e we get an array with the arrangement [input_size, hidden_size, num_directions].
    # First remove the num_directions dimension, then transpose into the correct shape
    hsize = size(Wh_WBh, 1)
    Wi = unflipweights(lt, permutedims(dropdims(Wi_WBi, dims=3)), hsize)
    Wh = unflipweights(lt, permutedims(dropdims(Wh_WBh, dims=3)), hsize)
    b = Wb_Rb isa Number ? Wb_Rb : dropdims(unflipweights(lt, sum(reshape(Wb_Rb, :, 2), dims=2), hsize),dims=2)
    return Wi, Wh, b, h3ds...
end

fluxlayers[:MaxPool] = function(params)
    _,k,p,s,_ = akpsd(params)
    return MaxPool(k, pad=p, stride=s)
end
fluxlayertypes[:MaxPool] = (pars...) -> FluxPoolLayer()


fluxlayers[:AveragePool] = function(params)
    _,k,p,s,_ = akpsd(params)
    return MeanPool(k, pad=p, stride=s)
end
fluxlayertypes[:AveragePool] = (pars...) -> FluxPoolLayer()


fluxlayers[:Dropout] = params -> Dropout(get(params, :ratio, 0.5))
fluxlayertypes[:Dropout] = (pars...) -> FluxDropOut()

invariantops[:GlobalAveragePool] = function(params)
    wrap = get(params, :wrap, identity)
    return wrap ∘ GlobalMeanPool()
end
fluxlayertypes[:GlobalAveragePool] = (pars...) -> FluxPoolLayer()

invariantops[:GlobalMaxPool] = function(params)
    wrap = get(params, :wrap, identity)
    return wrap ∘ GlobalMaxPool()
end
fluxlayertypes[:GlobalMaxPool] = (pars...) -> FluxPoolLayer()

"""
    Squeeze(dims)

Callable struct which performs `dropdims` on input using the provided `dims` where `dims` is compliant with the ONNX OP Squeeze (meaning it can be missing or use numpy indexing).
    
Mainly exists for pretty printing reaons though as its task can be performed by partially applied functions.

Designed to only be used when deserializing the `Squeeze` operation. 
"""
struct Squeeze{D}
    dims::D
end
(s::Squeeze)(x) = dropdims(x; dims=s.dims)
(s::Squeeze{Missing})(x) = dropdims(x; dims=Tuple(findall(i -> i == 1, size(x))))
(s::Squeeze{<:NumPyAxes})(x) = dropdims(x; dims=Tuple(numpy2fluxdim(s.dims, ndims(x))))

Base.show(io::IO, ::Squeeze{Missing}) = print(io, "Squeeze")
function Base.show(io::IO, s::Squeeze)
    print(io, "Squeeze(dims=")
    ioc = IOContext(io, :prefix => "[", :suffix=>"]") 
    show(ioc, s.dims)
    print(io, ")")
end


invariantops[:Squeeze] = function(params)
    np_axes = get(params, :axes, missing)
    dims = if !ismissing(np_axes)
        NumPyAxes(Tuple(np_axes))
    else
        np_axes
    end
    return Squeeze(dims)
end

"""
    Unsqueeze(dims)

Callable struct which performs `reshape` on input using the provided `dims` where `dims` is compliant with the ONNX OP `Unsqueeze` (meaning it can use numpy indexing).
    
Mainly exists for pretty printing reaons though as its task can be performed by partially applied functions.

Designed to only be used when deserializing the `Unsqueeze` operation. 
"""
struct Unsqueeze{D}
    dims::D
end

(u::Unsqueeze)(x) = unsqueeze_onnx(x, u.dims)

function Base.show(io::IO, s::Unsqueeze)
    print(io, "Unsqueeze(dims=")
    ioc = IOContext(io, :prefix => "[", :suffix=>"]") 
    show(ioc, s.dims)
    print(io, ")")
end

invariantops[:Unsqueeze] = function(params)
    haskey(params, :axes) || throw(ArgumentError("Must supply axes for Unsqueeze!"))
    return Unsqueeze(NumPyAxes(params[:axes]))
end

unsqueeze_onnx(x, np_axes) = reshape(x, insdims(size(x), np_axes))

struct Sorted{T}
    vals::T
    function Sorted(x)
        vals = issorted(x) ? x : sort(x)
        new{typeof(vals)}(vals) 
    end
end
Base.getindex(s::Sorted, args...) = Base.getindex(s.vals, args...)
Base.length(s::Sorted) = length(s.vals)

# Probably premature optimization: Allow for users to avoid numpy2fluxdim and sorting if they really want to.

function insdims(orgsize, np_axes::NumPyAxes; ndimsout=length(orgsize) + length(np_axes), kwargs...) 
    insdims(orgsize, numpy2fluxdim(np_axes, ndimsout); ndimsout, kwargs...)
end

insdims(orgsize, dimstoadd; kwargs...) = insdims(orgsize, Sorted(dimstoadd); kwargs...)
insdims(orgsize, dims::Sorted; ndimsout=length(orgsize) + length(dims), inssize=Returns(1)) = let 
    currax = Ref(1)
    dimoffs = Ref(0)
    ntuple(ndimsout) do i
        if currax[] <= length(dims) && dims[currax[]] == i
            ins = inssize(currax[])
            currax[] += 1
            dimoffs[] += 1
            ins
        else
            orgsize[i - dimoffs[]]
        end
    end
end


invariantops[:ReduceMean] = function(params)
    np_axes = get(params, :axes, missing)
    keepdims = Bool(get(params, :keepdims, 1))

    dimexp =
    if keepdims && ismissing(np_axes)
        # As mean returns a scalar when no dimensions are provided
        expanddims
    elseif !keepdims
        (out, x, dims) -> dropdims(out, dims=dims)
    else
        (out, x, dims) -> out
    end

    ismissing(np_axes) && return x -> dimexp(mean(x), x, missing)

    return function(x)
        dims = Tuple(numpy2fluxdim.(np_axes, ndims(x)))
        out = mean(x, dims=dims)
        return dimexp(out, x, dims)
    end
end
expanddims(out, x, dims) = fill(out, ntuple(i -> 1, ndims(x)))

invariantops[:Softmax] = params -> x -> onnxsoftmax(x; np_axis = get(params, :axis, 1))

function onnxsoftmax(x::AbstractArray{T, 2}; np_axis=1) where T
    dim = numpy2fluxdim(np_axis, 2)
    Flux.softmax(x; dims=dim)
end
function onnxsoftmax(x::AbstractArray{T, N}; np_axis=1) where {T,N}
    dim = numpy2fluxdim(np_axis, N)
    sz = size(x)
    reshape(Flux.softmax(reshape(x, prod(sz[1:dim]), :)), sz...)
end

pseudotransparentops[:Reshape] = function(params, shape)
    shape_t = Tuple(reverse(replace(shape, -1 => Colon())))
    return MeasureNout(Reshape(shape_t))
end

pseudotransparentops[:Flatten] = function(params)
    dim = -get(params,:axis, 1)
    return MeasureNout(Flatten(dim))
end


verts[:Input] = function(name, inputs, params; kwargs...)
    inshape = params[:size]
    ltype = params[:ltype]
    indims = length(inshape)

    insize = indims > 0 ? inshape[max(1, actdim(ltype))] : 1 # assume scalar
    return inputvertex(name, insize, ltype)
end

verts[:Add] = (name, inputs, params; kwargs...) -> elemwisevertex(name, inputs, params, +, 0; kwargs...)
verts[:Mul] = (name, inputs, params; kwargs...) -> elemwisevertex(name, inputs, params, *, 1; kwargs...)
verts[:Div] = (name, inputs, params; kwargs...) -> elemwisevertex(name, inputs, params, /, 1; kwargs...)

function elemwisevertex(name, inputs, params, op, id; traitdecoration=identity, layerfun=identity, kwargs...)
    c = reduce((c1,c2) -> op.(c1, c2), get(params, :Constant, id))
    c = length(c) == 1 ? c[] : c
    let cc = c
        opp, wrap = cc == id ? (op, layerfun) : (identity, f -> layerfun((x...) -> op.(cc, x...)))
        conf = VertexConf(traitdecoration = named(name) ∘ traitdecoration, outwrap = wrap, kwargs...)
        return NaiveNASlib.elemwise(opp, conf, inputs...)
    end
end


verts[:Concat] = function(name, inputs, params; traitdecoration=identity, layerfun=identity, kwargs...)
    dims = numpy2fluxdim(params[:axis], inputs[1])
    return conc(inputs..., dims=dims, traitdecoration = named(name) ∘ traitdecoration, outwrap=layerfun, kwargs...)
end

# Without parameters it needs its own type as well as constraints for propagation of size changes
matmul_op(name, inputs::AbstractVector{<:AbstractVertex}, params::AbstractDict; kwargs...) = throw(OpNotSupportedError("MatMul without parameter not supported!"))
matmul_op(name, inputs::AbstractVector{<:AbstractVertex}, params::AbstractDict, weight; kwargs...) = fluxvertex(name, Dense(weight, false, identity), inputs...; kwargs...)

verts[:MatMul] = matmul_op

function refresh()
    for (s, f) in actlayers
        fluxlayers[s] = f
    end

    for (s, f) in fluxrecurrentlayers
        fluxlayers[s] = f
    end

    for (s, f) in actfuns
        invariantops[s] = function(args...;kwargs...)
                                actfun = f(args...; kwargs...)
                                return x -> actfun.(x)
                            end
    end

    for (s, f) in fluxlayers
        verts[s] = function(name, inputs, args...; kwargs...) 
            # This is typically to select outputs, e.g. from recurrent layers
            kwargsnew = if s in keys(layerfuns)
                    mergewith(∘, Dict(:layerfun => layerfuns[s](args...)), Dict(kwargs))
            else
                kwargs
            end
            fluxvertex(name, f(args...), inputs...; kwargsnew...)
        end
    end

    for (s, f) in invariantops
        verts[s] = (name, inputs, args...;traitdecoration=identity, layerfun=identity, kwargs...) -> invariantvertex(layerfun(f(args...)), inputs...; traitdecoration = named(name) ∘ traitdecoration, kwargs...)
    end

    for (s,f) in pseudotransparentops
        verts[s] = function(name, inputs, args...;traitdecoration=identity, layerfun=identity, kwargs...)
            comp = f(args...)
            return absorbvertex(layerfun(comp), inputs...; traitdecoration = named(name) ∘ traitdecoration ∘ SizePseudoTransparent, kwargs...)
        end
    end

    for (s,f) in sources
        verts[s] = function(name, inputs, args...;kwargs...)
            @assert isempty(inputs) "Source of type $s got inputs $(inputs)!"
            return sourcevertex_with_outputs(f(args...), name)
        end
    end

    for s in keys(verts)
        get!(fluxlayertypes, s, (args...) -> missing)
    end

end

refresh()

list_supported_ops(io::IO=stdout) = foreach(ot -> println(io, ot), filter(ot -> ot != :Input, sort(collect(keys(verts)))))
