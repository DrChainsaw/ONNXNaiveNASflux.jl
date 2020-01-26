const actfuns = Dict{Symbol, Any}()
const rnnactfuns = Dict{Symbol, Any}() # Recurrent layers have activation functions as attributes and use different parameter names compared to their respective operations.
const actlayers = Dict{Symbol, Any}()
const fluxlayers = Dict{Symbol, Any}()
const invariantops = Dict{Symbol, Any}()
const verts = Dict{Symbol, Any}()

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


actfuns[:Relu] = params -> Flux.relu

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


_akpsd(params) = get(params, :activation, identity), get(params, :kernel_shape, 1), get(params, :pads, 0), get(params, :strides, 1), get(params, :dilations, 1)
akpsd(params) = a2t.(_akpsd(params))
a2t(x) = x
a2t(a::AbstractArray) = Tuple(a)

actlayers[:Conv] = function(params, weight::AbstractArray{T, N}, bias=zeros(T, size(weight, outdim(FluxConv{N-2}())))) where {T, N}
    a,_,p,s,d = akpsd(params)
    @assert get(params, :group, 1) == 1 "Group size not supported!" #Or?
    return Conv(weight, bias, a, pad=p, stride=s, dilation=d)
end

actlayers[:Gemm] = function(params, weight::AbstractArray{T, N}, bias=zeros(T, size(weight, outdim(FluxDense())))) where {T,N}
    act = get(params, :activation, identity)
    wt = Bool(get(params, :transB, 0)) ? permutedims : identity
    α = get(params, :alpha, 1)
    β = get(params, :beta, 1)
    return Dense(α * wt(weight), β * bias, act)
end

actlayers[:BatchNormalization] = function(params, γ, β, μ, σ²)
    λ = get(params, :activation, identity)
    ϵ = get(params, :epsilon, 1f-5)
    momentum = get(params, :momentum, 9f-1)

    return BatchNorm(λ, β, γ, μ, σ², ϵ, momentum)
end


default_Wb_Rb(Wh_WBh) = fill!(similar(Wh_WBh, (size(Wh_WBh, 2) * 2, size(Wh_WBh, 3))), 0)
default_init_h(Wb_Rb, sc) = fill!(similar(Wb_Rb, (size(Wb_Rb,1) ÷ sc, size(Wb_Rb,2))), 0)

fluxlayers[:RNN] = function(params, Wi_WBi, Wh_WBh, Wb_Rb=default_Wb_Rb(Wh_WBh), seqlen=[], h3d = default_init_h(Wb_Rb, 2))
    @assert size(Wi_WBi, 3) == 1 "Num directions must be 1! Bidirectional (num directions = 2) not supported!" # Or is it?

    Wi,Wh,b,h = recurrent_arrays(Wi_WBi, Wh_WBh, Wb_Rb, h3d)
    act = rnnactfuns[Symbol(get(params, :activations, "Tanh"))](1, params)
    cell = Flux.RNNCell(act, Wi, Wh, b, fill!(similar(b), 0))
    return Flux.Recur(cell, Flux.hidden(cell), h)
end

fluxlayers[:LSTM] = function(params, Wi_WBi, Wh_WBh, Wb_Rb=default_Wb_Rb(Wh_WBh), seqlen=[1], h3d = default_init_h(Wb_Rb, 8), c3d=default_init_h(Wb_Rb,8), peep=nothing)
    @assert size(Wi_WBi, 3) == 1 "Num directions must be 1! Bidirectional (num directions = 2) not supported!" # Or is it?
    @assert isnothing(peep) "Peepholes not supported!" # Or?
    Wi,Wh,b,h,c = recurrent_arrays(Wi_WBi, Wh_WBh, Wb_Rb, h3d, c3d)
    # Flux only supports default activation functions
    # We can only check that given values doesn't deviate
    supported = [:Sigmoid, :Tanh, :Tanh]
    acts = get(params, :activations, supported)
    @assert all(zip(supported, acts)) do (e,a)
        e == a
    end "Got unsupported activation function: $acts"

    # b, h and c must all be of the same type when creating a cell, but
    # it is actually Recur which has the state
    cell = Flux.LSTMCell(Wi, Wh, b, fill!(similar(b), 0), fill!(similar(b), 0))
    return Flux.Recur(cell, Flux.hidden(cell), (h, c))
end

function recurrent_arrays(Wi_WBi, Wh_WBh, Wb_Rb, h3ds...)
    # ONNX weights are on the form [num_directions, hidden_size, input_size] (where num_directions is 2 for bidirectional else 1)
    # Flux weights are of shape [hidden_size, input_size]
    # To spice things up a bit, all julia arrays are loaded in reverse order, i.e we get an array with the arrangement [input_size, hidden_size, num_directions].
    # First remove the num_directions dimension, then transpose into the correct shape
    Wi = permutedims(dropdims(Wi_WBi, dims=3))
    Wh = permutedims(dropdims(Wh_WBh, dims=3))
    b = dropdims(sum(reshape(Wb_Rb, :, 2), dims=2),dims=2)
    hs = (dropdims(h, dims=ndims(h)) for h in h3ds)
    return Wi, Wh, b, hs...
end

fluxlayers[:MaxPool] = function(params)
    _,k,p,s,_ = akpsd(params)
    return MaxPool(k, pad=p, stride=s)
end

fluxlayers[:AveragePool] = function(params)
    _,k,p,s,_ = akpsd(params)
    return MeanPool(k, pad=p, stride=s)
end

fluxlayers[:Dropout] = params -> Dropout(get(params, :ratio, 0.5))

invariantops[:GlobalAveragePool] = function(params)
    wrap = get(params, :wrap, identity)
    return x -> globalmeanpool(x, wrap)
end
function globalmeanpool(x::AbstractArray{T,N}, wrap) where T where N
    wrap(MeanPool(size(x)[1:N-2])(x))
end

invariantops[:Reshape] = function(params, shape)
    shape_t = Tuple(reverse(shape))
    any(s -> s == 0 || s == -1, shape_t) && return x -> reshape_keepshape(x, shape_t)
    return x -> reshape(x, shape_t)
end
function reshape_keepshape(x, shape)
    offs = ndims(x) - length(shape)
    newshape = map(enumerate(shape)) do (ind, new)
        new == -1 && return Colon()
        new == 0 && return size(x, ind+offs)
        return new
    end
    return reshape(x, newshape...)
end


invariantops[:Squeeze] = function(params)
    np_axes = get(params, :axes, missing)
    dimfun = ismissing(np_axes) ? x -> Tuple(findall(i -> i == 1, size(x))) : x -> Tuple(numpy2fluxdim.(np_axes, ndims(x)))
    return x -> dropdims(x, dims=dimfun(x))
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

verts[:Input] = function(name, inputs, params; kwargs...)
    inshape = reverse(params[:size])
    indims = length(inshape)
    insize = indims > 0 ? inshape[max(1, actdim(indims))] : 1 # assume scalar
    return inputvertex(name, insize, guess_layertype(indims))
end

verts[:Add] = function(name, inputs, params; traitdecoration=identity, layerfun=identity, kwargs...)
    conf = VertexConf(traitdecoration = t -> NamedTrait(traitdecoration(t), name), outwrap = layerfun, kwargs...)
    return NaiveNASlib.elemwise(+, conf, inputs...)
end

verts[:Concat] =  function(name, inputs, params; traitdecoration=identity, layerfun=identity, kwargs...)
    dims = numpy2fluxdim(params[:axis], inputs[1])
    return conc(inputs..., dims=dims, traitdecoration = t -> NamedTrait(traitdecoration(t), name), outwrap=layerfun, kwargs...)
end


function refresh()
    for (s, f) in actlayers
        fluxlayers[s] = f
    end

    for (s, f) in actfuns
        invariantops[s] = f
    end

    for (s, f) in fluxlayers
        verts[s] = (name, inputs, args...;kwargs...) -> mutable(name, f(args...), inputs...; kwargs...)
    end

    for (s, f) in invariantops
        verts[s] = (name, inputs, args...;traitdecoration=identity, layerfun=identity, kwargs...) -> invariantvertex(layerfun(f(args...)), inputs...; traitdecoration = t -> NamedTrait(traitdecoration(t), name), kwargs...)
    end
end

refresh()

list_supported_ops(io::IO=stdout) = foreach(ot -> println(io, ot), filter(ot -> ot != :Input, sort(collect(keys(verts)))))
